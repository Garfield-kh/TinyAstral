#!/usr/bin/env python
"""Multi-step replay check: drive two harness trees with the same scripted model replies and compare every step.

tests/equiv_train.py and tests/equiv_zeroshot.py only compare the step-0 request. This test replays a whole episode:
a local stub server (127.0.0.1, random port) answers POST /chat/completions with the next reply of a fixed list
(tests/replay_cases.json: the replies of paper 4B-student episodes, plus hand-written sequences that hit the error
paths); once the list is used up it answers give_up. The same reply list drives a reference tree (--old-root, e.g. a
`git worktree` of an older commit) and this tree, on the same episode (task, seed; Python / NumPy RNGs seeded with 0).

SAM3 is record / replay: during the reference run every /seg and /segment request is forwarded to the real server
(--sam-seg / --sam-segment; without them the stub answers with empty results) and its answer is recorded; during the
release run the i-th SAM3 request gets the i-th recorded answer (its prompts are checked against the recorded ones),
so both trees see the same perception even where the renderer is not bit-exact.

Compared per case:
  * every planner request, step by step: exact body bytes, identical images, and the "text" view that decides
    pass / fail (base64 images replaced by a placeholder; for RoboTwin also the measured gripper positions of the
    `Arms:` line, see docs/consistency.md item 8; LIBERO's visible wall clock `Time used: N/` masked). Renders are
    not bit-exact even between two runs of the same tree (RoboTwin; MuJoCo EGL on a shared GPU), so images are
    reported, not gated;
  * the episode output files (episode.json / result.json / results.json) with timing fields removed and the same
    masks applied to their strings (obs_text carries the LIBERO wall clock).

--ctrl also runs the reference tree a second time (same replies, same recorded SAM3 answers) and compares it with the
first reference run. A difference that the reference tree also shows against itself is simulator / renderer
nondeterminism, not a change of the harness: such a case is reported as NOISE (exit code 0); a difference that the
control does not show is FAIL.

The benchmark runtime must be set in the calling shell (see docs/benchmarks/<bench>.md and tests/equiv_train.py).

  python tests/equiv_replay.py --bench guava --old-root OLD_TREE [--case NAME] [--python PY] [--work DIR]
                               [--sam-seg http://127.0.0.1:8130] [--sam-segment http://127.0.0.1:8314] [--pythonpath P] [--ctrl]
Exit code 0 = every case is PASS (identical text-view requests at every step, identical episode fields and outcome) or
NOISE (every difference also appears between two runs of the reference tree)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASES = Path(__file__).resolve().parent / "replay_cases.json"
GIVE_UP = '</think>\n\n<tool_call>\n{"name": "give_up", "arguments": {"reason": "replay finished"}}\n</tool_call>'
SEEDED = ("import random, runpy, sys; import numpy as np; random.seed(0); np.random.seed(0); "
          "s = sys.argv[1]; sys.argv = sys.argv[1:]; sys.path.insert(0, __import__('os').path.dirname(s)); "
          "runpy.run_path(s, run_name='__main__')")
ZS_DRIVER = {"simplerenv": "zeroshot/simplerenv/run_simplerenv.py", "vlabench": "zeroshot/vlabench/run_vlabench.py",
             "embodiedbench": "zeroshot/embodiedbench/run_eb.py"}
VOLATILE = {"seconds", "t_start", "t_end", "llm_s", "usage", "response_meta", "first_success", "wall_s", "time", "elapsed",
            "harness", "harness_sha256", "timestamp", "date", "host", "perceive_s", "first_success_s", "seg_s"}


def train_entry(root):
    old = root / "hstar" / "harness_main.py"                 # pre-restructure layout
    return old if old.exists() else root / "harness" / "run.py"


def case_cmd(bench, root, args, url, sam, out):
    """(script, argv, extra env) of one episode of a case."""
    if bench == "robotwin":
        return train_entry(root), ["robotwin", *args, "--lane", "replay", "--model", "qwen", "--url", url, "--thinking", "1"], \
            {"PERC_SAM3": sam, "TINYASTRAL_OUT": out}
    if bench == "guava":
        return train_entry(root), ["guava", "--gen", "default", "--sam3", sam, *args, "--out", out, "--model", "qwen", "--url", url,
                                   "--thinking", "1"], {"GUAVA_SAM3_URL": sam}
    if bench == "libero":
        return train_entry(root), ["libero", "--gen", "default", *args, "--out", out, "--url", url, "--model", "qwen", "--thinking", "1",
                                   "--xml-calls", "1"], {"LIBERO_SAM3_URL": sam, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
    common = {"simplerenv": ["--url", url, "--model", "qwen", "--thinking", "1", "--sam3", sam, "--out", out, "--gen", "default"],
              "vlabench": ["--out", out, "--url", url, "--model", "qwen", "--thinking", "1", "--xml-calls", "1", "--sam3", sam,
                           "--gen", "default"],
              "embodiedbench": ["--url", url, "--model", "qwen", "--sam3", sam, "--tag", "replay", "--out-root", out, "--thinking", "1",
                                "--gen", "default"]}[bench]
    return root / ZS_DRIVER[bench], [*args, *common], {}


def sam_key(path, body):
    """the non-image part of a SAM3 request (prompts, top_k, ...)."""
    try:
        j = json.loads(body)
    except Exception:  # noqa: BLE001
        return path
    return path + " " + json.dumps({k: v for k, v in j.items() if "image" not in k}, sort_keys=True)


class Stub:
    def __init__(self, sam_seg=None, sam_segment=None):
        self.label, self.replies, self.ref_label = None, [], None
        self.chat, self.sam = {}, {}             # label -> [request body], label -> [(key, response bytes)]
        self.sam_mismatch = {}
        self.up = {"/seg": sam_seg, "/segment": sam_segment}
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, b):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                self._send(b"{}")

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path.endswith("/chat/completions"):
                    self._send(stub.answer_chat(body))
                else:
                    self._send(stub.answer_sam(self.path, body))

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        self.lock = threading.Lock()
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def answer_chat(self, body):
        with self.lock:
            got = self.chat.setdefault(self.label, [])
            got.append(body)
            i = len(got) - 1
        text = self.replies[i] if i < len(self.replies) else GIVE_UP
        return json.dumps({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 0, "completion_tokens": 0}}).encode()

    def _forward(self, path, body):
        ep = "/segment" if path.endswith("/segment") else "/seg"
        up = self.up.get(ep)
        if up:
            rq = urllib.request.Request(up.rstrip("/") + ep, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=600) as r:
                return r.read()
        if ep == "/segment":
            return json.dumps({"masks": []}).encode()
        try:
            prompts = json.loads(body).get("prompts") or []
        except Exception:  # noqa: BLE001
            prompts = []
        return json.dumps({"results": {p: [] for p in prompts}}).encode()

    def answer_sam(self, path, body):
        key = sam_key(path, body)
        with self.lock:
            mine = self.sam.setdefault(self.label, [])
            i = len(mine)
            ref = self.sam.get(self.ref_label) if self.ref_label and self.ref_label != self.label else None
        if ref is not None and i < len(ref) and ref[i][0] == key:
            out = ref[i][1]
        else:
            if ref is not None:
                self.sam_mismatch.setdefault(self.label, []).append(
                    {"i": i, "key": key[:300], "recorded": ref[i][0][:300] if i < len(ref) else None})
            out = self._forward(path, body)
        with self.lock:
            mine.append((key, out))
        return out


IMG = re.compile(rb"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+|<IMG sha=[0-9a-f]+>")   # 2nd form: saved requests_*.jsonl


def img_view(body):
    return IMG.sub(lambda m: b"<IMG sha=" + hashlib.sha256(m.group(0)).hexdigest()[:12].encode() + b">", body)


def mask(t, bench):
    """bytes or str: the masks shared by the request text view and the episode strings."""
    b = isinstance(t, bytes)
    t = t if b else t.encode()
    if bench == "robotwin":                        # measured gripper positions jitter 0.1 cm
        t = re.sub(rb"gripper at \[[-0-9., ]+\]", b"gripper at [<POS>]", t)
    t = re.sub(rb"Time used: \d+/", b"Time used: <T>/", t)   # LIBERO's visible wall clock
    return t if b else t.decode()


def text_view(body, bench):
    """images replaced by a placeholder: renders are not bit-exact between two runs of the SAME tree (RoboTwin, and the
    MuJoCo EGL renders of Guava / LIBERO-PRO when the GPU is shared), so images are reported (img_same) but not gated."""
    return mask(IMG.sub(b"<IMG>", body), bench)


def first_diff(a, b, ctx=160):
    n = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return f"at byte {n} (len {len(a)} vs {len(b)}):\n      old: {a[max(0, n - ctx):n + ctx]!r}\n      new: {b[max(0, n - ctx):n + ctx]!r}"


def strip(x, bench):
    if isinstance(x, dict):
        return {k: strip(v, bench) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [strip(v, bench) for v in x]
    if isinstance(x, str):
        return mask(x, bench)
    return x


def diff_paths(a, b, p="", out=None):
    out = [] if out is None else out
    if type(a) is not type(b):
        out.append(f"{p}: {str(a)[:120]} != {str(b)[:120]}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{p}/{k}: only in {'new' if k not in a else 'old'}")
            else:
                diff_paths(a[k], b[k], f"{p}/{k}", out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{p}: list length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{p}[{i}]", out)
    elif a != b:
        out.append(f"{p}: {str(a)[:120]!r} != {str(b)[:120]!r}")
    return out


def episode_files(out):
    return {str(f.relative_to(out)): f for f in sorted(Path(out).rglob("*.json"))
            if f.name in ("episode.json", "result.json", "results.json")}


def run_one(stub, label, py, root, case, work, pythonpath):
    stub.label = label
    url, sam = f"http://127.0.0.1:{stub.port}/v1", f"http://127.0.0.1:{stub.port}"
    out = work / f"out_{label}"
    script, argv, extra = case_cmd(case["bench"], root, case["args"], url, sam, str(out))
    env = {**os.environ, **extra, "PYTHONDONTWRITEBYTECODE": "1"}
    for k in ("HARNESS_MEMORY", "HARNESS_NO_OBJECT_LIST", "HARNESS_BENCH", "HARNESS_GEN"):
        env.pop(k, None)
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    cwd = str(work)
    if case["bench"] == "guava":                   # Guava episodes run with cwd = GUAVA_REPO
        cwd = os.environ["GUAVA_REPO"]
        env["PYTHONPATH"] = os.pathsep.join(x for x in (cwd, env.get("PYTHONPATH")) if x)
    log = work / f"{label}.log"
    with open(log, "w") as f:
        rc = subprocess.call([py, "-c", SEEDED, str(script)] + argv, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, timeout=5400)
    return rc, stub.chat.get(label, []), out, log


def saved_run(stub, label, py, root, case, work, pythonpath):
    """--rejudge: the requests and outputs a previous run of this script left in WORK (no episode is run)."""
    f = work / f"requests_{label}.jsonl"
    if not f.exists():
        raise SystemExit(f"--rejudge: {f} missing")
    reqs = [ln for ln in f.read_bytes().split(b"\n") if ln]
    stub.chat[label] = reqs
    prev = json.loads((work / "summary.json").read_text()) if (work / "summary.json").exists() else {}
    if label == "new":
        stub.sam_mismatch["new"] = prev.get("sam_mismatch", [])
    return 0, reqs, work / f"out_{label}", work / f"{label}.log"


def compare_runs(req_o, req_n, out_o, out_n, bench):
    """step-by-step comparison of two runs: (n, same_bytes, same_text, same_img, text-diff steps, step lines, episode diffs)."""
    n = max(len(req_o), len(req_n))
    same_bytes = same_text = same_img = 0
    diff_steps, lines = [], []
    for i in range(n):
        if i >= len(req_o) or i >= len(req_n):
            diff_steps.append(i)
            lines.append(f"    step {i}: only in {'old' if i < len(req_o) else 'new'}")
            continue
        if req_o[i] == req_n[i]:
            same_bytes += 1
        same_img += [m.group(0) for m in IMG.finditer(req_o[i])] == [m.group(0) for m in IMG.finditer(req_n[i])]
        to, tn = text_view(req_o[i], bench), text_view(req_n[i], bench)
        if to == tn:
            same_text += 1
        else:
            diff_steps.append(i)
            lines.append(f"    step {i}: text differs {first_diff(to, tn)}")
    fo, fn = episode_files(out_o), episode_files(out_n)
    ep_diffs = []
    for rel in sorted(set(fo) | set(fn)):
        if rel not in fo or rel not in fn:
            ep_diffs.append(f"{rel}: only in {'old' if rel in fo else 'new'}")
            continue
        d = diff_paths(strip(json.loads(fo[rel].read_text()), bench), strip(json.loads(fn[rel].read_text()), bench))
        ep_diffs += [f"{rel}{x}" for x in d]
    return n, same_bytes, same_text, same_img, diff_steps, lines, ep_diffs


def outcome(out):
    for rel, f in episode_files(out).items():
        j = json.loads(f.read_text())
        if isinstance(j, dict) and "episodes" in j:
            j = j["episodes"][0]
        if isinstance(j, dict):
            keys = [k for k in ("success", "success_strict", "success_final", "success_any", "success_15", "end_reason",
                                "finished_reason", "score") if k in j]
            if keys:
                return {k: j[k] for k in keys}
    return None


def ep_key(d):
    """an episode diff without its values (path only), to match a diff of the release run with one of the control."""
    return d.split(":", 1)[0]


def compare_case(case, a, work):
    stub = Stub(a.sam_seg, a.sam_segment)
    stub.replies, stub.ref_label = case["replies"], "old"
    bench = case["bench"]
    run = saved_run if a.rejudge else run_one
    rc_o, req_o, out_o, log_o = run(stub, "old", a.python, Path(a.old_root), case, work, a.pythonpath)
    rc_n, req_n, out_n, log_n = run(stub, "new", a.python, Path(a.new_root), case, work, a.pythonpath)
    n, same_bytes, same_text, same_img, dsteps, lines, ep_diffs = compare_runs(req_o, req_n, out_o, out_n, bench)
    oo, on = outcome(out_o), outcome(out_n)
    exact = rc_o == 0 and rc_n == 0 and same_text == n and not ep_diffs and n > 0 and oo == on
    ctrl = None
    if a.ctrl and (not a.rejudge or (work / "requests_ctrl.jsonl").exists()):
        rc_c, req_c, out_c, log_c = run(stub, "ctrl", a.python, Path(a.old_root), case, work, a.pythonpath)
        cn, _, c_text, c_img, c_steps, c_lines, c_ep = compare_runs(req_o, req_c, out_o, out_c, bench)
        ctrl = {"rc": rc_c, "steps": cn, "text_same": c_text, "img_same": c_img, "text_diff_steps": c_steps,
                "episode_diffs": c_ep, "step_diffs": c_lines, "outcome": outcome(out_c),
                "sam_mismatch": stub.sam_mismatch.get("ctrl", [])}
    verdict = "PASS" if exact else "FAIL"
    explained = []
    if not exact and ctrl and rc_o == 0 and rc_n == 0 and n > 0 and ctrl["rc"] == 0:
        c_first = min(ctrl["text_diff_steps"], default=None)
        c_keys = {ep_key(x) for x in ctrl["episode_diffs"]}
        # a release-run difference is explained when the control already differs at or before that step, or the same
        # episode field also differs in the control
        steps_ok = all(c_first is not None and c_first <= s for s in dsteps)
        n_first = min(dsteps, default=None)
        ep_ok = all(ep_key(x) in c_keys or (c_first is not None and n_first is not None and c_first <= n_first)
                    for x in ep_diffs)
        out_ok = oo == on or (c_first is not None and n_first is not None and c_first <= n_first)
        if steps_ok and ep_ok and out_ok:
            verdict = "NOISE"
            explained.append(f"control (reference tree vs itself) first differs at step {c_first}; release run first differs "
                             f"at step {n_first}")
    print(f"[{bench}:{case['name']}] {verdict}  requests old/new {len(req_o)}/{len(req_n)} (replies {len(case['replies'])}); "
          f"text-identical {same_text}/{n}, byte-identical {same_bytes}/{n}, images identical {same_img}/{n}; outcome old {oo} new {on}; "
          f"episode-field diffs {len(ep_diffs)}; rc {rc_o}/{rc_n}; sam3 replay mismatches {len(stub.sam_mismatch.get('new', []))}")
    if ctrl:
        print(f"    control old/old: text-identical {ctrl['text_same']}/{ctrl['steps']}, images identical {ctrl['img_same']}/{ctrl['steps']}, "
              f"text-diff steps {ctrl['text_diff_steps']}, episode-field diffs {len(ctrl['episode_diffs'])}, outcome {ctrl['outcome']}, "
              f"rc {ctrl['rc']}, sam3 replay mismatches {len(ctrl['sam_mismatch'])}")
        if same_img < n and ctrl["img_same"] < ctrl["steps"]:
            print("    images: the reference tree also renders different bytes against itself -> renderer nondeterminism")
    for x in explained:
        print("    " + x)
    for x in lines[:6]:
        print(x)
    for x in ep_diffs[:12]:
        print("    episode:", x)
    if ctrl:
        for x in ctrl["step_diffs"][:3]:
            print("    control:", x.strip())
        for x in ctrl["episode_diffs"][:6]:
            print("    control episode:", x)
    for x in stub.sam_mismatch.get("new", [])[:3]:
        print("    sam3:", x)
    if rc_o or rc_n or n == 0:
        print(f"    logs: {log_o} {log_n}")
    (work / "summary.json").write_text(json.dumps({"case": case["name"], "bench": bench, "verdict": verdict, "ok": verdict != "FAIL",
                                                   "steps": n, "text_same": same_text, "bytes_same": same_bytes, "img_same": same_img,
                                                   "outcome_old": oo, "outcome_new": on, "episode_diffs": ep_diffs, "step_diffs": lines,
                                                   "sam_mismatch": stub.sam_mismatch.get("new", []), "control": ctrl}, indent=1, default=str))
    reqs = [("old", req_o), ("new", req_n)] + ([("ctrl", stub.chat.get("ctrl", []))] if ctrl else [])
    for lbl, rq in [] if a.rejudge else reqs:
        with open(work / f"requests_{lbl}.jsonl", "wb") as f:
            for r in rq:
                f.write(img_view(r) + b"\n")
    stub.srv.shutdown()
    return verdict != "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["robotwin", "guava", "libero", "simplerenv", "vlabench", "embodiedbench"])
    ap.add_argument("--old-root", required=True, help="reference tree (e.g. git worktree of an older commit)")
    ap.add_argument("--new-root", default=str(REPO))
    ap.add_argument("--case", default="all", help="case name in replay_cases.json, or 'all' for every case of --bench")
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pythonpath", default=None, help="PYTHONPATH for both runs (default: inherited)")
    ap.add_argument("--sam-seg", default=None, help="real SAM3 /seg server (perception/sam3_srv.py) for the reference run")
    ap.add_argument("--sam-segment", default=None, help="real Guava SAM3 /segment server for the reference run")
    ap.add_argument("--work", default=None)
    ap.add_argument("--rejudge", action="store_true", help="re-judge the requests / outputs already in --work (no episode is run)")
    ap.add_argument("--ctrl", action="store_true", help="also run the reference tree twice, to tell nondeterminism from regressions")
    a = ap.parse_args()
    cases = [c for c in json.loads(Path(a.cases).read_text()) if c["bench"] == a.bench and a.case in ("all", c["name"])]
    if not cases:
        raise SystemExit(f"no case {a.case} for {a.bench}")
    work = Path(a.work or tempfile.mkdtemp(prefix=f"replay_{a.bench}_"))
    ok = True
    for c in cases:
        w = work / c["name"]
        w.mkdir(parents=True, exist_ok=True)
        ok = compare_case(c, a, w) and ok
    print("ALL PASS (or NOISE)" if ok else "SOME CASES FAILED", f"(work dir {work})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
