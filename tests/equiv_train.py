#!/usr/bin/env python
"""Step-0 planner-request check for the training benchmarks (RoboTwin-50 / Guava / LIBERO-PRO).

A local stub OpenAI-compatible server (127.0.0.1, random port) records every POST /chat/completions body and answers with
a give_up tool call, so the episode ends after the first request; SAM3 endpoints (/seg, /segment) are stubbed by the same
server with empty results. Python's and NumPy's global RNGs are seeded with 0 before the harness starts (RoboTwin samples
the instruction wording with them). The recorded step-0 request is compared with tests/train_ref_requests.json, the
digests of the request sent by the code used for the paper on the same episode:
  "full" = exact body bytes (includes the rendered images; RoboTwin's renderer is not bit-exact between runs, so only
           Guava / LIBERO are expected to match here),
  "text" = body with every base64 image replaced by a placeholder (SYSTEM, task memory, user text, sampling parameters);
           for RoboTwin a text difference that disappears once the measured gripper positions of the `Arms:` line are
           masked is renderer noise (0.1 cm) and passes.

The benchmark runtime (conda env, renderer variables, ROBODAWN_REPO / GUAVA_REPO / MOTORMIND_REPO) must be set in the
calling shell, see docs/benchmarks/<bench>.md. GPU use: one short render.

  python tests/equiv_train.py --bench robotwin|guava|libero [--python PY] [--work DIR] [--ref-script OTHER_ENTRY.py]
Exit code 0 = the text digest matched (and every byte comparison that was run matched)."""
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF_FILE = Path(__file__).resolve().parent / "train_ref_requests.json"
REPLY = '</think>\n\n<tool_call>\n{"name": "give_up", "arguments": {}}\n</tool_call>'
SEEDED = ("import random, runpy, sys; import numpy as np; random.seed(0); np.random.seed(0); "
          "s = sys.argv[1]; sys.argv = sys.argv[1:]; sys.path.insert(0, __import__('os').path.dirname(s)); "
          "runpy.run_path(s, run_name='__main__')")


def cases(url, sam, out):
    return {
        "robotwin": (["robotwin", "--task", "adjust_bottle", "--episode", "0", "--lane", "equiv", "--model", "qwen", "--url", url,
                      "--thinking", "1"], {"PERC_SAM3": sam, "TINYASTRAL_OUT": out}, None),
        "guava": (["guava", "--gen", "default", "--sam3", sam, "--task", "pick_up_carrot", "--seed-base", "42", "--episodes", "1",
                   "--offset", "0", "--out", out, "--model", "qwen", "--url", url, "--thinking", "1"], {"GUAVA_SAM3_URL": sam}, "guava"),
        "libero": (["libero", "--gen", "default", "--task", "57", "--seed", "1", "--out", out, "--url", url, "--model", "qwen",
                    "--thinking", "1", "--xml-calls", "1"], {"LIBERO_SAM3_URL": sam, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}, None),
    }


class Stub:
    def __init__(self):
        self.bodies, self.label = {}, None
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path.endswith("/chat/completions"):
                    stub.bodies.setdefault(stub.label, []).append(body)
                    out = {"choices": [{"message": {"role": "assistant", "content": REPLY}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
                elif self.path.endswith("/segment"):
                    out = {"masks": []}
                else:
                    try:
                        prompts = json.loads(body).get("prompts") or []
                    except Exception:  # noqa: BLE001
                        prompts = []
                    out = {"results": {p: [] for p in prompts}}
                b = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def noimg(body):
    return re.sub(rb"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", b"<IMG>", body)


def run(stub, label, py, script, bench, work):
    stub.label = label
    url, sam = f"http://127.0.0.1:{stub.port}/v1", f"http://127.0.0.1:{stub.port}"
    args, extra, cwd_kind = cases(url, sam, str(work / f"out_{label}"))[bench]
    env = {**os.environ, **extra, "PYTHONDONTWRITEBYTECODE": "1"}
    for k in ("HARNESS_MEMORY", "HARNESS_NO_OBJECT_LIST", "HARNESS_BENCH", "HARNESS_GEN"):
        env.pop(k, None)
    cwd = str(work)
    if cwd_kind == "guava":
        sys.path.insert(0, str(REPO / "harness"))
        import paths
        cwd = str(paths.GUAVA)
        env["PYTHONPATH"] = os.pathsep.join(x for x in (cwd, env.get("PYTHONPATH")) if x)
    log = work / f"{label}.log"
    with open(log, "w") as f:
        rc = subprocess.call([py, "-c", SEEDED, str(script)] + args, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, timeout=1800)
    bodies = stub.bodies.get(label, [])
    print(f"[{label}] rc={rc} requests={len(bodies)} log={log}")
    if not bodies:
        print(open(log).read()[-3000:])
        return None
    return bodies[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["robotwin", "guava", "libero"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--ref-script", default=None, help="another entry script with the same CLI to compare against")
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    work = Path(a.work or tempfile.mkdtemp(prefix=f"equiv_{a.bench}_"))
    work.mkdir(parents=True, exist_ok=True)
    stub = Stub()
    ok = True
    rel = run(stub, "release", a.python, REPO / "harness/run.py", a.bench, work)
    if rel is None:
        print("FAIL: no request was sent")
        return 1
    (work / "release_request.json").write_bytes(rel)
    if a.ref_script:
        ref = run(stub, "reference", a.python, Path(a.ref_script), a.bench, work)
        same_text = ref is not None and noimg(ref) == noimg(rel)
        print(f"reference vs release: bytes {'IDENTICAL' if ref == rel else 'differ'}; text-only {'IDENTICAL' if same_text else 'DIFFER'}")
        ok = ok and same_text
    d = json.loads(REF_FILE.read_text())[a.bench]
    full = hashlib.sha256(rel).hexdigest() == d["sha256_full"]
    text = hashlib.sha256(noimg(rel)).hexdigest() == d["sha256_text"]
    print(f"vs paper request: full-bytes {'MATCH' if full else 'differ'}; text-only {'MATCH' if text else 'DIFFER'}")
    if not text and d.get("sha256_text_masked_pos"):
        # RoboTwin's renderer is not bit-exact: the measured gripper positions in the `Arms:` line can differ by 0.1 cm
        pos = re.sub(rb"gripper at \[[-0-9., ]+\]", b"gripper at [<POS>]", noimg(rel))
        text = hashlib.sha256(pos).hexdigest() == d["sha256_text_masked_pos"]
        print(f"  text with the measured gripper positions masked: {'MATCH (renderer noise only, OK)' if text else 'DIFFER'}")
    return 0 if (ok and text) else 1


if __name__ == "__main__":
    sys.exit(main())
