#!/usr/bin/env python
"""Byte-level check of the step-0 planner request of the zero-shot drivers (SimplerEnv / VLABench / EB-Manipulation).

A local stub OpenAI-compatible server (127.0.0.1, random port) records the first POST /chat/completions body and answers
with a give_up tool call, so the episode ends right after the first request; SAM3 endpoints are stubbed by the same server
(empty results). The release driver runs one fixed episode with the paper settings; the recorded request is then
  * compared byte for byte with the request of a reference driver (--ref-driver, e.g. an older copy of the driver),
    run on the same episode with the same stub, and/or
  * compared with the SHA-256 digests in tests/zeroshot_ref_requests.json (the requests of the paper's code):
    "full" = exact body bytes (includes the rendered JPEG, so it can depend on the GPU / renderer build),
    "text" = body with every base64 image replaced by a placeholder (renderer independent: SYSTEM, user text,
    sampling parameters, message layout).

The benchmark runtime (conda env, renderer variables, Xvfb for EB, upstream repo variables) must already be set in the
calling shell, see docs/benchmarks/<bench>.md. GPU use: one short render (SimplerEnv: SAPIEN; VLABench: MuJoCo EGL).

  python tests/equiv_zeroshot.py --bench simplerenv [--case default|alt] [--python PY] [--ref-driver OLD/run_simplerenv.py --ref-pythonpath P]
                                 [--rel-pythonpath P] [--work DIR] [--save-ref]
Exit code 0 = every comparison that was run matched."""
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
REF_FILE = Path(__file__).resolve().parent / "zeroshot_ref_requests.json"
REPLY = '<tool_call>\n{"name": "give_up", "arguments": {"reason": "stub"}}\n</tool_call>'

# per bench: driver + episode cases (the "default" case is the one whose digests are stored in zeroshot_ref_requests.json)
BENCHES = {
    "simplerenv": {"driver": REPO / "zeroshot/simplerenv/run_simplerenv.py",
                   "cases": {"default": ["--key", "coke_h", "--idx", "0"],             # Google Robot
                             "alt": ["--key", "br_carrot", "--idx", "3"]},            # WidowX
                   "args": lambda url, sam, out: ["--url", url, "--model", "qwen", "--thinking", "1",
                                                  "--sam3", sam, "--out", out, "--gen", "default"]},
    "vlabench": {"driver": REPO / "zeroshot/vlabench/run_vlabench.py",
                 "cases": {"default": ["--track", "track_1_in_distribution", "--task", "select_painting", "--episodes", "0"],
                           "alt": ["--track", "track_3_common_sense", "--task", "select_fruit", "--episodes", "0"]},
                 "args": lambda url, sam, out: ["--out", out, "--url", url, "--model", "qwen", "--thinking", "1", "--xml-calls", "1",
                                                "--sam3", sam, "--gen", "default"]},
    "embodiedbench": {"driver": REPO / "zeroshot/embodiedbench/run_eb.py",
                      "cases": {"default": ["--eval-set", "base", "--indexes", "0"],
                                "alt": ["--eval-set", "visual", "--indexes", "2"]},
                      "args": lambda url, sam, out: ["--url", url, "--model", "qwen", "--sam3", sam, "--tag", "equiv",
                                                     "--out-root", out, "--thinking", "1", "--gen", "default"]},
}


class Stub:
    """records POST bodies per run label; answers chat with give_up, SAM3 (/seg, /segment) with empty results."""

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
                    out = {"choices": [{"message": {"content": REPLY}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
                elif self.path.endswith("/segment"):
                    out = {"masks": []}
                else:
                    out = {"results": {}}
                b = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def text_only(body):
    """request body with base64 images replaced by their length + SHA-256 prefix (renderer-independent view)."""
    return re.sub(rb"data:image/jpeg;base64,[A-Za-z0-9+/=]+",
                  lambda m: b"<IMG len=%d sha=%s>" % (len(m.group(0)), hashlib.sha256(m.group(0)).hexdigest()[:12].encode()), body)


def text_only_noimg(body):
    return re.sub(rb"data:image/jpeg;base64,[A-Za-z0-9+/=]+", b"<IMG>", body)


def first_diff(a, b, ctx=120):
    n = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return f"first difference at byte {n} (len {len(a)} vs {len(b)}):\n  A: {a[max(0, n - ctx):n + ctx]!r}\n  B: {b[max(0, n - ctx):n + ctx]!r}"


def run(stub, label, py, driver, args, pythonpath, work):
    stub.label = label
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    env.pop("HARNESS_BENCH", None)
    log = Path(work) / f"{label}.log"
    with open(log, "w") as f:
        rc = subprocess.call([py, str(driver)] + args, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=1800)
    bodies = stub.bodies.get(label, [])
    print(f"[{label}] rc={rc} requests={len(bodies)} log={log}")
    if not bodies:
        print(open(log).read()[-3000:])
        return None
    return bodies[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(BENCHES))
    ap.add_argument("--case", default="default", choices=["default", "alt"], help="which fixed episode to run")
    ap.add_argument("--python", default=sys.executable, help="interpreter of the benchmark env")
    ap.add_argument("--ref-driver", default=None, help="reference driver to compare against (same CLI)")
    ap.add_argument("--ref-pythonpath", default=None)
    ap.add_argument("--rel-pythonpath", default=None)
    ap.add_argument("--work", default=None, help="scratch dir (default: a new temp dir)")
    ap.add_argument("--save-ref", action="store_true", help="store the digests of the reference request in zeroshot_ref_requests.json")
    a = ap.parse_args()
    B = BENCHES[a.bench]
    work = Path(a.work or tempfile.mkdtemp(prefix=f"equiv_{a.bench}_"))
    work.mkdir(parents=True, exist_ok=True)
    stub = Stub()
    url, sam = f"http://127.0.0.1:{stub.port}/v1", f"http://127.0.0.1:{stub.port}"
    ok = True

    ref = None
    if a.ref_driver:
        ref = run(stub, "reference", a.python, a.ref_driver, B["cases"][a.case] + B["args"](url, sam, str(work / "out_reference")),
                  a.ref_pythonpath, work)
    rel = run(stub, "release", a.python, B["driver"], B["cases"][a.case] + B["args"](url, sam, str(work / "out_release")),
              a.rel_pythonpath, work)
    if rel is None:
        print("FAIL: the release driver sent no request")
        return 1
    (work / "release_request.json").write_bytes(rel)

    if ref is not None:
        (work / "reference_request.json").write_bytes(ref)
        if ref == rel:
            print(f"BYTE-IDENTICAL: reference vs release step-0 request ({len(rel)} bytes, sha256 {hashlib.sha256(rel).hexdigest()[:16]})")
        else:
            ok = False
            print("DIFFERENT: reference vs release step-0 request; " + first_diff(ref, rel))
            print("text-only view identical:", text_only(ref) == text_only(rel))

    digests = json.loads(REF_FILE.read_text()) if REF_FILE.exists() else {}
    src = ref if ref is not None else rel
    if a.case != "default":
        pass                                   # digests are stored for the default case only
    elif a.save_ref:
        digests[a.bench] = {"len": len(src), "sha256_full": hashlib.sha256(src).hexdigest(),
                            "sha256_text": hashlib.sha256(text_only_noimg(src)).hexdigest()}
        REF_FILE.write_text(json.dumps(digests, indent=1) + "\n")
        print(f"saved digests for {a.bench} -> {REF_FILE}")
    elif a.bench in digests:
        d = digests[a.bench]
        full = hashlib.sha256(rel).hexdigest() == d["sha256_full"]
        text = hashlib.sha256(text_only_noimg(rel)).hexdigest() == d["sha256_text"]
        print(f"vs stored paper request: full-bytes {'MATCH' if full else 'differ'}; text-only {'MATCH' if text else 'DIFFER'}")
        ok = ok and text
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
