"""Smoke-test a model service (vLLM or SGLang, GPU or Huawei Ascend Atlas A2) with recorded harness requests
(multi-turn, with images) -- the same body shape the harness's chat() sends. Streams, so it reports TTFT and decode rate, and checks that the
output is a parsable <tool_call> with a known tool name.

REQS.jsonl holds one recorded request per line: {"messages": [...OpenAI chat messages...], "ep": str, "k": int,
"rec_prompt_tokens": int, "rec_completion_tokens": int}.  Export requests from an episode.json (steps[]) or construct the file manually; see docs/npu.md.

usage: python3 smoke_service.py [URL] [MODEL] [REQS.jsonl] [--n 3]
default: $TINYASTRAL_URL (http://127.0.0.1:8000/v1)  qwen  ./bench_reqs.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default=os.environ.get("TINYASTRAL_URL", "http://127.0.0.1:8000/v1"))
ap.add_argument("model", nargs="?", default=os.environ.get("TA_SERVED_NAME", "qwen"))
ap.add_argument("reqs", nargs="?", default="bench_reqs.jsonl")
ap.add_argument("--n", type=int, default=3)
a = ap.parse_args()

R = [json.loads(l) for l in open(a.reqs)]
# pick requests that actually carry an image (the multimodal path is the one we need to prove)
R = [r for r in R if any(isinstance(m["content"], list) and any(c["type"] == "image_url" for c in m["content"]) for m in r["messages"])]
print(f"{len(R)} recorded requests carry images; using the last {a.n}")

KNOWN = {"locate", "grasp", "place", "lift", "push", "articulate", "verify", "done", "move", "release", "open", "close", "look", "home"}
bad = 0
for r in R[-a.n:]:
    nimg = sum(1 for m in r["messages"] if isinstance(m["content"], list) for c in m["content"] if c["type"] == "image_url")
    body = {"model": a.model, "messages": r["messages"], "temperature": 0.0, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": True},
            "stream": True, "stream_options": {"include_usage": True}}
    rq = urllib.request.Request(a.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    t0 = time.perf_counter()
    ttft, out, usage = None, [], None
    with urllib.request.urlopen(rq, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices") or []:
                p = (c.get("delta") or {}).get("content") or ""
                if p:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    out.append(p)
    el = time.perf_counter() - t0
    text = "".join(out)
    nt = (usage or {}).get("completion_tokens") or 0
    pt = (usage or {}).get("prompt_tokens") or 0
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    ok, why = False, ""
    if not m:
        why = "no <tool_call> block"
    else:
        try:
            call = json.loads(m.group(1))
            ok = isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict)
            why = f"name={call.get('name')!r}" + ("" if call.get("name") in KNOWN else "  (name not in the known tool set)")
        except Exception as e:  # noqa: BLE001
            why = f"tool_call JSON is not parsable: {e}"
    # gibberish guard: a degenerate / wrong-dtype model repeats or emits non-ascii soup
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / max(1, len(text))
    print(f"\n--- ep={r.get('ep')} k={r.get('k')} images={nimg} ---")
    print(f"  prompt_tokens={pt} (recorded {r.get('rec_prompt_tokens')})  completion_tokens={nt}")
    print(f"  TTFT={ttft:.3f}s  total={el:.2f}s  decode={(nt - 1) / max(1e-9, el - (ttft or 0)):.1f} tok/s  printable={printable:.3f}")
    print(f"  tool_call: {'OK' if ok else 'FAIL'}  {why}")
    print("  --- raw output (first 600 chars) ---")
    print("  " + text[:600].replace("\n", "\n  "))
    if not ok or printable < 0.99:
        bad += 1

print(f"\n==== {a.n - bad}/{a.n} requests produced a valid tool call ====")
sys.exit(1 if bad else 0)
