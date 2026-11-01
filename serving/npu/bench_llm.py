"""Replay recorded harness requests against an OpenAI-compatible server; fixed output length (= recorded completion tokens,
ignore_eos) so different servers do identical work. Streams to time TTFT and decode separately.
usage: bench_llm.py URL MODEL reqs.jsonl out.jsonl [--conc 1] [--limit N] [--free]   (--free: natural stop, temperature 0, keep text)"""
import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def one(url, model, r, free):
    body = {"model": model, "messages": r["messages"], "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    if free:
        body["max_tokens"] = 4096
    else:
        body["max_tokens"] = int(r["rec_completion_tokens"] or 256)
        body["ignore_eos"] = True
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text, usage, n_chunks = None, [], None, 0
    with urllib.request.urlopen(req, timeout=600) as resp:
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
                delta = c.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
                    if free:
                        text.append(piece)
    total = time.perf_counter() - t0
    ct = (usage or {}).get("completion_tokens") or 0
    out = {"ep": r["ep"], "k": r["k"], "ttft_s": ttft, "total_s": total, "prompt_tokens": (usage or {}).get("prompt_tokens"),
           "completion_tokens": ct, "rec_prompt_tokens": r["rec_prompt_tokens"],
           "decode_tok_s": (ct - 1) / (total - ttft) if ttft and ct > 1 and total > ttft else None}
    if free:
        out["text"] = "".join(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("model"); ap.add_argument("reqs"); ap.add_argument("out")
    ap.add_argument("--conc", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--free", action="store_true")
    a = ap.parse_args()
    R = [json.loads(l) for l in open(a.reqs)]
    if a.limit:
        R = R[: a.limit]
    one(a.url, a.model, R[0], a.free)          # warm-up (graph capture / kernel compile)
    lock, f = threading.Lock(), open(a.out, "w")
    t0 = time.perf_counter()

    def job(r):
        try:
            o = one(a.url, a.model, r, a.free)
        except Exception as e:  # noqa: BLE001
            o = {"ep": r["ep"], "k": r["k"], "error": repr(e)[:300]}
        with lock:
            f.write(json.dumps(o) + "\n"); f.flush()
        return o

    with ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(job, R))
    wall = time.perf_counter() - t0
    ok = [o for o in res if "error" not in o]
    import statistics as s
    summ = {"n": len(R), "ok": len(ok), "conc": a.conc, "wall_s": wall,
            "total_s_median": s.median(o["total_s"] for o in ok),
            "ttft_s_median": s.median(o["ttft_s"] for o in ok if o["ttft_s"]),
            "decode_tok_s_median": s.median(o["decode_tok_s"] for o in ok if o["decode_tok_s"]),
            "gen_tok_per_s_aggregate": sum(o["completion_tokens"] for o in ok) / wall}
    print(json.dumps(summ))
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
