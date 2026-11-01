"""Proposers: who writes candidate edits.  The loop only sees this interface, so any agent can be the master.

A candidate is a dict:
  {"summary": str, "proposer": str, "failure_class": str, "failure_detail": str, "target_tasks": [task, ...],
   "markers": [unique trace string, ...], "capabilities": [{"capability": str, "category": str, "failure_class": str}],
   "diff": unified diff vs the parent}

  LLMProposer     any OpenAI-compatible chat endpoint (base_url, model, api key from an environment variable)
  ManualProposer  writes BRIEF.md into the round directory and waits for candidate files from a human or a CLI agent
  load_proposer   "llm" | "manual" | "module:Class" (e.g. a scripted proposer for tests)
"""
import importlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.request

PROMPT = pathlib.Path(__file__).resolve().parent.parent / "prompts" / "master_brief.md"


def render_brief(ctx):
    """Fill prompts/master_brief.md.  ctx keys: round, version, rules, harness, failures, ledger, n, tasks."""
    fails = "\n\n".join(f"### {f['task']} / seed {f['seed']}\n```\n{f['trace'][-4000:]}\n```" for f in ctx["failures"])
    src = "\n\n".join(f"### {p}\n```\n{s}\n```" for p, s in ctx["harness"].items())
    text = PROMPT.read_text()
    for k, v in dict(round=ctx["round"], version=ctx["version"], rules=ctx["rules"], harness=src, failures=fails or "(none)",
                     ledger=ctx["ledger"] or "(empty)", n=ctx["n"], tasks=", ".join(ctx["tasks"])).items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def parse_candidates(text):
    """```candidate {json}``` followed by ```diff ...``` blocks, in order."""
    blocks = re.findall(r"```(candidate|diff)\n(.*?)```", text, re.S)
    out, cur = [], None
    for kind, body in blocks:
        if kind == "candidate":
            cur = json.loads(body)
        elif cur is not None:
            cur["diff"] = body
            out.append(cur)
            cur = None
    return out


class LLMProposer:
    def __init__(self, cfg):
        p = cfg.get("proposer", {})
        self.url = p["base_url"].rstrip("/") + "/chat/completions"
        self.model, self.key = p["model"], os.environ.get(p.get("api_key_env", "MASTER_API_KEY"), "")
        self.temperature, self.timeout = p.get("temperature", 0.2), p.get("timeout_s", 600)

    def propose(self, ctx, round_dir):
        brief = render_brief(ctx)
        (round_dir / "BRIEF.md").write_text(brief)
        req = urllib.request.Request(self.url, data=json.dumps(dict(model=self.model, temperature=self.temperature,
                                     messages=[{"role": "user", "content": brief}])).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
        (round_dir / "proposer_reply.md").write_text(text)
        return [dict(c, proposer=c.get("proposer") or self.model) for c in parse_candidates(text)][: ctx["n"]]


class ManualProposer:
    """Human or CLI agent: read BRIEF.md, write candidates/<name>.json (+ <name>.diff), then touch candidates/DONE."""

    def __init__(self, cfg):
        self.poll = cfg.get("proposer", {}).get("poll_s", 10)

    def propose(self, ctx, round_dir):
        cdir = round_dir / "candidates"
        cdir.mkdir(parents=True, exist_ok=True)
        (round_dir / "BRIEF.md").write_text(render_brief(ctx))
        print(f"[manual proposer] brief written to {round_dir / 'BRIEF.md'}; waiting for {cdir / 'DONE'}", flush=True)
        while not (cdir / "DONE").exists():
            time.sleep(self.poll)
        out = []
        for f in sorted(cdir.glob("*.json")):
            c = json.loads(f.read_text())
            c["diff"] = f.with_suffix(".diff").read_text()
            out.append(c)
        return out[: ctx["n"]]


def load_proposer(cfg):
    kind = cfg.get("proposer", {}).get("kind", "manual")
    if kind == "llm":
        return LLMProposer(cfg)
    if kind == "manual":
        return ManualProposer(cfg)
    if cfg["_dir"] not in sys.path:
        sys.path.insert(0, cfg["_dir"])
    mod, cls = kind.split(":")
    return getattr(importlib.import_module(mod), cls)(cfg)
