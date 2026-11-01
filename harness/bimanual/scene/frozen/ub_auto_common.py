"""Shared pieces of the AUTONOMOUS-termination bake-off (R / H / V arms, step 0).

checker_g(): the experiment-1 checker in config g (ce1_eval: STRICT_LIFT + CROPS + RULES_E + LIFT_FIX(2) + GATE_F),
with the g50 relation doc patch (on() covers inside, task_success()). Task-level "done" is judged with the single
criterion task_success() and the overall instruction as the goal; cameras agent/top/left/right for that case.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

P = Path(__file__).resolve().parents[1]
OP = P / "pilot"
sys.path.insert(0, str(OP))
import ce1_eval as E  # noqa: E402

REPL = [  # same text patches as g50_cl.REPL (checker-relevant part)
    ('a rests on top of b (b may be "table")', 'a rests on top of b or inside it (b may be "table")'),
    ("  above(a, b)            a is higher than b and over it",
     "  above(a, b)            a is higher than b and over it\n"
     "  task_success()         the overall task instruction is fulfilled (for steps no other relation describes, e.g.\n"
     "                         pressing, opening, turning, shaking)"),
]
ORIG_OUT = E.OUT
_demo_done = {}
TASK_CAMS = ["agent", "top", "left", "right"]
_cams_b = E.cams_b


def _cams(smp):
    if [c.replace(" ", "") for c in smp["criteria"]] == ["task_success()"]:
        return list(TASK_CAMS)
    return _cams_b(smp)


E.cams_b = _cams


def set_g(crops=True):
    E.STRICT_LIFT = True
    E.CROPS = crops
    E.RULES_E = True
    E.LIFT_FIX = True
    E.LIFT_FIX2 = True
    E.GATE_F = True


def g_messages(smp, states, out_dir, live_checks=True, crops=True):
    """smp: {episode_id, task, goal, arm_view, arm_main, criteria, pool:[{idx, stage}], current}."""
    set_g(crops)
    if crops and smp.get("task") and not _demo_done.get(smp["task"]):
        # the paired lift demo (pick samples) is built from the checker_exp1 data dir; cache it before switching OUT
        E.OUT = E.K.OUT = ORIG_OUT
        E.lift_demo(smp["task"])
        _demo_done[smp["task"]] = True
    E.OUT = Path(out_dir)
    E.K.OUT = Path(out_dir)
    E.load_states.__defaults__[0][smp["episode_id"]] = states
    if not crops:
        E.crops_for = lambda *a, **k: []
        E.fixed_crops = lambda *a, **k: []
    msgs = E.build_d(smp)
    s = msgs[0]["content"]
    for a, b in REPL:
        s = s.replace(a, b)
    if not live_checks:
        s = s.replace(E.CHECKS, ' Use "next_check": null.')
    msgs[0]["content"] = s
    return msgs


def chat_json(api, msgs, n, log=None, model="qwen"):
    """guided-JSON checker call, T=0 then one retry at T=0.3. Returns (text, n_calls)."""
    calls = 0
    text = ""
    for temp in (0.0, 0.3):
        calls += 1
        body = {"model": model, "messages": msgs, "max_tokens": 500, "temperature": temp,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "checker", "schema": E.schema_for(n)}}}
        t0, err = time.time(), None
        for att in range(3):
            try:
                req = urllib.request.Request(api + "/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
                r = json.load(urllib.request.urlopen(req, timeout=600))
                text = r["choices"][0]["message"].get("content") or ""
                err = None
                break
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                time.sleep(5 * (att + 1))
        if log:
            log({"kind": "model_call", "role": "checker", "seconds": round(time.time() - t0, 1), "output": text, "error": err})
        if E.parse(text):
            return text, calls
    return text, calls


def parse(text):
    return E.parse(text) or {"decision": "unsatisfied", "criteria": [], "next_check": None}


def strip(msgs):
    """drop private keys (e.g. _file) from content parts; keep every image (the checker was tuned without trimming)."""
    out = []
    for m in msgs:
        c = m["content"]
        if isinstance(c, list):
            c = [{k: v for k, v in p.items() if not k.startswith("_")} for p in c]
        out.append({"role": m["role"], "content": c})
    return out
