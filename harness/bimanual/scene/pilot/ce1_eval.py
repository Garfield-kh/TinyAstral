#!/usr/bin/env python
"""Checker experiment 1 - evaluate one fixed model (Flash) under three input conditions, no training.

(a) v2 checker input: BEFORE = first frame of the pool, AFTER = last frame, cameras agent/top/wrist of the arm, v2 rules.
(b) criterion-chosen temporal views (all pool frames, at most 3, cameras chosen by criterion type) + measured robot state
    per frame and at the moment of judgement (gripper command/opening, fingertip position; no object state).
(c) (b) + at most one active check (lift_small / observe_now / show_view:<cam> / move_arm_away:<arm>); the simulator
    outcome was recorded with the episode (expert lift frame, later frames, removed views); unavailable -> "not possible".
Same short JSON output for all (criteria[] + decision + confidence + next_check), guided JSON, max_tokens 500, T=0.
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as C2  # noqa: E402
import ce1_crops as K  # noqa: E402
import re as _re

import paths as _paths  # noqa: E402
OUT = _paths.OUT / "checker_exp1"
SCHEMA = {"type": "object", "properties": {
    "criteria": {"type": "array", "items": {"type": "object", "properties": {
        "criterion": {"type": "string"}, "status": {"type": "string", "enum": ["satisfied", "unsatisfied", "unknown"]},
        "evidence": {"type": "string", "maxLength": 160}}, "required": ["criterion", "status", "evidence"]}},
    "decision": {"type": "string", "enum": ["satisfied", "unsatisfied", "unknown"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "next_check": {"type": ["string", "null"]}},
    "required": ["criteria", "decision", "confidence", "next_check"]}
OUTFMT = ('Answer with ONE JSON object: {"criteria": [{"criterion": "<relation>", "status": "satisfied|unsatisfied|unknown", '
          '"evidence": "<short>"}], "decision": "satisfied|unsatisfied|unknown", "confidence": <0..1>, "next_check": <null or a check>}. '
          '"decision" is "unsatisfied" if any criterion is unsatisfied, otherwise "unknown" if any is unknown, otherwise "satisfied".')
SYS_A = C2.CHECKER_SYSTEM.split("Answer with ONE JSON object only")[0] + OUTFMT + ' Use "next_check": null.'
SYS_B = """You are the CHECKER of a two-arm table-top robot. Decide, criterion by criterion, whether the subgoal is achieved NOW, using only the evidence given: a short sequence of camera frames (oldest first) and the measured robot state (gripper command and opening, fingertip positions in cm; z = height above the table). No object positions are given.
""" + C2.RELATION_DOC + """
Evidence rules:
- holding(arm, obj): a closed gripper next to the object is NOT enough. It is shown when the object moves together with the hand (e.g. the hand rose and the object rose with it), or the object hangs in the air held only by that gripper. A gripper commanded open holds nothing.
- not holding(arm, obj): the gripper is open and clearly apart from the object.
- on(a, b): the object must rest on b after release; an object still in a closed gripper or a visible gap above b is not "on".
- lifted(obj, h): judge the height of the object's bottom above where it was.
- If the frames do not show what a criterion needs (no frame after the action, the contact or support is hidden, a needed view is missing), the status is "unknown" - do not guess.
""" + OUTFMT
CHECKS = """
You may request ONE check if a criterion is unknown, by setting "decision": "unknown" and "next_check" to one of:
  "lift_small"           - the arm lifts a few cm and the frames are shown again (undone afterwards; judge the state before the check)
  "observe_now"          - take new frames now
  "show_view:<camera>"   - add a camera: agent, top, left or right
  "move_arm_away:<arm>"  - move the idle arm out of the view
You then get the result and must give the final answer. Otherwise use "next_check": null."""


def schema_for(n):
    if not n:
        return SCHEMA
    sc = json.loads(json.dumps(SCHEMA))
    sc["properties"]["criteria"]["minItems"] = n
    sc["properties"]["criteria"]["maxItems"] = n
    return sc


def img(rel):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode((OUT / rel).read_bytes()).decode()}}


def txt(t):
    return {"type": "text", "text": t}


def mstate(S):
    parts = []
    for a in ("left", "right"):
        x = S["arms"][a]
        g = "open" if x["gripper_cmd"] > 0.8 else ("closed" if x["gripper_cmd"] < 0.2 else "half-closed")
        parts.append(f"{a}: gripper commanded {g}, measured opening {x['opening']:.2f}, fingertips at {[round(v) for v in x['tcp']]}")
    return "; ".join(parts)


def cams_b(smp):
    names = [C2.parse(c)[1] for c in smp["criteria"]]
    if smp["arm_view"] == "both":
        return ["agent", "left", "right"]
    if any(n in ("on", "at", "on_table") for n in names):
        return ["agent", "top", smp["arm_view"]]
    return ["agent", smp["arm_view"]]


def load_states(eid, cache={}):
    if eid not in cache:
        cache[eid] = json.load(open(OUT / "episodes" / f"{eid}.json"))["states"]
    return cache[eid]


def crit_objs(smp, S):
    names = set()
    for c in smp["criteria"]:
        for x in C2.parse(c)[2]:
            if isinstance(x, str) and x in S[0]["objects"]:
                names.add(x)
    return sorted(names)


def crops_for(smp, idx, cams):
    """close-up crops (agent, and top when given) around the criteria objects + the main arm's fingertips."""
    S = load_states(smp["episode_id"])
    st = S[idx]
    objs = crit_objs(smp, S)
    tcp = st["arms"][smp["arm_main"]]["tcp"]
    st2 = json.loads(json.dumps(st))
    st2["objects"]["__tcp"] = {"lo": [tcp[0] - 3, tcp[1] - 3, max(tcp[2] - 3, 0)], "hi": [tcp[0] + 3, tcp[1] + 3, tcp[2] + 3]}
    out = []
    for cam in ("agent", "top"):
        if cam not in cams:
            continue
        rel = f"crops/{smp['episode_id']}/{idx:03d}_{cam}_{'-'.join(objs)}.jpg"
        if not (OUT / rel).exists():
            if K.crop(st2, objs + ["__tcp"], cam, rel) is None:
                continue
        out.append((cam, rel))
    return out


def since_last(S, a_idx, b_idx):
    A, B = S[a_idx], S[b_idx]
    acts = []
    for arm in ("left", "right"):
        ga, gb = A["arms"][arm]["gripper_cmd"], B["arms"][arm]["gripper_cmd"]
        if ga < 0.5 <= gb:
            acts.append(f"{arm} gripper opened")
        elif ga >= 0.5 > gb:
            acts.append(f"{arm} gripper closed")
        d = [B["arms"][arm]["tcp"][k] - A["arms"][arm]["tcp"][k] for k in range(3)]
        if sum(x * x for x in d) ** 0.5 > 2:
            acts.append(f"{arm} fingertips moved ({d[0]:+.0f}, {d[1]:+.0f}, {d[2]:+.0f}) cm")
    return acts


def build(smp, cond):
    if cond in ("d", "e", "f", "f2", "g"):
        return build_d(smp)
    cond = "c" if cond == "c2" else cond
    S = load_states(smp["episode_id"])
    crit_text = f"Subgoal: {smp['goal']} (arm: {smp['arm_view']})\nCompletion criteria: {'; '.join(smp['criteria'])}"
    if cond == "a":
        cams = ["agent", "top"] + (["left", "right"] if smp["arm_view"] == "both" else [smp["arm_view"]])
        f0, f1 = smp["pool"][0], smp["pool"][-1]
        c0 = [c for c in cams if c in f0.get("cams", cams)]
        c1 = [c for c in cams if c in f1.get("cams", cams)]
        content = ([txt("BEFORE:")] + [img(S[f0["idx"]]["frames"][c]) for c in c0] + [txt("AFTER (now):")]
                   + [img(S[f1["idx"]]["frames"][c]) for c in c1] + [txt(crit_text)])
        return [{"role": "system", "content": SYS_A}, {"role": "user", "content": content}]
    cams = cams_b(smp)
    frames = smp["pool"][-3:]
    content = []
    for k, f in enumerate(frames):
        cs = [c for c in cams if c in f.get("cams", cams)]
        content.append(txt(f"Frame {k + 1} ({f['stage']}) - cameras {', '.join(cs)}; measured: {mstate(S[f['idx']])}"))
        content += [img(S[f["idx"]]["frames"][c]) for c in cs]
    if smp["current"] != frames[-1]["idx"]:
        content.append(txt(f"Measured NOW (no image of this moment): {mstate(S[smp['current']])}"))
    content.append(txt(crit_text))
    sysm = SYS_B + (CHECKS if cond == "c" else ' Use "next_check": null.')
    return [{"role": "system", "content": sysm}, {"role": "user", "content": content}]


def build_d(smp):
    S = load_states(smp["episode_id"])
    crit_text = f"Subgoal: {smp['goal']} (arm: {smp['arm_view']})\nCompletion criteria: {'; '.join(smp['criteria'])}"
    cams = cams_b(smp)
    frames = smp["pool"][-3:]
    fix = LIFT_FIX and is_pick(smp)
    if fix:
        cams = ["agent", "top"]
    content = []
    for k, f in enumerate(frames):
        cs = [c for c in cams if c in f.get("cams", cams)]
        mv = ""
        if RULES_E and k > 0:
            acts = since_last(S, frames[k - 1]["idx"], f["idx"])
            mv = "; since the previous frame: " + ("; ".join(acts) if acts else "no robot motion")
        content.append(txt(f"Frame {k + 1} ({f['stage']}) - cameras {', '.join(cs)}; measured: {mstate(S[f['idx']])}{mv}"))
        if not smp.get("_is_demo"):  # demos: close-ups only (32-image limit per prompt)
            content += [img(S[f["idx"]]["frames"][c]) for c in cs]
        cr = fixed_crops(smp, f["idx"]) if fix else crops_for(smp, f["idx"], cs)
        if cr:
            content.append(txt(f"Frame {k + 1} close-ups: " + ", ".join(c for c, _ in cr)))
            content += [img(r) for _, r in cr]
    if smp["current"] != frames[-1]["idx"]:
        acts = since_last(S, frames[-1]["idx"], smp["current"])
        content.append(txt("After the last image the robot did: " + ("; ".join(acts) if acts else "nothing measurable")
                           + f". No image shows the result. Measured NOW: {mstate(S[smp['current']])}"))
    content.append(txt(crit_text))
    sysm = (SYS_B.replace(OUTFMT, "") + SYS_D_EXTRA + (SYS_E_EXTRA if RULES_E else "") + (SYS_F_EXTRA if LIFT_FIX and (fix or not GATE_F) else "")
            + "\n" + OUTFMT + CHECKS)
    msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": content}]
    if fix and not smp.get("_is_demo"):
        msgs = msgs[:1] + lift_demo(smp["task"]) + msgs[1:]
    return msgs


def is_pick(smp):
    names = {C2.parse(c)[1] for c in smp["criteria"]}
    return smp["arm_view"] in ("left", "right") and bool(names & {"holding", "lifted"}) and not names & {"on", "at", "on_table"}


def fixed_crops(smp, idx):
    """r2: the SAME image window for every frame of a pick sample: the criteria objects where they were before the grasp
    plus the fingertips at every frame (so 'object stayed on the table' vs 'object rose with the hand' is comparable)."""
    S = load_states(smp["episode_id"])
    objs = crit_objs(smp, S)
    p0 = smp["pool"][-3:][0]["idx"]
    lo, hi = [1e9] * 3, [-1e9] * 3
    boxes = [(S[p0]["objects"][o]["lo"], S[p0]["objects"][o]["hi"]) for o in objs]
    for f in (smp["pool"][-3:][1:] if LIFT_FIX2 else smp["pool"][-3:]):  # f2: skip the pre-grasp tcp (arm far away)
        t = S[f["idx"]]["arms"][smp["arm_main"]]["tcp"]
        boxes.append(([t[0] - 3, t[1] - 3, max(t[2] - 3, 0)], [t[0] + 3, t[1] + 3, t[2] + 3]))
    for a, b in boxes:
        lo = [min(lo[k], a[k]) for k in range(3)]
        hi = [max(hi[k], b[k]) for k in range(3)]
    st2 = json.loads(json.dumps(S[idx]))
    st2["objects"]["__win"] = {"lo": lo, "hi": hi}
    out = []
    for cam in ("agent", "top"):
        rel = f"crops/{smp['episode_id']}/{'fix2' if LIFT_FIX2 else 'fix'}_{p0:03d}_{smp['pool'][-1]['idx']:03d}_{idx:03d}_{cam}.jpg"
        if not (OUT / rel).exists():
            if K.crop(st2, ["__win"], cam, rel) is None:
                continue
        out.append((cam, rel))
    return out


DEMO_IDS = {"place_empty_cup": "val__place_empty_cup__s49", "stack_blocks_three": "val__stack_blocks_three__s49"}


def lift_demo(task, cache={}):
    """r2 paired demo: an empty lift and the real (re-)lift of the same scene (dedicated val seeds 4900+, not in any test set)."""
    task = task if task in DEMO_IDS else "stack_blocks_three"
    if task in cache:
        return cache[task]
    rows = [json.loads(l) for l in open(OUT / "demo_lift.jsonl")]
    eps = [r["episode_id"] for r in rows if r["task"] == task]
    out = []
    for eid in eps:
        n = next((r for r in rows if r["episode_id"] == eid and r["kind"] in ("N1_missed_rise", "N3_one_finger_rise")), None)
        p = next((r for r in rows if r["episode_id"] == eid and r["kind"] == "P1_pick_done" and r["category"] == "positive"), None)
        if n and p:
            for r, ans in ((n, "unsatisfied"), (p, "satisfied")):
                r = dict(r, _is_demo=True)
                u = build_d(r)[1]
                obj = r["criteria"][0].split(",")[1].strip(" )")
                ev = ("the object is still on the table where it was before the grasp; the closed gripper rose without it"
                      if ans == "unsatisfied" else "the object left the table and is between the fingers at the new height")
                a = {"criteria": [{"criterion": c, "status": ans, "evidence": ev} for c in r["criteria"]], "decision": ans,
                     "confidence": 1.0, "next_check": None}
                out += [{"role": "user", "content": [txt(f"EXAMPLE ({'empty lift' if ans == 'unsatisfied' else 'real lift'}, "
                                                          f"same scene as the other example; {obj})")] + u["content"]},
                        {"role": "assistant", "content": json.dumps(a)}]
            break
    cache[task] = out
    return out


def check_result(smp, req):
    S = load_states(smp["episode_id"])
    req = (req or "").strip()
    ch = smp.get("checks") or {}
    cams = cams_b(smp)
    if req == "lift_small" and ch.get("lift_small"):
        idx = ch["lift_small"][0]
        if STRICT_LIFT:
            cur = S[smp["current"]]
            return ([txt(f"CHECK lift_small - test lift (measured during the test: {mstate(S[idx])}):")]
                    + [img(S[idx]["frames"][c]) for c in cams]
                    + ([img(r) for _, r in crops_for(smp, idx, cams)] if CROPS else [])
                    + [txt("The arm was then lowered back. Use the test lift ONLY to judge holding (did the object move "
                           "together with the hand?). Every other criterion (lifted, on, ...) must be judged on the state "
                           f"BEFORE the test, shown again here (measured: {mstate(cur)}):")]
                    + [img(cur["frames"][c]) for c in cams])
        return ([txt(f"CHECK lift_small (the arm lifted a few cm; this is undone afterwards): measured {mstate(S[idx])}")]
                + [img(S[idx]["frames"][c]) for c in cams])
    if req == "observe_now" and ch.get("observe_now"):
        out = [txt("CHECK observe_now: new frames")]
        for idx in ch["observe_now"][-2:]:
            out += [txt(f"measured {mstate(S[idx])}")] + [img(S[idx]["frames"][c]) for c in cams]
            if CROPS:
                out += [img(r) for _, r in crops_for(smp, idx, cams)]
        return out
    if req.startswith("show_view:") and ch.get("show_view"):
        cam = req.split(":", 1)[1].strip()
        if cam in ("agent", "top", "left", "right"):
            out = [txt(f"CHECK show_view:{cam}")]
            for f in smp["pool"][-3:]:
                out += [txt(f"{f['stage']}:"), img(S[f["idx"]]["frames"][cam])]
                if CROPS:
                    out += [img(r) for _, r in crops_for(smp, f["idx"], [cam])]
            return out
    return None


STRICT_LIFT = False
RETRY = False
CROPS = False
RULES_E = False
SYS_E_EXTRA = """
Hard rules (they override your own inference):
- If the text contains "After the last image the robot did:", every criterion about an object is "unknown" and you must set
  "next_check": "observe_now" - even if the result seems obvious. Only robot-state criteria (gripper_open/closed) may be decided.
- If the object a criterion is about is hidden by an arm or gripper in the view that criterion needs (top camera for at(),
  side/perspective camera for on() and lifted()), that criterion is "unknown" - request "move_arm_away:<arm>" or "observe_now".
- Each frame line says how far the fingertips moved since the previous frame (a measured robot fact). For holding/lifted,
  check in the close-ups whether the object moved by about the same amount."""
LIFT_FIX = False
LIFT_FIX2 = False
GATE_F = False  # r3 (cond g): SYS_F_EXTRA only in pick (holding/lifted) prompts; other samples get exactly the e prompt
SYS_F_EXTRA = """
Pick-up rule (holding / lifted). For a pick-up you get FIXED external views (overview "agent" and "top") at: before the grasp,
after the gripper closed, after the lift. The close-ups show the SAME image window in all three frames - compare them.
- holding(...) and lifted(...) pass ONLY if BOTH: (1) the object has LEFT its support (in the after-lift frame there is a gap
  under it / it is no longer where it stood before the grasp) AND (2) it MOVED WITH THE GRIPPER (it is between the fingers at
  the new, higher position).
- If, after the lift, the object still stands on the table (or its support) where it was before the grasp, both are
  "unsatisfied" - a closed gripper that rose proves nothing.
- If the object or the fingertips are hidden in the after-lift views so you cannot tell, the status is "unknown".
Two examples (an empty lift and a real lift in the same scene) come first."""
SYS_D_EXTRA = """
Evidence policy (applies to every criterion):
- The measured state proves ROBOT facts only: whether a gripper is commanded open/closed, its opening, and where the fingertips are.
- A criterion about an OBJECT (on, at, lifted, holding, not holding) needs an image taken AFTER the robot's last action that
  could change it. If the text says the robot did something after the last image, such a criterion is "unknown" - request
  "observe_now" instead of inferring the result.
- Height and support (on, lifted) cannot be judged from the top camera alone (it looks straight down): if only top-camera
  images are given, they are "unknown" - request "show_view:agent". at(a, x, y) needs the top camera.
- holding: compare the object in the close-up crops before and after the hand moved. If the hand rose and the object stayed
  where it was, the grasp FAILED (unsatisfied), even though the gripper is closed.
Each frame comes with close-up crops around the objects of the criteria (and the fingertips); the full views are also given."""
lock = threading.Lock()


class Client:
    def __init__(self, apis, per):
        self.apis = apis
        self.sem = {a: threading.Semaphore(per) for a in apis}
        self.i = 0

    def call(self, messages, max_tokens, n_crit=None, temperature=0):
        with lock:
            api = self.apis[self.i % len(self.apis)]
            self.i += 1
        body = {"model": "qwen", "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "checker", "schema": schema_for(n_crit)}}}
        err = None
        with self.sem[api]:
            for att in range(4):
                try:
                    req = urllib.request.Request(api + "/chat/completions", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
                    r = json.load(urllib.request.urlopen(req, timeout=600))
                    return r["choices"][0]["message"].get("content") or "", r.get("usage") or {}, None
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                    time.sleep(5 * (att + 1))
        return "", {}, err


def parse(t):
    try:
        o = json.loads(t)
        if o.get("decision") in ("satisfied", "unsatisfied", "unknown"):
            return o
    except Exception:  # noqa: BLE001
        pass
    return None


def run_one(cl, smp, cond, max_tokens):
    msgs = build(smp, cond)
    nc = len(smp["criteria"]) if cond in ("d", "e", "f", "f2", "g") else None
    text, usage, err = cl.call(msgs, max_tokens, nc)
    o = parse(text)
    retried = False
    if o is None and RETRY:                      # guided-JSON whitespace loop: one retry with the same input at T=0.3
        text, usage, err = cl.call(msgs, max_tokens, nc, temperature=0.3)
        o = parse(text)
        retried = True
    rec = {"id": smp["id"], "cond": cond, "outputs": [text], "error": err, "checks": 0, "retried": retried, "check_request": None, "check_available": None}
    if cond in ("c", "c2", "d", "e", "f", "f2", "g") and o and o["decision"] == "unknown" and o.get("next_check"):
        res = check_result(smp, o["next_check"])
        rec.update(checks=1, check_request=o["next_check"], check_available=res is not None)
        follow = res if res else [txt(f"CHECK {o['next_check']}: not possible here. Give your final answer with the evidence you have.")]
        msgs2 = msgs + [{"role": "assistant", "content": text},
                        {"role": "user", "content": follow + [txt('Give the final answer now ("next_check": null).')]}]
        text2, u2, err2 = cl.call(msgs2, max_tokens, nc)
        o2 = parse(text2)
        if o2 is None and RETRY:
            text2, u2, err2 = cl.call(msgs2, max_tokens, nc, temperature=0.3)
            o2 = parse(text2)
            rec["retried"] = True
        rec["outputs"].append(text2)
        o = o2
        rec["error"] = rec["error"] or err2
    rec["parsed"] = o is not None
    rec["decision"] = o["decision"] if o else None
    rec["confidence"] = float(o.get("confidence", 0)) if o else None
    rec["criteria_out"] = o.get("criteria") if o else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--conds", default="a,b,c")
    ap.add_argument("--apis", default=_paths.DEFAULT_URL)
    ap.add_argument("--per_api", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    smps = [json.loads(l) for l in open(OUT / f"{a.split}.jsonl")]
    if a.limit:
        smps = smps[: a.limit]
    cl = Client(a.apis.split(","), a.per_api)
    od = OUT / "results"
    od.mkdir(exist_ok=True)
    global STRICT_LIFT, CROPS, RULES_E, LIFT_FIX, LIFT_FIX2, GATE_F
    for cond in a.conds.split(","):
        STRICT_LIFT = cond in ("c2", "d", "e", "f", "f2", "g")
        CROPS = cond in ("d", "e", "f", "f2", "g")
        RULES_E = cond in ("e", "f", "f2", "g")
        LIFT_FIX = cond in ("f", "f2", "g")
        LIFT_FIX2 = cond in ("f2", "g")
        GATE_F = cond == "g"
        path = od / f"{a.split}_{cond}{os.environ.get('RES_SUFFIX', '')}.jsonl"  # r3: RES_SUFFIX for replicate runs
        done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
        todo = [s for s in smps if s["id"] not in done]
        print(a.split, cond, "todo", len(todo), flush=True)

        def one(s, cond=cond, path=path):
            r = run_one(cl, s, cond, a.max_tokens)
            with lock:
                with open(path, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with ThreadPoolExecutor(len(cl.apis) * a.per_api) as ex:
            list(ex.map(one, todo))


if __name__ == "__main__":
    main()
