#!/usr/bin/env python
"""Closed-loop v3 harness (Flash, oracle object state in the text, held-out seeds 2000+ = same as the v2 closed loop).

Executor answers the v3 JSON {seen, reported, assessment, uncertain, memory_delta, next, tool_call}; it never declares
completion: tool_call request_check(subgoal_id) runs the CHECKER (round-2 input "e": temporal frames + close-up crops +
measured state + evidence policy; one active check executed live in the simulator). Only a "satisfied" verdict advances
the subgoal. MEMORY is maintained by the harness (confirmed subgoals, held object offsets, successful grasp parameters,
recent failures with counts) plus the model's memory_delta notes.
Conditions: A zero-shot; B + v3 demonstration (full JSON incl. evidence); C + same demonstration with only tool_call.
Budget 120 model calls per episode (planner + executor + checker); wall clock 3600 s safety cap. Success = check_success().
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import closed_loop as CL  # noqa: E402
import common_v2 as C  # noqa: E402
import v3_build as VB  # noqa: E402
import ce1_eval as E  # noqa: E402
import ce1_crops as K  # noqa: E402
import record_v2 as RV2  # noqa: E402
from record import ArmTag, wrist_to_tcp  # noqa: E402

import paths as _paths  # noqa: E402
V3 = _paths.OUT / "oracle_expert_v3"
import os
OUT = Path(os.environ.get("V3_CL_OUT", str(V3 / "closed_loop")))
SEEDS = _paths.OUT / "oracle_expert_pilot_v2/closed_loop/seedcache"
CHECKER_DEMO = ""
VIEW_NAME = {"agent": "overview", "left": "left wrist", "right": "right wrist", "top": "top"}


def limgs(ep, S, cams):
    """r2: label every image with its view (the r1 annotator confused left/right wrist without labels)."""
    out = []
    for c in cams:
        if c in S["frames"]:
            out += [CL.txt(VIEW_NAME.get(c, c) + ":")] + ep.imgs(S, [c])
    return out


CHECKER_SYS = (E.SYS_B.replace(E.OUTFMT, "") + E.SYS_D_EXTRA + E.SYS_E_EXTRA + "\n" + E.OUTFMT + E.CHECKS)


# ============================================================================ tools with centring error
class Tools3(CL.Tools):
    """v2 tools + object_offset_cm / grasp_centering_error_cm on grasp. The planned grasp point is taken from the SAME
    grasp_actor call the v2 tool makes (captured), so the robot behaves exactly as with the v2 tools."""

    def t_grasp(self, arm, object, part=None, lift_cm=None):
        T = self.T
        S0 = self.ep.states[-1]
        cap = {}
        orig = T.grasp_actor

        def capture(*a, **kw):
            out = orig(*a, **kw)
            if "act" not in cap and out and out[0] is not None and out[1]:
                cap["act"] = out
            return out
        T.grasp_actor = capture
        try:
            fb, S = super().t_grasp(arm, object, part, lift_cm)
        finally:
            del T.grasp_actor            # back to the class method
        if S["held"].get(arm) == object:
            off = VB.offset_of(S, arm, object)
            fb["object_offset_cm"] = off
            if "act" in cap:
                mv = [x for x in cap["act"][1] if x.action == "move"]
                if mv:
                    tcp = wrist_to_tcp(np.asarray(mv[-1].target_pose, float))
                    tz = self.sim.table_z
                    o = S0["objects"][object]["center"]
                    nom = [o[0] - tcp[0] * 100, o[1] - tcp[1] * 100, o[2] - (tcp[2] - tz) * 100]
                    fb["grasp_centering_error_cm"] = round(((off[0] - nom[0]) ** 2 + (off[1] - nom[1]) ** 2) ** 0.5, 1)
        return fb, S


# ============================================================================ demonstrations
XTASK = {"place_empty_cup": "stack_blocks_three", "stack_blocks_three": "handover_block", "handover_block": "place_empty_cup"}


def demo_text(task, cond):
    if cond in ("A", "N6"):                          # N6 = S6 harness (next line + hand-off rule), no demo at all
        return {"planner": "", "executor": "", "checker": ""} if cond == "N6" else {"planner": "", "executor": ""}
    if cond == "X6":                                   # X6 = S6 with the frozen S6 demo of a DIFFERENT task (cross-task ICL)
        f = OUT / "demos" / f"{task}_X6.json"
        if f.exists():
            return json.loads(f.read_text())
        d = json.loads((V3 / "closed_loop_r3s6" / "demos" / f"{XTASK[task]}_S6.json").read_text())
        d["demo_task"] = XTASK[task]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(d, ensure_ascii=False))
        return d
    f = OUT / "demos" / f"{task}_{cond}.json"          # frozen at first use so every episode of a condition sees the same demo
    if f.exists():
        return json.loads(f.read_text())
    d = v2_demo(task) if cond == "S3" else _demo_text(task, "S4" if cond in ("S5", "S6") else cond)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, ensure_ascii=False))
    return d


def _demo_text(task, cond):
    ex = [json.loads(l) for l in open(V3 / "executor.jsonl")]
    pl = [json.loads(l) for l in open(V3 / "planner.jsonl")]
    nsub = {r["episode_id"]: len(r["subgoals"]) for r in json.load(open(V3 / "build" / "planner.json"))}
    have = {}
    for r in ex:
        if r["task"] == task and not r.get("branch"):
            have.setdefault(r["episode_id"], set()).add(r["subgoal_index"])
    # r2: only an episode whose clean subgoal conversations are complete (r1 picked s0, which lacked subgoal 2 = first place)
    clean = sorted(e for e, s in have.items() if s == set(range(nsub.get(e, -1))))
    # r3: prefer the complete episode that shows the most place -> home(same arm) -> request_check sequences (arm hand-offs
    # on stack); tasks without any keep clean[0] as in r2
    ph = {}
    for c in json.load(open(V3 / "build" / "conversations.json")):
        if c["branch"] is None and c["task"] == task:
            tl = [t["call"] for t in c["turns"] if t["kind"] == "tool"]
            ph[c["episode_id"]] = ph.get(c["episode_id"], 0) + sum(
                1 for a, b in zip(tl, tl[1:]) if a["tool"] == "place" and b["tool"] == "home" and a["args"]["arm"] == b["args"]["arm"])
    eid = max(clean, key=lambda e: (ph.get(e, 0), -clean.index(e)))
    rec = None if cond == "S4" else next((r for r in ex if r["task"] == task and r.get("branch") and r["episode_id"] != eid), None)

    def text_of(c):
        if isinstance(c, str):
            return c
        return "\n".join(p["text"] for p in c if p.get("type") == "text") + ("" if all(p.get("type") == "text" for p in c) else "\n[images omitted]")

    def asst(s):
        if cond in ("B", "S1", "S2", "S4"):
            return s
        try:
            return json.dumps({"tool_call": json.loads(s)["tool_call"]}, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return s
    p = next(r for r in pl if r["episode_id"] == eid)
    pb = ("\n\n# Example: a successful episode of the same task (scene differs; images omitted)\nUSER: "
          + text_of(p["messages"][1]["content"]) + "\nASSISTANT: " + p["messages"][2]["content"])
    eb = ["\n\n# Example: how a successful episode of the same task was executed (scene differs; images omitted)."]
    convs = sorted((r for r in ex if r["episode_id"] == eid and not r.get("branch")), key=lambda r: r["subgoal_index"])
    if rec:
        convs.append(rec)
    for r in convs:
        eb.append(f"\n## Subgoal {r['subgoal_index'] + 1}" + (f" (a recovery from: {r['branch']['actual']})" if r.get("branch") else ""))
        for m in r["messages"][1:]:
            if m["role"] == "user":
                eb.append("USER: " + text_of(m["content"]))
            elif m["role"] == "assistant":
                eb.append("ASSISTANT: " + asst(m["content"]))
            elif m["role"] == "tool":
                eb.append("TOOL RESULT: " + text_of(m["content"]))
    if cond == "C":
        eb.insert(1, "(only the tool calls are shown; your own answers must still use the full JSON format)")
    out = {"planner": pb, "executor": "\n".join(eb), "demo_episode": eid}
    if cond in ("S1", "S4"):
        out["checker"] = checker_demo(task, eid)
    return out


def checker_demo(task, eid):
    """r2: 4 v3-format checker examples from the demo episode (as v2 did): 2 satisfied, 1 unsatisfied, 1 unknown if present."""
    ck = [json.loads(l) for l in open(V3 / "checker.jsonl")]
    rows = [r for r in ck if r["episode_id"] == eid] or [r for r in ck if r["task"] == task]
    pick = []
    for lab, n in (("satisfied", 2), ("unsatisfied", 1), ("unknown", 1)):
        pick += [r for r in rows if r["label"] == lab][:n]
    pick += [r for r in rows if r not in pick][: 4 - len(pick)]

    def t(c):
        return c if isinstance(c, str) else "\n".join(p["text"] for p in c if p.get("type") == "text") + "\n[images omitted]"
    cb = ["\n\n# Examples from an episode of the same task (scene differs; images omitted)"]
    for r in pick[:4]:
        cb.append("USER: " + t(r["messages"][1]["content"]) + "\nASSISTANT: " + r["messages"][2]["content"])
    return "\n".join(cb)


def v2_demo(task):
    """S3: the v2 demo (v2 C = without think) converted to the v3 interface: <tool_call>X</tool_call> -> {"tool_call": X};
    "Subgoal complete." -> request_check + a satisfied CHECKER result."""
    d = CL.demo_block(task, "C")
    lines, sg, out = d["executor"].split("\n"), 0, []
    for ln in lines:
        m = re.match(r"## Subgoal (\d+)", ln)
        if m:
            sg = int(m.group(1))
        if ln.startswith("ASSISTANT: "):
            body = ln[len("ASSISTANT: "):]
            mt = re.search(r"<tool_call>(.*?)</tool_call>", body, re.S)
            if mt:
                ln = "ASSISTANT: " + json.dumps({"tool_call": json.loads(mt.group(1))}, ensure_ascii=False)
            elif "Subgoal complete" in body:
                ln = ("ASSISTANT: " + json.dumps({"tool_call": {"name": "request_check", "arguments": {"subgoal_id": sg}}})
                      + '\nTOOL RESULT: CHECKER: {"checker": "satisfied", "criteria": [...], "next_check": null}')
        out.append(ln)
    out.insert(3, "(only the tool calls are shown; your own answers must still use the full JSON format)")
    ex = [json.loads(l) for l in open(V3 / "executor.jsonl")]
    return {"planner": d["planner"], "executor": "\n".join(out), "checker": checker_demo(task, None) if False else
            checker_demo(task, _first_complete(task, ex)), "demo_episode": "v2:" + d["demo_episode"]}


def _first_complete(task, ex):
    nsub = {r["episode_id"]: len(r["subgoals"]) for r in json.load(open(V3 / "build" / "planner.json"))}
    have = {}
    for r in ex:
        if r["task"] == task and not r.get("branch"):
            have.setdefault(r["episode_id"], set()).add(r["subgoal_index"])
    return sorted(e for e, s in have.items() if s == set(range(nsub.get(e, -1))))[0]


# ============================================================================ checker (live)
def checker(ep, sub, frames_idx, tools):
    """round-2 'e' checker with one live active check. Returns (verdict dict, n_calls)."""
    S = ep.states
    K.OUT = ep.dir
    smp = {"episode_id": "_live", "goal": sub["goal"], "arm_view": sub["arm"],
           "arm_main": sub["arm"] if sub["arm"] in ("left", "right") else sub.get("arm_main", "right"),
           "criteria": sub["success"] or ["gripper_open(left)"], "pool": [{"idx": i, "stage": f"after step {k}"} for k, i in enumerate(frames_idx)],
           "current": frames_idx[-1], "checks": {}}
    E.load_states.__defaults__[0]["_live"] = S
    E.OUT = ep.dir
    E.STRICT_LIFT, E.CROPS, E.RULES_E = True, True, True
    msgs = E.build_d(smp)
    msgs[0]["content"] = CHECKER_SYS + CHECKER_DEMO
    msgs = CL.api_msgs([{"role": m["role"], "content": m["content"]} for m in msgs])
    nc = len(smp["criteria"])
    text = ep.llm.chat("checker", msgs, max_tokens=500) if False else _chat_json(ep, msgs, nc)
    o = E.parse(text) or {"decision": "unsatisfied", "criteria": [], "next_check": None}
    calls = 1
    if o.get("decision") == "unknown" and o.get("next_check"):
        res = live_check(ep, smp, o["next_check"], tools)
        follow = res or [CL.txt(f"CHECK {o['next_check']}: not possible here. Give your final answer with the evidence you have.")]
        msgs2 = msgs + [{"role": "assistant", "content": text},
                        {"role": "user", "content": follow + [CL.txt('Give the final answer now ("next_check": null).')]}]
        text2 = _chat_json(ep, CL.api_msgs(msgs2), nc)
        calls += 1
        o = E.parse(text2) or o
    return {"checker": o.get("decision"), "criteria": o.get("criteria") or [], "next_check": o.get("next_check")}, calls


def _chat_json(ep, msgs, n):
    """checker call with guided JSON (criteria length fixed); one retry on a whitespace loop."""
    llm = ep.llm
    if llm.calls >= CL.MAX_CALLS:
        raise CL.Budget("max_calls")
    for temp in (0.0, 0.3):
        llm.calls += 1
        body = {"model": llm.model, "messages": msgs, "max_tokens": 500, "temperature": temp,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "checker", "schema": E.schema_for(n)}}}
        t0 = time.time()
        text, err = "", None
        try:
            import urllib.request
            req = urllib.request.Request(llm.api + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
            r = json.load(urllib.request.urlopen(req, timeout=900))
            text = r["choices"][0]["message"].get("content") or ""
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        ep.trace({"kind": "model_call", "role": "checker", "call": llm.calls, "seconds": round(time.time() - t0, 1),
                  "input": CL.strip_images(msgs), "output": text, "error": err})
        if E.parse(text):
            return text
        if llm.calls >= CL.MAX_CALLS:
            raise CL.Budget("max_calls")
    return text


def live_check(ep, smp, req, tools):
    req = (req or "").strip()
    T = ep.sim.T
    cams = E.cams_b(smp)
    S = ep.states
    if req.startswith("lift_small"):
        arm = smp["arm_main"]
        if S[-1]["arms"][arm]["gripper_cmd"] > 0.8:
            return None
        tools._run(T.move_by_displacement(ArmTag(arm), z=0.03))
        st_up = ep.snap("check lift")
        idx_up = len(ep.states) - 1
        tools._run(T.move_by_displacement(ArmTag(arm), z=-0.03))
        ep.snap("check lowered")
        return ([CL.txt(f"CHECK lift_small - test lift (measured: {E.mstate(st_up)}):")] + [CL.img(ep.dir, st_up["frames"][c]) for c in cams]
                + [CL.img(ep.dir, r) for _, r in E.crops_for(dict(smp, current=idx_up), idx_up, cams)]
                + [CL.txt("The arm was then lowered back. Use the test lift ONLY to judge holding; judge every other criterion on the state before the test.")])
    if req.startswith("observe_now"):
        st = ep.snap("check observe")
        idx = len(ep.states) - 1
        return [CL.txt(f"CHECK observe_now (measured: {E.mstate(st)}):")] + [CL.img(ep.dir, st["frames"][c]) for c in cams] \
            + [CL.img(ep.dir, r) for _, r in E.crops_for(smp, idx, cams)]
    if req.startswith("show_view:"):
        cam = req.split(":", 1)[1].strip()
        if cam in ("agent", "top", "left", "right"):
            st = S[-1]
            return [CL.txt(f"CHECK show_view:{cam}"), CL.img(ep.dir, st["frames"][cam])]
    if req.startswith("move_arm_away:"):
        arm = req.split(":", 1)[1].strip()
        if arm in ("left", "right") and arm != smp["arm_main"]:
            tools._run(T.back_to_origin(ArmTag(arm)))
            st = ep.snap("check arm away")
            idx = len(ep.states) - 1
            return [CL.txt(f"CHECK move_arm_away:{arm} (measured: {E.mstate(st)}):")] + [CL.img(ep.dir, st["frames"][c]) for c in cams] \
                + [CL.img(ep.dir, r) for _, r in E.crops_for(smp, idx, cams)]
    return None


# ============================================================================ executor loop
SCHEMA_EXEC = None


def parse_exec(text):
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    i, j = t.find("{"), t.rfind("}")
    try:
        o = json.loads(t[i:j + 1])
        tc = o.get("tool_call") or {}
        if not isinstance(tc, dict) or not tc.get("name"):
            return ("invalid", 'the JSON has no "tool_call": {"name": ..., "arguments": {...}}'), o
        a = tc.get("arguments") or {}
        if isinstance(a, str):
            a = json.loads(a)
        return ("tool", tc["name"], a), o
    except Exception as exc:  # noqa: BLE001
        return ("invalid", f"answer is not one JSON object ({exc})"), None


def agent_loop(ep, tools, instr, demos):
    mem = VB.Memory()
    mem.M["notes"] = []
    ep.mem = mem
    history, replans = [], 0
    subs = CL.plan_call(ep, instr, demos)
    for k, s in enumerate(subs):
        s["id"] = k + 1
    while True:
        if not subs:
            return
        failed = False
        for k, sub in enumerate(subs):
            if sub["id"] in mem.M["confirmed_subgoals"]:
                continue
            res = run_subgoal(ep, tools, instr, sub, k, len(subs), demos, mem, subs[k + 1] if k + 1 < len(subs) else None)
            history.append(f"- subgoal '{sub['goal']}': {res}")
            if res != "done":
                failed = True
                break
        replans += 1
        if replans > 3:
            return
        mem.M["confirmed_subgoals"] = []
        subs = CL.plan_call(ep, instr, demos, history + (["(all subgoals confirmed by the checker)"] if not failed else []))
        for k, s in enumerate(subs):
            s["id"] = k + 1


HOME_RULE = ""      # r3 cond S6 = S5 + this rule in the executor system prompt (the expert's behaviour, stated)
HOME_RULE_TEXT = ("\n\nArm hand-off rule (what the expert does): after a successful place/release, if the NEXT subgoal uses the "
                  "OTHER arm, call home(<the arm that just placed>) BEFORE request_check, so the placing arm is out of the way. "
                  "If the same arm works next, request_check directly.")
NEXT_LINE = False   # r3 cond S5: the executor also sees the next subgoal (as in the r3 training data)


def run_subgoal(ep, tools, instr, sub, k, n, demos, mem, nxt=None):
    S_start = ep.states[-1]
    start_idx = len(ep.states) - 1
    frames = [start_idx]
    cams = CL.cams_for(sub["arm"])
    first = (f"Overall task: {instr}\nCurrent subgoal (id {sub['id']}, {k + 1}/{n}): {sub['goal']} (arm: {sub['arm']})\n"
             + ((f"Next subgoal: {nxt['goal']} (arm: {nxt['arm']})\n" if nxt else "Next subgoal: none (this is the last one)\n") if NEXT_LINE else "")
             + f"Success criteria: {'; '.join(sub['success'])}\nObjects in the scene: {', '.join(ep.sim.names)}\n"
             + C.arm_state_text(S_start) + "\n" + ep.oracle_text(S_start) + "\n" + VB.memory_text(mem.M))
    msgs = [{"role": "system", "content": VB.EXECUTOR_SYSTEM_V3 + HOME_RULE + demos["executor"]},
            {"role": "user", "content": limgs(ep, S_start, cams) + [CL.txt(first)]}]
    invalid, steps, rejected = 0, 0, 0
    while True:
        reply = ep.llm.chat("executor", CL.api_msgs(msgs), max_tokens=700)
        msgs.append({"role": "assistant", "content": reply})
        act, obj = parse_exec(reply)
        if obj:
            for d in (obj.get("memory_delta") or [])[:2]:
                mem.M["notes"] = (mem.M["notes"] + [str(d)[:100]])[-6:]
        if act[0] == "invalid":
            invalid += 1
            ep.trace({"kind": "invalid", "why": act[1]})
            if invalid > 3:
                return "failed (format)"
            msgs.append({"role": "user", "content": f"Format error: {act[1]}. Answer with one JSON object as specified."})
            continue
        name, args = act[1], act[2]
        if name == "request_check":
            sub2 = dict(sub)
            if sub2["arm"] == "both":
                sub2["arm_main"] = next((t["args"].get("arm") for t in reversed(ep.tool_log) if t["name"] in ("grasp", "place")), "right") if getattr(ep, "tool_log", None) else "right"
            # all stage snapshots since the subgoal started (the tools snapshot each stage: closed / lifted / released /
            # retreated), without the checker's own check snapshots -> the checker sees e.g. before grasp / closed / lifted,
            # the input format it was tuned on in experiment 1
            stage = [start_idx] + [j for j in range(start_idx + 1, len(ep.states)) if not str(ep.states[j]["label"]).startswith("check")]
            v, nc = checker(ep, sub2, stage[-3:], tools)
            ep.trace({"kind": "checker", "label": v["checker"], "verdict": v, "subgoal": sub})
            if v["checker"] == "satisfied":
                mem.confirm(sub["id"])
                return "done"
            rejected += 1
            if rejected > 3:
                return "checker rejected 4 times"
            msgs.append({"role": "user", "content": [CL.txt("CHECKER: " + json.dumps(v, ensure_ascii=False) + "\n" + VB.memory_text(mem.M))]})
            continue
        steps += 1
        try:
            fb, S = tools.call(name, args)
        except (CL.Success, CL.Budget):
            raise
        except (TypeError, ValueError, KeyError) as exc:
            fb, S = {"ok": False, "stage": "error", "note": f"bad call: {exc}"}, ep.states[-1]
        except Exception as exc:  # noqa: BLE001
            ep.sim.T.plan_success = True
            msg = str(exc) or type(exc).__name__
            if "target_pose cannot be None" in msg:
                msg = f"no reachable pose for this {name} with the {args.get('arm', '?')} arm"
            fb, S = {"ok": False, "stage": "error", "note": f"{name} failed: {msg}"}, ep.states[-1]
        ep.tool_log = getattr(ep, "tool_log", []) + [{"name": name, "args": args}]
        mem.after_tool({"tool": name, "args": args, "feedback": fb}, S)
        frames.append(len(ep.states) - 1)
        ep.trace({"kind": "tool", "name": name, "args": args, "feedback": fb})
        content = [CL.txt("TOOL RESULT: " + json.dumps(fb, ensure_ascii=False) + "\n" + VB.memory_text(mem.M))]
        if name != "locate" and fb.get("stage") != "error":
            content = [CL.txt("TOOL RESULT:")] + limgs(ep, S, cams) + [CL.txt(json.dumps(fb, ensure_ascii=False) + "\n"
                                                                            + ep.oracle_text(S) + "\n" + VB.memory_text(mem.M))]
        msgs.append({"role": "user", "content": content})
        if steps > 25:
            return "too many steps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cond", choices=["A", "B", "C", "S1", "S2", "S3", "S4", "S5", "S6", "X6", "N6"], required=True)
    ap.add_argument("--episodes", default="0-9")
    ap.add_argument("--api", default=_paths.DEFAULT_URL)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    CL.OUT = OUT
    CL.agent_loop = agent_loop
    CL.Tools = Tools3
    sim = CL.SimEnv(a.task, seed_dir=str(SEEDS), task_config="demo_randomized", seed_base_index=-1)
    lo, hi = [int(x) for x in a.episodes.split("-")]
    global NEXT_LINE, HOME_RULE
    NEXT_LINE = a.cond in ("S5", "S6", "X6", "N6")
    HOME_RULE = HOME_RULE_TEXT if a.cond in ("S6", "X6", "N6") else ""
    demos = demo_text(a.task, a.cond)
    demos.setdefault("checker", "")
    global CHECKER_DEMO
    CHECKER_DEMO = demos["checker"]
    if a.out:
        OUT_ = Path(a.out)
    else:
        OUT_ = OUT
    for i in range(lo, hi + 1):
        od = OUT_ / a.cond / a.task / f"episode_{i:03d}"
        if (od / "result.json").exists():
            continue
        r = CL.run_episode(sim, a.task, a.cond, i, od, a.api, "qwen", demos)
        print(json.dumps({k: r.get(k) for k in ("task", "condition", "episode_index", "seed", "success", "finished_reason", "model_calls", "seconds")}), flush=True)
    sim.rd.close()


if __name__ == "__main__":
    main()
