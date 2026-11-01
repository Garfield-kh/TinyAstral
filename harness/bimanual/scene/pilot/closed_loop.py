#!/usr/bin/env python
"""Closed-loop check of the v2 oracle data: Flash runs the v2 harness (planner -> executor -> checker, tools v2 executed
in RoboTwin) on held-out seeds, with oracle object state in the text.

Conditions (only the system prompts differ):
  A  zero-shot
  B  + one complete demonstration of the same task from oracle_expert_pilot_v2 (planner, executor, checker turns,
     with our synthesised thinks; images omitted)
  C  same as B but the <think> blocks of the demonstration removed (actions + feedback only)
Budget (UniBench oracle rule): 120 model calls per episode; wall clock 3600 s only as a safety cap (infra_timeout).
Success = RoboTwin check_success(), checked after every tool call; the episode ends at the first success.
Seeds: held-out RoboTwin collection seeds validated by the expert from 2000 on (v2 data used < 1500; eval seeds >= 100000).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import record as R1  # noqa: E402  (RoboTwin paths, SimEnv)
import record_v2 as RV2  # noqa: E402
import common_v2 as C  # noqa: E402
import clean_v2  # noqa: E402
from record import SimEnv, ArmTag, rename_thin  # noqa: E402
import transforms3d as t3d  # noqa: E402

import paths as _paths  # noqa: E402
V2 = _paths.OUT / "oracle_expert_pilot_v2"
OUT = V2 / "closed_loop"
MAX_CALLS, WALL_S, KEEP_IMAGES = 120, 3600, 6
HELD_OUT_START = 2000


# ============================================================================ LLM
class LLM:
    def __init__(self, api, model, trace):
        self.api, self.model, self.trace = api, model, trace
        self.calls = 0
        self.prompt_tokens = self.completion_tokens = 0

    def chat(self, role, messages, max_tokens=1500):
        if self.calls >= MAX_CALLS:
            raise Budget("max_calls")
        self.calls += 1
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}}
        t0, err, text, usage = time.time(), None, "", {}
        for att in range(4):
            try:
                req = urllib.request.Request(self.api + "/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
                r = json.load(urllib.request.urlopen(req, timeout=900))
                text = r["choices"][0]["message"].get("content") or ""
                usage = r.get("usage") or {}
                err = None
                break
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                time.sleep(5 * (att + 1))
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.trace({"kind": "model_call", "role": role, "call": self.calls, "seconds": round(time.time() - t0, 1),
                    "input": strip_images(messages), "output": text, "error": err, "usage": usage})
        if err:
            raise Budget("error: " + err)
        return text


class Budget(Exception):
    pass


class Success(Exception):
    pass


def strip_images(messages):
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = [{"type": "image", "file": p.get("_file", "")} if p.get("type") == "image_url" else p for p in c]
        out.append({"role": m["role"], "content": c})
    return out


def img(base: Path, rel: str):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode((base / rel).read_bytes()).decode()},
            "_file": rel}


def api_msgs(msgs):
    """drop the private _file key and keep only the newest KEEP_IMAGES images."""
    cnt, out = 0, []
    for m in reversed(msgs):
        c = m["content"]
        if isinstance(c, list):
            cc = []
            for p in reversed(c):
                if p.get("type") == "image_url":
                    cnt += 1
                    if cnt > KEEP_IMAGES:
                        continue
                    p = {"type": "image_url", "image_url": p["image_url"]}
                cc.append(p)
            c = list(reversed(cc)) or [{"type": "text", "text": "(images omitted)"}]
        out.append({"role": m["role"], "content": c})
    return list(reversed(out))


def txt(t):
    return {"type": "text", "text": t}


def parse_json(t):
    t = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S)
    i, j = t.find("{"), t.rfind("}")
    return json.loads(t[i:j + 1])


_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_exec(text):
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = _TC.search(t) or re.search(r"(\{\s*\"name\"\s*:.*\})", t, re.S)
    if m:
        try:
            o = json.loads(m.group(1))
            a = o.get("arguments") or {}
            if isinstance(a, str):
                a = json.loads(a)
            return ("tool", o.get("name"), a)
        except Exception as exc:  # noqa: BLE001
            return ("invalid", f"tool_call is not valid JSON ({exc})")
    if re.search(r"subgoal\s+complete", t, re.I):
        return ("complete",)
    if re.search(r"subgoal\s+failed", t, re.I):
        return ("failed",)
    return ("invalid", 'no <tool_call>{...}</tool_call> and no "Subgoal complete." / "Subgoal failed."')


# ============================================================================ demonstrations
def demo_block(task, cond):
    if cond == "A":
        return {"planner": "", "executor": "", "checker": ""}
    plan = [json.loads(l) for l in open(V2 / "planner.jsonl")]
    exe = [json.loads(l) for l in open(V2 / "executor.jsonl")]
    chk = [json.loads(l) for l in open(V2 / "checker.jsonl")]
    cands = sorted(r["episode_id"] for r in plan if r["task"] == task and not r["perturbed"])
    eid = cands[0]
    strip = (lambda s: re.sub(r"<think>.*?</think>\s*", "", s, flags=re.S)) if cond == "C" else (lambda s: s)

    def text_of(c):
        if isinstance(c, str):
            return c
        return "\n".join(p["text"] for p in c if p.get("type") == "text") + ("" if all(p.get("type") == "text" for p in c) else "\n[images omitted]")
    p = next(r for r in plan if r["episode_id"] == eid)
    pb = ("\n\n# Example: a successful episode of the same task (scene differs; images omitted)\nUSER: "
          + text_of(p["messages"][1]["content"]) + "\nASSISTANT: " + p["messages"][2]["content"])
    eb = ["\n\n# Example: how a successful episode of the same task was executed (scene differs; images omitted). "
          "Each block is one subgoal conversation."]
    for r in sorted((r for r in exe if r["episode_id"] == eid), key=lambda r: r["subgoal_index"]):
        eb.append(f"\n## Subgoal {r['subgoal_index'] + 1}")
        for m in r["messages"][1:]:
            if m["role"] == "user":
                eb.append("USER: " + text_of(m["content"]))
            elif m["role"] == "assistant":
                eb.append("ASSISTANT: " + strip(m["content"]))
            elif m["role"] == "tool":
                eb.append("TOOL RESULT: " + text_of(m["content"]))
    cb = ["\n\n# Examples from a successful episode of the same task (images omitted)"]
    for r in [r for r in chk if r["episode_id"] == eid][:4]:
        cb.append("USER: " + text_of(r["messages"][1]["content"]) + "\nASSISTANT: " + r["messages"][2]["content"])
    return {"planner": pb, "executor": "\n".join(eb), "checker": "\n".join(cb), "demo_episode": eid}


# ============================================================================ tools v2 in the simulator
class Tools:
    def __init__(self, ep):
        self.ep = ep
        self.sim, self.T = ep.sim, ep.sim.T

    def _run(self, action):
        self.T.plan_success = True
        if action is None or action[0] is None or not action[1]:
            return False
        self.T.move(action)
        ok = bool(self.T.plan_success)
        self.T.plan_success = True
        return ok

    def _state(self):
        return self.ep.snap("tool")

    def held(self, S, arm):
        return S["held"].get(arm) if S["arms"][arm]["gripper_cmd"] <= 0.8 else None

    def call(self, name, args):
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        fn = getattr(self, "t_" + str(name), None)
        if fn is None:
            raise ValueError(f"unknown tool '{name}' (tools: locate, grasp, place, move, rotate, gripper, home)")
        return fn(**args)

    def _arm(self, arm):
        if arm not in ("left", "right"):
            raise ValueError('arm must be "left" or "right"')
        return ArmTag(arm)

    def _obj(self, name):
        if name not in self.sim.names:
            raise ValueError(f"unknown object '{name}' (objects: {', '.join(self.sim.names)})")
        return self.sim.names[name]

    def t_locate(self, object, part=None):
        self._obj(object)
        S = self.ep.states[-1]
        note, res = R1.locate_result(S, object)
        return RV2.fb(S, True, "done", note, {"object": res}), S

    def t_grasp(self, arm, object, part=None, lift_cm=None):
        tag, actor = self._arm(arm), self._obj(object)
        S0 = self.ep.states[-1]
        if S0["arms"][arm]["gripper_cmd"] < 0.8 and not self.held(S0, arm):
            self._run(self.T.open_gripper(tag))
        cp = None
        if part in ("upper half", "lower half", "upper", "lower", "top", "bottom"):
            c = np.asarray(actor.get_pose().p)
            ids = [i for i, p in actor.iter_contact_points("list")]
            pts = {i: np.asarray(p)[:3] for i, p in actor.iter_contact_points("list")}
            up = part in ("upper half", "upper", "top")
            sel = [i for i in ids if (pts[i][2] >= c[2]) == up]
            cp = sel or None
        act = self.T.grasp_actor(actor, tag, pre_grasp_dis=0.1, grasp_dis=0, contact_point_id=cp)
        if act is None or act[0] is None or not act[1]:
            self.T.plan_success = True
            S = self._state()
            return RV2.fb(S, False, "approach", f"no reachable grasp pose for {object} with the {arm} arm"), S
        acts = list(act[1])
        if not self._run((act[0], acts[:1])):
            S = self._state()
            return RV2.fb(S, False, "approach", f"could not reach the pre-grasp pose of {object}"), S
        ok = self._run((act[0], acts[1:]))
        S = self._state()
        holding = self.held(S, arm) == object
        if not holding:
            touch = [o for a, o in S["contacts"] if a == arm]
            note = (f"closed, but only one finger touches {object}: not held; lift skipped" if object in touch else
                    f"closed on nothing: {object} is not between the fingers; lift skipped")
            if not ok:
                note = "motion planner failed on the final approach; " + note
            return RV2.fb(S, False, "closed", note, {"lifted_cm": 0.0}), S
        lift = 8.0 if lift_cm is None else float(lift_cm)
        if lift > 0:
            self._run(self.T.move_by_displacement(tag, z=lift / 100.0))
            S = self._state()
            d = S["objects"][object]["lo"][2] - S0["objects"][object]["lo"][2]
            return RV2.fb(S, self.held(S, arm) == object, "lifted", f"grasped {object} with the {arm} gripper and lifted it {d:.0f} cm",
                          {"lifted_cm": round(max(d, 0), 1)}), S
        return RV2.fb(S, True, "closed", f"grasped {object} with the {arm} gripper (no lift requested)", {"lifted_cm": 0.0}), S

    def t_place(self, arm, object, target, relation="on", release=True, retreat=True, **kw):
        tag, actor = self._arm(arm), self._obj(object)
        S0 = self.ep.states[-1]
        if self.held(S0, arm) != object:
            return RV2.fb(S0, False, "reach", f"the {arm} arm is not holding {object}"), S0
        bias = float(getattr(self.T, "table_z_bias", 0.0))
        tz = 0.74 + bias
        params = dict(functional_point_id=0, pre_dis=0.05, dis=0.0, is_open=bool(release), pre_dis_axis="fp")
        tgt = str(target)
        rel_expr = None
        if tgt == "handover_point":
            tp = [0.0, 0.0, 0.90, 0, 1, 0, 0]
            params.update(pre_dis=0.0, dis=0.0, constrain="free")
            params.pop("pre_dis_axis")
        elif tgt == "table_center":
            tp = [0.0, -0.13, 0.75 + bias, 0, 1, 0, 0]
            rel_expr = f"at({object}, 0, -13)"
        elif tgt == "table_free_spot":
            tp = [-0.2 if arm == "left" else 0.2, -0.1, tz + 0.01, 0, 1, 0, 0]
            rel_expr = f"on({object}, table)"
        elif tgt.startswith("beside:"):
            b = self._obj(tgt[7:])
            p = np.asarray(b.get_pose().p)
            tp = [p[0] + (-0.12 if arm == "left" else 0.12), p[1], tz + 0.01, 0, 1, 0, 0]
            rel_expr = f"on({object}, table)"
        else:
            tb = self._obj(tgt)
            fps = []
            for i in range(8):
                try:
                    fp = tb.get_functional_point(i, "pose")
                except Exception:  # noqa: BLE001
                    break
                if fp is None:
                    break
                fps.append((i, fp))
            if fps:
                i, pose = max(fps, key=lambda x: x[1].p[2])
                tp = pose
            else:
                lo, hi = self.sim.aabb(tgt)
                tp = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2], 0, 1, 0, 0]
            rel_expr = f"on({object}, {tgt})"
        act = self.T.place_actor(actor, tag, target_pose=tp, **params)
        if act is None or act[0] is None or not act[1]:
            self.T.plan_success = True
            S = self._state()
            return RV2.fb(S, False, "reach", f"no reachable place pose for {object} at {tgt}"), S
        acts = list(act[1])
        n_move = sum(1 for a in acts if a.action == "move")
        if not self._run((act[0], acts[:n_move])):
            S = self._state()
            return RV2.fb(S, False, "reach", f"could not reach the place pose at {tgt}"), S
        if not release:
            S = self._state()
            ok = self.held(S, arm) == object
            return RV2.fb(S, ok, "reached", f"{object} is at the {tgt.replace('_', ' ')} and still held" if ok else f"{object} was dropped"), S
        self._run((act[0], acts[n_move:]))
        S = self._state()
        rv = C.evaluate(rel_expr, S, self.ep.states[0])[0] if rel_expr else True
        stage = "released"
        if retreat is not False:
            self._run(self.T.move_by_displacement(tag, z=0.07))
            S = self._state()
            stage = "retreated"
        note = f"put {object} on {tgt.replace('_', ' ')}, released it" + (" and backed off" if stage == "retreated" else "") if rv else \
            f"released {object} but it is not resting on {tgt.replace('_', ' ')}"
        return RV2.fb(S, bool(rv), stage, note, {"relation": f"{rel_expr} = {str(bool(rv)).lower()}"}), S

    def t_move(self, arm, dx=0, dy=0, dz=0):
        tag = self._arm(arm)
        d = [float(dx or 0), float(dy or 0), float(dz or 0)]
        if max(abs(x) for x in d) > 40:
            raise ValueError("each offset must be within +-40 cm")
        ok = self._run(self.T.move_by_displacement(tag, x=d[0] / 100, y=d[1] / 100, z=d[2] / 100))
        S = self._state()
        h = self.held(S, arm)
        return RV2.fb(S, ok, "done", (f"moved the {arm} fingertips by ({d[0]:+.0f}, {d[1]:+.0f}, {d[2]:+.0f}) cm" if ok
                                      else "motion planner could not reach that offset") + (f"; holding {h}" if h else "")), S

    def t_rotate(self, arm, axis, deg):
        tag = self._arm(arm)
        ax = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1], "roll": [1, 0, 0], "pitch": [0, 1, 0], "yaw": [0, 0, 1]}.get(str(axis))
        if ax is None:
            raise ValueError('axis must be "x", "y" or "z"')
        pose = np.asarray(self.T.get_arm_pose(tag), float)
        Rm = t3d.axangles.axangle2mat(ax, math.radians(float(deg))) @ t3d.quaternions.quat2mat(pose[3:])
        tcp = R1.wrist_to_tcp(pose)
        q = t3d.quaternions.mat2quat(Rm)
        wrist = tcp - Rm[:, 0] * 0.12
        ok = self._run(self.T.move_to_pose(tag, np.concatenate([wrist, q])))
        S = self._state()
        return RV2.fb(S, ok, "done", f"rotated the {arm} gripper {float(deg):+.0f} deg about {axis}" if ok else "could not rotate there"), S

    def t_gripper(self, arm, action):
        tag = self._arm(arm)
        if action not in ("open", "close"):
            raise ValueError('action must be "open" or "close"')
        S0 = self.ep.states[-1]
        self._run(self.T.open_gripper(tag) if action == "open" else self.T.close_gripper(tag))
        S = self._state()
        if action == "open":
            lost = self.held(S0, arm)
            note = f"{arm} gripper opened" + (f"; let go of {lost}" if lost else "")
        else:
            h = S["held"].get(arm)
            note = f"{arm} gripper closed" + (f" on {h}" if h else ": nothing held")
        return RV2.fb(S, True, "done", note), S

    def t_home(self, arm):
        tag = self._arm(arm)
        ok = self._run(self.T.back_to_origin(tag))
        S = self._state()
        return RV2.fb(S, ok, "done", f"{arm} arm is at its start pose" if ok else "could not move home"), S


# ============================================================================ episode
class Episode:
    def __init__(self, sim, task, cond, idx, outdir, llm_api, model):
        self.sim, self.task, self.cond, self.idx = sim, task, cond, idx
        self.dir = outdir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "trace.jsonl").write_text("")
        self.llm = LLM(llm_api, model, self.trace)
        self.states = []
        self.rec = None
        self.t0 = time.time()

    def trace(self, rec):
        rec["t"] = round(time.time() - self.t0, 1)
        with open(self.dir / "trace.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def snap(self, label):
        idx = self.rec.snap(label)
        st = self.rec.states[idx]
        self.states = self.rec.states
        self.trace({"kind": "state", "idx": idx, "label": label, "frames": st["frames"], "held": st["held"],
                    "objects": {n: o["center"] for n, o in st["objects"].items()},
                    "arms": {a: st["arms"][a] for a in st["arms"]}})
        if self.sim.T.check_success():
            self.trace({"kind": "success", "state": idx})
            raise Success()
        if time.time() - self.t0 > WALL_S:
            raise Budget("infra_timeout")
        return st

    def oracle_text(self, S):
        lines = [f"- {n}: centre ({o['center'][0]:.1f}, {o['center'][1]:.1f}, {o['center'][2]:.1f}), size "
                 f"({o['size'][0]:.0f}, {o['size'][1]:.0f}, {o['size'][2]:.0f}), top z {o['hi'][2]:.1f}" for n, o in S["objects"].items()]
        return "OBJECTS (exact simulator state in cm; z = height above the table):\n" + "\n".join(lines)

    def imgs(self, S, cams):
        return [img(self.dir, S["frames"][c]) for c in cams if c in S["frames"]]


def run_episode(sim, task, cond, idx, outdir, api, model, demos):
    ep = Episode(sim, task, cond, idx, outdir, api, model)
    info = sim.rd.reset_episode(idx)
    sim.T = sim.rd.env
    T = sim.T
    T.eval_success = False
    T.plan_success = True
    sim._fingers()
    sim._build_names()
    rename_thin(sim)
    ep.rec = RV2.RecorderV2(sim, outdir, task, info.seed, False, np.random.RandomState(0))
    ep.rec.fdir = outdir / "frames"
    ep.rec.fdir.mkdir(parents=True, exist_ok=True)
    ep.rec.ep_id = "."
    # frames are stored as frames/./NNN_cam.jpg relative to the episode dir
    instr0 = info.instruction
    result = {"episode_index": idx, "seed": info.seed, "task": task, "condition": cond, "instruction_original": instr0,
              "demo_episode": demos.get("demo_episode"), "success": False, "finished_reason": None}
    tools = Tools(ep)
    try:
        S0 = ep.snap("start")
        fake = {"states": [S0], "objects": list(sim.names), "calls": [], "seed": info.seed}
        d, cf, _m = clean_v2.check(task, instr0, fake)
        instr = instr0
        if d or cf:
            pool = clean_v2.all_candidates(task, info.expert_info or {}, info.seed)
            ok = [c for c in pool if not any(clean_v2.check(task, c, fake)[:2])]
            if task == "handover_block" and any("pad" in c.lower() for c in ok):
                ok = [c for c in ok if "pad" in c.lower()]
            if ok:
                instr = str(np.random.RandomState(info.seed).choice(sorted(set(ok))))
        result.update(instruction=instr, instruction_fixed=instr != instr0)
        ep.trace({"kind": "episode", **result})
        T.set_instruction(instruction=instr)
        agent_loop(ep, tools, instr, demos)
        result["finished_reason"] = "agent_done"
    except Success:
        result.update(success=True, finished_reason="success")
    except Budget as b:
        result["finished_reason"] = "max_calls" if str(b) == "max_calls" else ("infra_timeout" if "infra_timeout" in str(b) else "error")
        result["error"] = str(b)
    except Exception as exc:  # noqa: BLE001
        result.update(finished_reason="error", error=f"{type(exc).__name__}: {exc}")
        ep.trace({"kind": "error", "traceback": traceback.format_exc()})
    result.update(seconds=round(time.time() - ep.t0, 1), model_calls=ep.llm.calls, prompt_tokens=ep.llm.prompt_tokens,
                  completion_tokens=ep.llm.completion_tokens, agent_claimed_done=result["finished_reason"] == "agent_done",
                  n_states=len(ep.states))
    ep.trace({"kind": "result", **result})
    (outdir / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    try:
        sim.rd.close_episode()
    except Exception:  # noqa: BLE001
        pass
    return result


def cams_for(arm):
    return ["agent"] + (["left", "right"] if arm == "both" else [arm if arm in ("left", "right") else "left"])


def plan_call(ep, instr, demos, history=None):
    S = ep.states[-1]
    text = C.planner_user_text(instr, list(ep.sim.names), S) + "\n" + ep.oracle_text(S)
    if history:
        text += ("\n\nThe previous plan did not finish the task. What happened so far:\n" + "\n".join(history)
                 + '\nWrite the REMAINING subgoals from the current scene. If the task is already complete, answer {"analysis": "...", "subgoals": []}.')
    msgs = [{"role": "system", "content": C.PLANNER_SYSTEM + demos["planner"]},
            {"role": "user", "content": ep.imgs(S, ["agent", "top", "left", "right"]) + [txt(text)]}]
    for att in range(3):
        reply = ep.llm.chat("planner", api_msgs(msgs), max_tokens=1500)
        try:
            obj = parse_json(reply)
            subs = obj.get("subgoals")
            if not isinstance(subs, list):
                raise ValueError('"subgoals" must be a list')
            out = []
            for s in subs:
                crit = s.get("success") or []
                crit = [crit] if isinstance(crit, str) else crit
                good = []
                for c in crit:
                    try:
                        C.validate(c, list(ep.sim.names))
                        good.append(c)
                    except Exception:  # noqa: BLE001
                        pass
                out.append({"goal": str(s.get("goal", "")), "arm": s.get("arm") if s.get("arm") in ("left", "right", "both") else "both",
                            "success": good})
            return out
        except Exception as exc:  # noqa: BLE001
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"Invalid answer ({exc}). Answer with one JSON object as specified."}]
    return []


def check_call(ep, sub, S_before, demos):
    S = ep.states[-1]
    cams = ["agent", "top"] + (["left", "right"] if sub["arm"] == "both" else [sub["arm"]])
    text = (f"Subgoal: {sub['goal']} (arm: {sub['arm']})\nCompletion criteria: {'; '.join(sub['success']) or '(none given: judge the subgoal itself)'}\n"
            + ep.oracle_text(S))
    msgs = [{"role": "system", "content": C.CHECKER_SYSTEM + demos["checker"]},
            {"role": "user", "content": [txt("BEFORE:")] + ep.imgs(S_before, cams) + [txt("AFTER (now):")] + ep.imgs(S, cams) + [txt(text)]}]
    reply = ep.llm.chat("checker", api_msgs(msgs), max_tokens=1200)
    try:
        o = parse_json(reply)
        lab = o.get("subgoal", "unsatisfied")
        bad = [f"{c.get('criterion')}: {c.get('status')} ({c.get('evidence', '')})" for c in o.get("criteria", []) if c.get("status") != "satisfied"]
        return lab, bad
    except Exception:  # noqa: BLE001
        return "unsatisfied", ["checker answer could not be parsed"]


def agent_loop(ep, tools, instr, demos):
    history, replans = [], 0
    subs = plan_call(ep, instr, demos)
    while True:
        if not subs:
            return                                   # planner says done / gave up
        failed = False
        for k, sub in enumerate(subs):
            res = run_subgoal(ep, tools, instr, sub, k, len(subs), demos)
            history.append(f"- subgoal '{sub['goal']}': {res}")
            if res != "done":
                failed = True
                break
        replans += 1
        if replans > 3:
            return
        subs = plan_call(ep, instr, demos, history + (["(all subgoals reported done)"] if not failed else []))


def run_subgoal(ep, tools, instr, sub, k, n, demos):
    S_start = ep.states[-1]
    cams = cams_for(sub["arm"])
    first = C.executor_first_text(instr, k + 1, n, sub, list(ep.sim.names), S_start) + "\n" + ep.oracle_text(S_start)
    msgs = [{"role": "system", "content": C.EXECUTOR_SYSTEM + demos["executor"]},
            {"role": "user", "content": ep.imgs(S_start, cams) + [txt(first)]}]
    rejections, invalid, steps = 0, 0, 0
    while True:
        reply = ep.llm.chat("executor", api_msgs(msgs), max_tokens=800)
        msgs.append({"role": "assistant", "content": reply})
        act = parse_exec(reply)
        if act[0] == "invalid":
            invalid += 1
            ep.trace({"kind": "invalid", "why": act[1]})
            if invalid > 3:
                return "failed (format)"
            msgs.append({"role": "user", "content": f"Format error: {act[1]}. Reply with <think>...</think> then exactly one "
                                                    "<tool_call>{...}</tool_call>, or a line \"Subgoal complete.\" / \"Subgoal failed.\""})
            continue
        if act[0] == "failed":
            return "executor declared failure"
        if act[0] == "complete":
            lab, bad = check_call(ep, sub, S_start, demos)
            ep.trace({"kind": "checker", "label": lab, "bad": bad, "subgoal": sub})
            if lab == "satisfied":
                return "done"
            rejections += 1
            if rejections > 2:
                return "checker rejected completion 3 times"
            msgs.append({"role": "user", "content": "Checker: the subgoal is NOT complete yet: " + "; ".join(bad) + ". Continue."})
            continue
        steps += 1
        name, args = act[1], act[2]
        try:
            fb, S = tools.call(name, args)
        except (Success, Budget):
            raise
        except (TypeError, ValueError, KeyError) as exc:
            fb, S = {"ok": False, "stage": "error", "note": f"bad call: {exc}"}, ep.states[-1]
        except Exception as exc:  # noqa: BLE001  (e.g. RoboTwin finds no reachable grasp/place pose)
            ep.sim.T.plan_success = True
            msg = str(exc) or type(exc).__name__
            if "target_pose cannot be None" in msg:
                msg = f"no reachable pose for this {name} with the {args.get('arm', '?')} arm"
            fb, S = {"ok": False, "stage": "error", "note": f"{name} failed: {msg}"}, ep.states[-1]
            fb.update(holding={a: ep.states[-1]["held"].get(a) for a in ("left", "right")},
                      tcp={a: ep.states[-1]["arms"][a]["tcp"] for a in ("left", "right")})
        ep.trace({"kind": "tool", "name": name, "args": args, "feedback": fb})
        content = [txt("TOOL RESULT: " + json.dumps(fb, ensure_ascii=False))]
        if name != "locate" and fb.get("stage") != "error":
            content = [txt("TOOL RESULT:")] + ep.imgs(S, cams) + [txt(json.dumps(fb, ensure_ascii=False) + "\n" + ep.oracle_text(S))]
        msgs.append({"role": "user", "content": content})
        if steps > 25:
            return "too many steps"


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cond", choices=["A", "B", "C"])
    ap.add_argument("--episodes", default="0-9")
    ap.add_argument("--api", default=_paths.DEFAULT_URL)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--validate", type=int, default=0, help="only validate N held-out seeds (fills the seed cache)")
    a = ap.parse_args()
    sim = SimEnv(a.task, seed_dir=str(OUT / "seedcache"), task_config="demo_randomized", seed_base_index=-1)
    sc = sim.rd.seed_cache
    if not sc.get("seeds") or sc.get("next_seed", 0) < HELD_OUT_START:
        sim.rd.seed_cache = {"seed_base": 0, "seeds": [], "infos": [], "next_seed": HELD_OUT_START}
    if a.validate:
        sim.rd.validate_seeds(a.validate)
        print("validated", sim.rd.seed_cache["seeds"], flush=True)
        return
    lo, hi = [int(x) for x in a.episodes.split("-")]
    demos = demo_block(a.task, a.cond)
    for i in range(lo, hi + 1):
        od = OUT / a.cond / a.task / f"episode_{i:03d}"
        if (od / "result.json").exists():
            continue
        r = run_episode(sim, a.task, a.cond, i, od, a.api, a.model, demos)
        print(json.dumps({k: r[k] for k in ("task", "condition", "episode_index", "seed", "success", "finished_reason", "model_calls", "seconds")}), flush=True)
    sim.rd.close()


if __name__ == "__main__":
    main()
