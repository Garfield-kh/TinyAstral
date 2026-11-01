"""cot_v1 shared pieces: V+ (VP2f) message rendering, LLM client, action tolerance, leak / sim-state audits.
Pure python (no RoboTwin import) so the LLM synthesis can run anywhere."""
from __future__ import annotations

import base64
import json
import os
import math
import random
import re
import threading
import time
import urllib.request
from pathlib import Path

P = Path(__file__).resolve().parent
D = P
FROZEN_SYS = (D / "frozen/executor_system_vp2f.txt").read_text()
if os.environ.get("COT_FIXED_TOOLS") == "1":
    from tool_spec import TOOL_DOC
    FROZEN_SYS += TOOL_DOC
VIEW = {"agent": "overview", "left": "left wrist", "right": "right wrist", "top": "top"}
KEEP_IMAGES = 6
TOOLS = ("locate", "grasp", "place", "move", "press", "rotate", "gripper", "home", "cmd", "open_articulated", "request_check")


# ------------------------------------------------------------------ text pieces identical to the live V+ harness
def arm_state_text(st):
    lines = []
    for arm in ("left", "right"):
        a = st["arms"][arm]
        g = "open" if a["gripper_cmd"] > 0.8 else ("closed" if a["gripper_cmd"] < 0.2 else "half-closed")
        h = st["held"].get(arm) or "nothing"
        lines.append(f"{arm} arm: fingertips at {a['tcp']} cm, gripper {g}, holding {h}")
    return "\n".join(lines)


def oracle_text(S):
    lines = [f"- {n}: centre ({o['center'][0]:.1f}, {o['center'][1]:.1f}, {o['center'][2]:.1f}), size "
             f"({o['size'][0]:.0f}, {o['size'][1]:.0f}, {o['size'][2]:.0f}), top z {o['hi'][2]:.1f}" for n, o in S["objects"].items()]
    return "OBJECTS (exact simulator state in cm; z = height above the table):\n" + "\n".join(lines)


def memory_text(M):
    return "MEMORY: " + json.dumps(M, ensure_ascii=False)


def live_memory(mem_before, notes):
    """the live harness memory = v3 Memory + the executor's memory_delta notes (last 6)."""
    m = {k: mem_before[k] for k in ("confirmed_subgoals", "holding", "grasp_params", "recent_failures")}
    m["notes"] = list(notes)[-6:]
    return m


def cams_for(arm):
    return ["agent"] + (["left", "right"] if arm == "both" else [arm if arm in ("left", "right") else "left"])


NUM_CM = re.compile(r"\b(\d+(?:\.\d+)?)\s*cm\b")


def mask_goal(goal):
    """subgoal text spelled from the expert's motions ("move down 4 cm") leaks action parameters -> numbers masked."""
    return NUM_CM.sub("a few cm", goal)


def goal_text(sd, instr):
    g = mask_goal(sd["goal"])
    if any("task_success" in str(c) for c in sd["success"]) and "[overall task:" not in g:
        g += f" [overall task: {instr}]"
    return g


# ------------------------------------------------------------------ images
_IMG = {}


def img(root, rel):
    key = (str(root), rel)
    if key not in _IMG:
        _IMG[key] = "data:image/jpeg;base64," + base64.b64encode((Path(root) / rel).read_bytes()).decode()
        if len(_IMG) > 4000:
            _IMG.pop(next(iter(_IMG)))
    return {"type": "image_url", "image_url": {"url": _IMG[key]}}


def txt(t):
    return {"type": "text", "text": t}


def limgs(root, S, cams, with_images=True):
    out = []
    for c in cams:
        if c in S["frames"]:
            out += [txt(VIEW.get(c, c) + ":")] + ([img(root, S["frames"][c])] if with_images else [txt("[image]")])
    return out


def keep_newest_images(msgs, n=KEEP_IMAGES):
    cnt, out = 0, []
    for m in reversed(msgs):
        c = m["content"]
        if isinstance(c, list):
            cc = []
            for p in reversed(c):
                if p.get("type") == "image_url":
                    cnt += 1
                    if cnt > n:
                        continue
                cc.append(p)
            c = list(reversed(cc)) or [txt("(images omitted)")]
        out.append({"role": m["role"], "content": c})
    return list(reversed(out))


# ------------------------------------------------------------------ episode -> V+ message stack
def first_text(E, conv, sd, S0, mem):
    sub = E["subgoals"]
    nx = next((s for s in sub if s["id"] == sd["id"] + 1), None)
    return (f"Overall task: {E['instruction']}\nCurrent subgoal (id {sd['id']}, {sd['id']}/{len(sub)}): {goal_text(sd, E['instruction'])} (arm: {sd['arm']})\n"
            + (f"Next subgoal: {goal_text(nx, E['instruction'])} (arm: {nx['arm']})\n" if nx else "Next subgoal: none (this is the last one)\n")
            + f"Success criteria: {'; '.join(sd['success'])}\nObjects in the scene: {', '.join(E['objects'])}\n"
            + arm_state_text(S0) + "\n" + oracle_text(S0) + "\n" + memory_text(mem))


def result_content(E, conv, t, mem, with_images=True):
    """what the live harness appends after the executor's call (role user)."""
    S = E["states"]
    sd = E["subgoals"][conv["subgoal"]]
    cams = cams_for(sd["arm"])
    if t["kind"] == "check":
        v = {k: t["verdict"][k] for k in ("checker", "criteria", "next_check")}
        return [txt("CHECKER: " + json.dumps(v, ensure_ascii=False) + "\n" + memory_text(mem))]
    fb = json.dumps(t["feedback"], ensure_ascii=False)
    if t["gt"]["name"] == "locate" or t["feedback"].get("stage") == "error":
        return [txt("TOOL RESULT: " + fb + "\n" + memory_text(mem))]
    Sa = S[t["after"]]
    return [txt("TOOL RESULT:")] + limgs(E["root"], Sa, cams, with_images) + [txt(fb + "\n" + oracle_text(Sa) + "\n" + memory_text(mem))]


def assistant_json(cot, call):
    o = {k: cot.get(k, [] if k in ("seen", "reported", "uncertain", "memory_delta") else "") for k in
         ("seen", "reported", "assessment", "uncertain", "memory_delta", "next")}
    if call is not None:
        o["tool_call"] = call
    return o


def stack(E, conv, i, cots, notes0, system=None, with_images=True, demo=""):
    """V+ message stack up to (not including) the executor answer at turn i of conv.
    cots[j] = final CoT of turn j (j < i); notes0 = memory notes before the conversation.
    Returns (msgs, notes_before_turn_i)."""
    S = E["states"]
    sd = E["subgoals"][conv["subgoal"]]
    T = conv["turns"]
    notes = list(notes0)
    s0 = T[0]["obs"]
    msgs = [{"role": "system", "content": (system or FROZEN_SYS) + demo},
            {"role": "user", "content": limgs(E["root"], S[s0], cams_for(sd["arm"]), with_images)
             + [txt(first_text(E, conv, sd, S[s0], live_memory(T[0]["memory_before"], notes)))]}]
    for j in range(i):
        c = cots[j]
        msgs.append({"role": "assistant", "content": json.dumps(assistant_json(c, T[j]["gt"]), ensure_ascii=False)})
        for d in (c.get("memory_delta") or [])[:2]:
            notes = (notes + [str(d)[:100]])[-6:]
        mem_after = T[j + 1]["memory_before"] if j + 1 < len(T) else T[j]["memory_before"]
        msgs.append({"role": "user", "content": result_content(E, conv, T[j], live_memory(mem_after, notes), with_images)})
    return msgs, notes


def prefix_text(msgs):
    """plain-text rendering of a stack (images -> [image]) for prompts that quote the context."""
    out = []
    for m in msgs[1:]:
        c = m["content"]
        if isinstance(c, list):
            c = "".join(p["text"] if p.get("type") == "text" else "[image]" for p in c)
        out.append(f"[{m['role'].upper()}] {c}")
    return "\n".join(out)


# ------------------------------------------------------------------ LLM client (round-robin over endpoints with caps)
class Pool:
    def __init__(self, caps):
        self.caps = dict(caps)
        self.sems = {a: threading.Semaphore(n) for a, n in caps.items()}
        self.lock = threading.Lock()
        self.busy = {a: 0 for a in caps}

    def pick(self):
        with self.lock:
            a = min(self.caps, key=lambda x: (self.busy[x] / self.caps[x], random.random()))
            self.busy[a] += 1
        return a

    def chat(self, messages, max_tokens=600, temperature=0.0, schema=None, prefill=None, tag=""):
        body = {"model": "qwen", "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False}}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        if prefill is not None:
            body["messages"] = messages + [{"role": "assistant", "content": prefill}]
            body["continue_final_message"] = True
            body["add_generation_prompt"] = False
        err = None
        for att in range(5):
            a = self.pick()
            try:
                with self.sems[a]:
                    req = urllib.request.Request(a + "/chat/completions", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
                    r = json.load(urllib.request.urlopen(req, timeout=600))
                return r["choices"][0]["message"].get("content") or "", r.get("usage") or {}
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                time.sleep(4 * (att + 1))
            finally:
                with self.lock:
                    self.busy[a] -= 1
        raise RuntimeError(err)


def pj(t):
    t = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S)
    i, j = t.find("{"), t.rfind("}")
    return json.loads(t[i:j + 1])


# ------------------------------------------------------------------ actions: normalisation and tolerance
def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def normalise(call):
    """-> canonical dict {name, arm, ...}; cmd strings are folded into move / rotate / gripper / home when possible."""
    if not isinstance(call, dict):
        return {"name": "invalid"}
    name = str(call.get("name") or "")
    a = call.get("arguments") or {}
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:  # noqa: BLE001
            a = {}
    if name == "cmd":
        cmds = a.get("commands") or []
        cmds = [cmds] if isinstance(cmds, str) else list(cmds)
        moves, other = {}, []
        arm = None
        for c in cmds:
            m = re.match(r"\s*(left|right)\s+move\s+([xyz])\s+([-+]?[\d.]+)", str(c), re.I)
            if m:
                arm = arm or m.group(1).lower()
                moves[m.group(2).lower()] = moves.get(m.group(2).lower(), 0) + float(m.group(3))
                continue
            other.append(str(c).strip().lower())
        if moves and not other:
            return {"name": "move", "arm": arm, "d": [moves.get("x", 0), moves.get("y", 0), moves.get("z", 0)]}
        if len(other) == 1 and not moves:
            m = re.match(r"(left|right)\s+gripper\s+(open|close)$", other[0])
            if m:
                return {"name": "gripper", "arm": m.group(1), "action": m.group(2)}
            m = re.match(r"(left|right)\s+home$", other[0])
            if m:
                return {"name": "home", "arm": m.group(1)}
            m = re.match(r"(left|right)\s+rotate\s+(roll|pitch|yaw)\s+([-+]?[\d.]+)$", other[0])
            if m:
                return {"name": "rotate", "arm": m.group(1), "axis": {"roll": "x", "pitch": "y", "yaw": "z"}[m.group(2)],
                        "deg": float(m.group(3))}
        return {"name": "cmd", "cmds": cmds}
    out = {"name": name}
    for k in ("arm", "object", "target", "action", "axis", "part"):
        if a.get(k) is not None:
            out[k] = str(a[k]).strip().lower().replace(" ", "_") if k in ("object", "target") else str(a[k]).strip().lower()
    if name == "move":
        out["d"] = [_f(a.get("dx")), _f(a.get("dy")), _f(a.get("dz"))]
        out["r"] = [_f(a.get(k)) for k in ("rx", "ry", "rz")]
    if name == "rotate":
        out["deg"] = _f(a.get("deg"))
        out["axis"] = {"roll": "x", "pitch": "y", "yaw": "z"}.get(out.get("axis"), out.get("axis"))
    if name == "grasp":
        lc = a.get("lift_cm")
        out["lift0"] = lc is not None and _f(lc, 1) == 0
        out["lift_cm"] = _f(lc, 8)
        out["approach_cm"] = _f(a.get("approach_cm"), 10)
        out["preshape"] = a.get("preshape")
        cp = a.get("contact")
        out["contact"] = sorted(cp if isinstance(cp, list) else [cp]) if cp is not None else None
    if name == "place":
        out["release"] = a.get("release") not in (False, "false", 0)
        out["relation"] = str(a.get("relation") or "on").lower()
    if name == "press":
        out["depth"] = _f(a.get("depth_cm"), 2)
    if name == "open_articulated":
        out["fraction"] = _f(a.get("fraction"), 0.9)
    return out


def compare(pred, gt):
    """tolerance from the protocol: same tool, arm and target; position within 2 cm and angle within 10 deg.
    Returns (verdict, reason): verdict in {match, near, differ}; 'near' = equivalent candidate for L2 (e.g. a skipped
    locate, or same tool/arm/object with a parameter outside tolerance)."""
    p, g = normalise(pred), normalise(gt)
    if p["name"] == "invalid":
        return "differ", "invalid"
    if p["name"] != g["name"]:
        if g["name"] == "locate" and p["name"] == "grasp" and p.get("object") == g.get("object"):
            return "near", "skipped locate"
        if g["name"] == "locate" and p["name"] == "locate":
            pass
        return "differ", f"tool {p['name']} vs {g['name']}"
    n = g["name"]
    for k in ("arm", "object", "target", "action"):
        if k in g and p.get(k) != g.get(k):
            return "differ", f"{k} {p.get(k)} vs {g.get(k)}"
    if n == "request_check":
        return "match", ""
    if n == "move":
        err = sum((x - y) ** 2 for x, y in zip(p["d"], g["d"])) ** 0.5
        if err > 2.0:
            return "near", f"move off by {err:.1f} cm"
        def quat(angles):
            x, y, z = [math.radians(v) / 2 for v in angles]
            cx,cy,cz = math.cos(x),math.cos(y),math.cos(z)
            sx,sy,sz = math.sin(x),math.sin(y),math.sin(z)
            return (cx*cy*cz+sx*sy*sz, sx*cy*cz-cx*sy*sz,
                    cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz)
        dot = abs(sum(x*y for x,y in zip(quat(p.get("r", [0,0,0])), quat(g.get("r", [0,0,0])))))
        angle = math.degrees(2*math.acos(min(1.0, dot)))
        return ("match", "") if angle <= 10 else ("near", f"rotation off by {angle:.1f} deg")
    if n == "rotate":
        if p.get("axis") != g.get("axis"):
            return "differ", "axis"
        return ("match", "") if abs(p["deg"] - g["deg"]) <= 10 else ("near", f"angle off by {abs(p['deg'] - g['deg']):.0f} deg")
    if n == "grasp":
        if (p.get("part") or None) != (g.get("part") or None):
            return "near", f"part {p.get('part')} vs {g.get('part')}"
        if p["lift0"] != g["lift0"] or abs(p["lift_cm"]-g["lift_cm"]) > 2:
            return "near", "lift"
        if abs(p["approach_cm"]-g["approach_cm"]) > 1:
            return "near", "approach"
        if p["contact"] != g["contact"]:
            return "near", "contact"
        if p["preshape"] != g["preshape"]:
            return "near", "preshape"
    if n == "place":
        if p["release"] != g["release"]:
            return "differ", "release"
    return "match", ""


# ------------------------------------------------------------------ leak audit (rule part)
TOOL_RX = re.compile(r"\b(locate|grasp|place|press|rotate|home|cmd|open_articulated|request_check|request check|"
                     r"table_center|table_free_spot|handover_point|left_of|right_of|lift_cm|depth_cm|dx|dy|dz)\b(?!\w)|\b(move|gripper)\s*\(",
                     re.I)
TRIPLE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?")


def gt_numbers(gt):
    a = gt.get("arguments") or {}
    out = []
    for k in ("dx", "dy", "dz", "deg", "lift_cm", "depth_cm"):
        if a.get(k) not in (None, 0, "0"):
            out.append(abs(_f(a.get(k))))
    if gt.get("name") == "cmd":
        for c in a.get("commands") or []:
            out += [abs(float(x)) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", str(c))]
    return [x for x in out if x >= 1]


def leak_check(cot, gt, allowed_text="", prev_tool=None):
    """next must not name any tool, give coordinates or the current call's parameters (a number equal to a parameter of
    the current call that is not already in the subgoal / criteria text). assessment describes the PAST step: it may
    name the previous tool, but not the current call's tool (unless it is the same as the previous one), coordinates
    of a target or the current parameters."""
    issues = []
    allowed = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", allowed_text)}
    cur = str(gt.get("name") or "")
    for fld in ("next", "assessment"):
        s = str(cot.get(fld) or "")
        for m in TOOL_RX.finditer(s):
            w = (m.group(1) or m.group(2) or "").lower().replace(" ", "_")
            if fld == "next" or (w == cur and cur != prev_tool) or (w in ("request_check",) and cur == "request_check"
                                                                      and prev_tool != "request_check"):
                issues.append(f"{fld}: tool word '{m.group(0)}'")
                break
        if TRIPLE.search(s) and fld == "next":
            issues.append(f"{fld}: coordinate triple")
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
        for x in nums:
            if any(abs(x - g) < 0.51 for g in gt_numbers(gt)) and x not in allowed:
                if fld == "next":
                    issues.append(f"{fld}: parameter value {x:g}")
                break
    return issues


# ------------------------------------------------------------------ automatic evidence audit against the recorded state
CLAUSE = re.compile(r"[,;.()\[\]{}]| while | and | but | whereas |; ")
ARM_RX = re.compile(r"\b(left|right)\b")


def _objs_in(s, objects):
    s2 = s.lower().replace("_", " ")
    return [o for o in objects if o.replace("_", " ") in s2 or o in s.lower()]


def auto_audit(cot, S, objects):
    """contradictions between seen/reported items and the recorded simulator state of the observation (clause level:
    a clause is checked only when it names exactly one arm / one object)."""
    bad = []
    items = [("seen", x) for x in (cot.get("seen") or [])] + [("reported", x) for x in (cot.get("reported") or [])]
    for fld, s in items:
        s = str(s)
        if re.search(r"\b(will|would|expected|should|after the next)\b", s.lower()):
            continue
        for cl in CLAUSE.split(s):
            low = " " + cl.lower().strip() + " "
            arms = set(ARM_RX.findall(low))
            obs = _objs_in(cl, objects)
            neg = re.search(r"\b(not|no longer|nothing|never|without)\b", low)
            if len(arms) == 1:
                arm = arms.pop()
                g = S["arms"][arm]["gripper_cmd"]
                held = S["held"].get(arm)
                if re.search(r"\bgripper\b.*\b(is |stays |remains |now |still )?(open|opened)\b", low) and not neg and "half" not in low and g < 0.2:
                    bad.append(f"{fld}: '{s}' (open) but {arm} gripper_cmd={g}")
                if re.search(r"\bgripper\b.*\b(is |stays |remains |now |still )?(closed|shut)\b", low) and not neg and "half" not in low and g > 0.8:
                    bad.append(f"{fld}: '{s}' (closed) but {arm} gripper_cmd={g}")
                if re.search(r"\b(empty|holds nothing|holding nothing)\b", low) and held and "not empty" not in low:
                    bad.append(f"{fld}: '{s}' (empty) but {arm} holds {held}")
                if re.search(r"\b(holds|holding|grips|gripping|grasping|has grasped)\b", low) and not neg and len(obs) == 1 \
                        and held != obs[0]:
                    bad.append(f"{fld}: '{s}' ({arm} holds {obs[0]}) but {arm} holds {held}")
            elif len(obs) == 1 and not arms and not neg:
                o = obs[0]
                oz = S["objects"][o]["lo"][2]
                if re.search(r"\b(in the air|lifted|raised|off the table)\b", low) and oz < 0.8:
                    bad.append(f"{fld}: '{s}' ({o} in the air) but bottom z={oz}")
                if re.search(r"\b(rests|resting|lies|lying|sits|sitting|stands|standing) on the table\b", low) and oz > 3 \
                        and o in S["held"].values():
                    bad.append(f"{fld}: '{s}' ({o} on the table) but it is held, bottom z={oz}")
    return bad


_legacy_auto_audit = auto_audit

def auto_audit(cot, S, objects):
    from fact_rules import audit as strict_audit
    return list(dict.fromkeys(_legacy_auto_audit(cot, S, objects) + strict_audit(cot, S, objects)))


_base_leak_check = leak_check

def leak_check(cot, gt, allowed_text="", prev_tool=None):
    issues = _base_leak_check(cot, gt, allowed_text, prev_tool)
    text = str(cot.get("next") or "")
    if re.search(r"\b(?:grasps|grasping|grasped|places|placing|placed|rotates|rotating|rotated|presses|pressing|pressed|locates|locating|located)\b", text, re.I):
        issues.append("next: inflected tool word")
    if re.search(r"\b(?:rx|ry|rz|contact|approach_cm|preshape|lift_cm)\s*[:=]", text, re.I):
        issues.append("next: action parameter")
    if re.match(r"\s*(?:open|close|move|turn|lift|pick|set|push|return)\b", text, re.I):
        issues.append("next: imperative instead of observable condition")
    return list(dict.fromkeys(issues))
