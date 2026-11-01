#!/usr/bin/env python
"""SimplerEnv zero-shot driver for the TinyAstral harness H* (planner loop + tool layer unchanged; only the adapter and
this protocol driver are SimplerEnv specific).

Protocol:
  * episodes = official visual-matching configurations (episodes_simplerenv.py, transcribed from the official scripts);
  * observation = the policy camera RGB of the official env (overlay included), as given to RT-1/Octo/OpenVLA;
    the instruction = env.get_language_instruction();
  * planner prompt + tool set = the single-arm Guava prompt/tools of the harness (prompts.GUAVA_SYSTEM, through
    HM.system_text("guava", ...)), unchanged;
  * every robot motion goes through env.step with the official controller; the official evaluate() runs every step;
  * success_final = official success flag when the agent ends (the paper's official measure for SimplerEnv);
    success_strict = the official flag at the step the official evaluator would have stopped
    (min(agent end, official max_episode_steps)). Neither is shown to the planner. Multi-subtask env (apple into
    drawer): when the agent calls done on a non-final subtask the runner advances the env subtask (as the official
    evaluator does on predicted termination) and the next instruction starts a fresh planner context.

Usage: run_simplerenv.py --key coke_h --idx 12 [--url $TINYASTRAL_URL] [--model qwen] [--out DIR] [--thinking 1]
                         [--sam3 URL] [--gen default|none|all|f1,f2]
Episode keys: coke_h coke_v coke_s move_near drawer_open drawer_close apple_drawer br_spoon br_carrot br_stack br_eggplant
(see episodes_simplerenv.py; run_all.py runs the full 960-episode set)."""
from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))           # zeroshot/ (zs_common), last: its subdirs must not shadow packages
import zs_common as ZC  # noqa: E402

BENCH = "guava"
HM = ZC.load_harness(BENCH)

import episodes_simplerenv as ES  # noqa: E402
from adapter_simplerenv import SimplerAdapter  # noqa: E402
import adapter_simplerenv as AS  # noqa: E402
import planner as HU  # noqa: E402  (= HM: harness/single_arm/planner.py)

CTRL = Path(HU.__file__).resolve().parent


def guava_system():
    """GUAVA_SYSTEM of harness/single_arm/prompts.py verbatim (read without importing the Guava runtime)."""
    tree = ast.parse((CTRL / "prompts.py").read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "GUAVA_SYSTEM" for t in n.targets):
            return ast.literal_eval(n.value)
    raise RuntimeError("GUAVA_SYSTEM not found")


def simpler_profile():
    return {"arms": ("right",), "units": "m", "retry": 2, "regrasp": (90.0,), "backoff": 0.08, "retreat": (0.08,),
            "closed_w": AS.CLOSED_W, "open_w": AS.OPEN_W, "safe_z": AS.SAFE_Z, "clear_view": False,
            "same_object_guard": False, "default_relation": "near", "pixel_lr": True,
            "tools": {"locate", "grasp", "lift", "place", "carry", "push", "articulate", "release", "home", "verify", "done", "give_up"}}


class SimplerDriver(HU.Driver):
    keep_turns, attempts = 3, 3

    def __init__(self, A, instruction, folder, cap, system):
        self.A, self.instruction, self.folder = A, instruction, folder
        (folder / "img").mkdir(parents=True, exist_ok=True)
        self.system, self.cap, self.wall = system, cap, 1e9
        self.tag = ""

    def sleep(self, attempt):
        return 2 + 10 * attempt

    def observe(self, dec, H):
        self.A.refresh()
        ip = self.folder / "img" / f"{self.tag}{dec:03d}.jpg"
        HU._jpg(self.A.rgb(), ip, 90)
        return [{"type": "image_url", "image_url": {"url": str(ip)}}], ip.name

    def encode(self, c):
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(c["image_url"]["url"]).read_bytes()).decode()}}

    def first_text(self):
        return f"Task: {self.instruction}\n"

    def budget_text(self, el, S):
        return f"\nCalls used: {S['n_actions']}/{self.cap}."

    def success(self):
        return self.A.success()

    def trace(self):
        return {"env_steps": self.A.n_steps}


def harness_system(flags):
    return HM.system_text(BENCH, guava_system(), flags)


def run_episode(ep, a, system, flags):
    """one SimplerEnv episode; system = the full SYSTEM to send (harness_system(flags))."""
    if system != harness_system(flags):
        raise RuntimeError("run_simplerenv: system does not match system_text(guava, base, flags)")
    d = Path(a.out) / f"{ep['key']}__{ep['idx']:04d}"
    if (d / "result.json").exists():
        return json.loads((d / "result.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env, cam = ES.build(ep)
    ES.reset(env, ep)
    U = env.unwrapped
    A = SimplerAdapter(env, cam, a.sam3)
    A.start_episode()
    parts, end, steps_all = [], None, []
    while True:
        instr = U.get_language_instruction()
        D = SimplerDriver(A, instr, d, a.cap, system)
        D.tag = f"p{len(parts)}_"
        H = HM.make_harness(BENCH, A, simpler_profile(), flags)
        try:
            S = HM.run_loop(D, H, a.url, a.model, a.thinking, a.max_tokens)
        except RuntimeError as e:                       # adapter step cap (runaway safety, far above the official horizon)
            S = {"end": f"error:{e}", "steps": [], "n_actions": 0, "usage": [], "llm_s": 0.0, "seconds": round(time.time() - t0, 1)}
        ZC.check_sent(S, system)                        # no-op when the loop raised (no messages)
        end = S["end"]
        parts.append({"instruction": instr, "end": end, "calls": S["n_actions"], "env_steps_after": A.n_steps})
        steps_all += [{**{k: v for k, v in s.items() if k != "obs_text"}, "part": len(parts) - 1} for s in S["steps"]]
        if end in ("model_done", "done_after_rejections") and not U.is_final_subtask():
            U.advance_to_next_subtask()
            continue
        break
    tr = A.info_trace
    n = A.n_steps
    strict_i = min(n, ep["max_steps"]) - 1
    success_strict = bool(tr[strict_i]["success"]) if n > 0 else False
    res = {"key": ep["key"], "idx": ep["idx"], "env_name": ep["env_name"], "kw": {k: str(v) for k, v in ep["kw"].items()},
           "instruction": parts[0]["instruction"], "success_strict": success_strict, "success_final": A.success(),
           "success_ever": any(x["success"] for x in tr), "env_steps": n, "official_max_steps": ep["max_steps"],
           "within_official_steps": n <= ep["max_steps"], "end_reason": end, "parts": parts,
           "calls": sum(p["calls"] for p in parts), "seconds": round(time.time() - t0, 1),
           "seg_calls": A.stats["seg_calls"], "seg_s": round(A.stats["seg_s"], 1), "table_z_base": round(A.tz, 3),
           "official_info_final": A.official_info(), "model": a.model, "url": a.url, "thinking": a.thinking, "harness": "unified"}
    HU._jpg(A.rgb(), d / "img" / "final.jpg", 90)
    (d / "episode.json").write_text(json.dumps({**res, "system": system, "steps": steps_all, "info_trace": tr}, default=HU.jdefault))
    (d / "result.json").write_text(json.dumps(res, default=str))
    ZC.tag(d, HM, BENCH, flags, a.gen_spec, system, __file__)
    env.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--url", default=ZC.planner_url())
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=4096)
    ap.add_argument("--sam3", default=os.environ.get("SIMPLER_SAM3_URL", "http://127.0.0.1:8160"),
                    help="SAM3 server (perception/sam3_srv.py, POST /seg); default $SIMPLER_SAM3_URL")
    ap.add_argument("--cap", type=int, default=25, help="tool-call limit per episode (paper: 25)")
    ap.add_argument("--out", default=str(ZC.out_root() / "zeroshot" / "simplerenv"))
    ZC.add_gen_arg(ap)
    a = ap.parse_args()
    flags, a.gen_spec = ZC.gen_flags(HM, BENCH, a.gen)
    system = harness_system(flags)
    eps = ES.manifest(a.key)
    for i in a.idx:
        try:
            r = run_episode(eps[i], a, system, flags)
            print(json.dumps({k: r[k] for k in ("key", "idx", "success_strict", "success_final", "env_steps", "calls", "end_reason", "seconds")}), flush=True)
        except Exception as e:  # noqa: BLE001
            d = Path(a.out) / f"{a.key}__{i:04d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "error.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}))
            print("EPISODE_ERROR", a.key, i, type(e).__name__, str(e)[:200], flush=True)


if __name__ == "__main__":
    main()
