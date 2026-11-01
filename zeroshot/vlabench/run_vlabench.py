#!/usr/bin/env python
"""VLABench zero-shot driver for the TinyAstral harness H* (tool layer + planner loop unchanged, LIBERO profile:
single Panda, same tool manifest, same retry / guard settings).

Protocol:
  * episodes: the official standard evaluation episodes VLABench/configs/evaluation/tracks/<track>.json[task][k]
    (fixed layouts / instructions), built exactly like the official evaluator; the paper uses the first 10 episodes
    of every task in tracks 1, 2, 3, 4, 6 (500 episodes);
  * planner input: the official instruction string + the task camera image (annotated with the names the planner has
    located) + gripper state + previous tool result; nothing else from the episode config;
  * SYSTEM = HM.system_text("libero", <LIBERO prompt with the VLABench frame sentence>, flags);
  * budget: 40 tool calls, no wall clock (the official evaluator has no wall clock either; chat timeout 600 s per call);
  * success: the official termination condition, checked after every simulator step (official evaluator semantics:
    the episode ends as soon as it fires) = the paper's official measure. It is never shown to the planner; the loop
    simply stops. progress_score / intention_score = the official post-episode metrics.

Usage:
  run_vlabench.py --track track_3_common_sense --task select_fruit --episodes 0 1 2 [--out DIR] [--url $TINYASTRAL_URL]
                  [--model qwen] [--thinking 1] [--xml-calls 1] [--sam3 http://127.0.0.1:8164] [--gen default|none|all|f1,f2]
  run_vlabench.py --probe --track ... --task ... --episodes 0 --out DIR          (reset + render only, no model)"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))           # zeroshot/ (zs_common), last: its subdirs must not shadow packages
import zs_common as ZC  # noqa: E402

BENCH = "libero"
HM = ZC.load_harness(BENCH)

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):   # keep many parallel sims from oversubscribing the CPU
    os.environ[_v] = os.environ.get("ZS_THREADS", "2")
import argparse  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ZC.HARNESS))

MAX_CALLS = 40

LIBERO_FRAME = ("Coordinates: metres, robot frame: +x points toward the camera, +y points to the image right, "
                "z = height above the table.")
VLAB_FRAME = ("Coordinates: metres, robot base frame: +x points forward from the robot base, +y to the robot's left, "
              "z = height above the table / counter surface in front of the robot.")


def system_prompt():
    import libero_protocol as HL                   # the LIBERO single-arm prompt (embodiment + tools, no tasks)
    s = HL.SYSTEM
    assert LIBERO_FRAME in s, "LIBERO prompt changed: update the frame sentence"
    return s.replace(LIBERO_FRAME, VLAB_FRAME)


def harness_system(flags):
    return HM.system_text(BENCH, system_prompt(), flags)


class VLABenchDriver:
    keep_turns, attempts, max_reject = 3, 3, 2
    wall_visible, default_end, cap_end = False, "call_limit", "call_limit"

    def __init__(self, env, A, folder, system):
        self.env, self.A, self.folder = env, A, folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "img").mkdir(exist_ok=True)
        self.system = system
        self.cap, self.wall = MAX_CALLS, 1e9
        self.instruction = env.instruction
        self.ended_by_success = False

    def record(self, body, k, attempt):
        pass

    def before_call(self):
        pass

    def after_error(self):
        pass

    def trace(self):
        return None

    def sleep(self, attempt):
        return 2

    def observe(self, dec, H):
        from planner import _jpg
        if dec == 0:
            self.A.deadline = self.env.deadline = self.t0 + self.wall
        ip = self.folder / "img" / f"{dec:03d}.jpg"
        _jpg(self.A.annotated_rgb(), ip, 90)
        return [{"type": "image_url", "image_url": {"url": str(ip)}}], ip.name

    def closeup(self, dec, im):
        from planner import _jpg
        cp = self.folder / "img" / f"{dec:03d}_closeup.jpg"
        _jpg(im, cp, 92)
        return [{"type": "image_url", "image_url": {"url": str(cp)}}]

    def encode(self, c):
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(c["image_url"]["url"]).read_bytes()).decode()}}

    def first_text(self):
        return f"Task: {self.instruction}\n"

    def budget_text(self, el, S):
        return f"\nCalls used: {S['n_actions']}/{self.cap}."

    def success(self):
        """official termination flag (logging + protocol stop only; the planner never sees it)."""
        s = self.A.success()
        if s and not self.ended_by_success:
            self.ended_by_success = True
            self.cap = 0                   # the official episode ends when the success condition fires
        return s


def official_max_episode_length(task):
    """VLABench/evaluation/evaluator/base.py: 200 unless configs/task_config.json[<config key>].evaluation.max_episode_length;
    one policy step = env.step x max_substeps (=1 in scripts/evaluate_policy.py), i.e. env.step calls."""
    from VLABench.configs import name2config
    from VLABench.utils.utils import find_key_by_value
    tc = json.load(open(os.path.join(os.environ["VLABENCH_ROOT"], "configs/task_config.json")))
    k = find_key_by_value(name2config, task)
    return int((tc.get(k) or {}).get("evaluation", {}).get("max_episode_length") or 200)


def probe(a):
    from PIL import Image
    from env_vlabench import VLABenchEnv
    from adapter_vlabench import VLABenchAdapter
    for k in a.episodes:
        t0 = time.time()
        env = VLABenchEnv(a.track, a.task, k)
        t1 = time.time()
        A = VLABenchAdapter(env)
        A.start_episode()
        d = a.out / f"probe_{a.track}_{a.task}_e{k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for v in A._views:
            Image.fromarray(v["rgb"]).save(d / f"{v['name']}.jpg", quality=90)
        h = A.hand_pose()
        t2 = time.time()
        A.move([h[0], h[1], h[2] + 0.05])
        t3 = time.time()
        A.run({"right": [{"type": "grip", "val": 0.0}]})
        w_closed = A.width()
        A.run({"right": [{"type": "grip", "val": 1.0}]})
        info = {"track": a.track, "task": a.task, "episode": k, "instruction": env.instruction, "cams": env.cam_names,
                "views": [v["name"] for v in A._views], "table_z_world_offset": A.tz, "hand": [round(float(x), 3) for x in h],
                "hand_after_up5cm": [round(float(x), 3) for x in A.hand_pose()[:3]], "width_open": round(A.width(), 4),
                "width_closed_empty": round(w_closed, 4), "load_s": round(t1 - t0, 1), "start_s": round(t2 - t1, 1),
                "move5cm_s": round(t3 - t2, 1), "steps": env.n_steps, "success_flag": env.success_flag}
        (d / "probe.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info), flush=True)
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ZC.out_root() / "zeroshot" / "vlabench")
    ap.add_argument("--url", default=ZC.planner_url())
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=4096)
    ap.add_argument("--xml-calls", dest="xml_calls", type=int, default=1)
    ap.add_argument("--sam3", default=None, help="SAM3 server (perception/sam3_srv.py, POST /seg); default $LIBERO_SAM3_URL")
    ap.add_argument("--probe", action="store_true")
    ZC.add_gen_arg(ap)
    a = ap.parse_args()
    flags, gen_spec = ZC.gen_flags(HM, BENCH, a.gen)
    if a.sam3:
        os.environ["LIBERO_SAM3_URL"] = a.sam3
    import adapter_libero as AL
    if a.sam3:
        AL.SAM3 = a.sam3
    if a.probe:
        return probe(a)
    import planner as HU
    HU.XML_CALLS = bool(a.xml_calls)
    from env_vlabench import VLABenchEnv
    from adapter_vlabench import VLABenchAdapter
    system = harness_system(flags)
    for k in a.episodes:
        d = a.out / a.track / a.task / f"e{k:02d}"
        if (d / "result.json").exists():
            continue
        env = None
        try:
            env = VLABenchEnv(a.track, a.task, k)
            A = VLABenchAdapter(env)
            A.cur_approach_vec = None
            A.start_episode()
            D = VLABenchDriver(env, A, d, system)
            H = HM.make_harness(BENCH, A, HU.libero_profile(), flags)
            S = HM.run_loop(D, H, a.url, a.model, a.thinking, a.max_tokens)
            ZC.check_sent(S, D.system)
            if D.ended_by_success:
                S["end"] = "official_success"
            from PIL import Image
            Image.fromarray(A.rgb()).save(d / "img" / "final.jpg", quality=90)
            off = env.official_metrics()
            mel = official_max_episode_length(a.task)
            off["success_final"] = bool(off["success"])            # official termination fired before the agent ended
            off["success_step"] = env.success_step
            off["official_max_episode_length"] = mel
            off["success_strict"] = bool(off["success"] and env.success_step is not None and env.success_step <= mel)
            res = {"bench": "vlabench", "track": a.track, "task": a.task, "episode": k, "instruction": env.instruction,
                   **off, "end_reason": S["end"], "calls": S["n_actions"], "rejects": S["rejects"], "seconds": S["seconds"],
                   "llm_s": round(S["llm_s"], 1), "sim_steps": env.n_steps,
                   "perception": {x: (round(v, 1) if isinstance(v, float) else v) for x, v in A.stats.items()},
                   "model": a.model, "url": a.url, "thinking": a.thinking, "xml_calls": a.xml_calls,
                   "objects": list(A.names), "infra_error": S.get("infra_error"), "harness": "unified/libero_profile",
                   "budget": {"calls": MAX_CALLS, "wall_s": None}}
            (d / "episode.json").write_text(json.dumps({**res, "system": D.system, "steps": S["steps"], "usage": S["usage"]},
                                                       default=HU.jdefault))
            (d / "result.json").write_text(json.dumps(res, default=str))
            ZC.tag(d, HM, BENCH, flags, gen_spec, D.system, __file__)
            print(json.dumps({x: res[x] for x in ("task", "episode", "success", "progress_score", "end_reason", "calls", "seconds")}), flush=True)
        except Exception as e:  # noqa: BLE001
            d.mkdir(parents=True, exist_ok=True)
            (d / "error.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}))
            print("EPISODE_ERROR", a.task, k, type(e).__name__, str(e)[:200], flush=True)
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
