#!/usr/bin/env python
"""EB-Manipulation (EmbodiedBench) zero-shot driver for the TinyAstral harness H*: the harness tool layer (Guava single-arm
profile without articulate, plus a straight-line move tool) drives the official EBManEnv. Perception uses only the front
camera RGB + the camera point cloud + SAM3; actions go through the official 7-d discrete action interface one step at a
time (adapter_eb.py); success = the official per-step task_success (logged only, never returned to the model).

Success measures written to result.json:
  success_any = task_success occurred within this episode's env-step budget (--max-env-steps, default 60). This is the
                official success flag latched over the episode and is the paper's official measure for EB-Manipulation;
  success_15  = task_success occurred within the first 15 env steps (the official 15-step horizon).
SYSTEM = HM.system_text("guava", <Guava prompt without articulate + move doc>, flags).

Import order matters: the official EBManEnv is imported first (it binds EB's own tools.grasploc), then 'tools' is dropped
from sys.modules and harness/ goes first on sys.path; the single-arm planner is imported only after that.
The working directory is switched to <out-root>/<tag>/<eval-set> because EB uses relative paths (TTMS_FOLDER, running/).

Usage (set the EB environment first, see docs/benchmarks/embodiedbench.md):
  python run_eb.py --eval-set base --indexes 1,13 [--url $TINYASTRAL_URL] --sam3 http://HOST:8830 --tag mymodel [--gen ...]
  eval sets: base, common_sense, complex, spatial (48 episodes each), visual (36): 228 in total."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))           # zeroshot/ (zs_common), last: its subdirs must not shadow packages
import zs_common as ZC  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EB = ZC.third_party("EMBODIEDBENCH_REPO", "EmbodiedBench")
BENCH = "guava"
OUT = ZC.out_root() / "zeroshot" / "embodiedbench"

MOVE_DOC = ("- move(direction: str, distance_cm: float) -> move the gripper (and whatever it holds) in a straight line; "
            "direction in left/right/front/back/up/down. Returns how far the gripper actually went.\n")


def eb_profile():
    import adapter_eb as AE
    return {"arms": ("right",), "units": "m", "retry": 1, "regrasp": (90.0,), "backoff": 0.06, "retreat": (0.06,),
            "closed_w": AE.CLOSED_W, "open_w": AE.OPEN_W, "safe_z": AE.SAFE_Z, "clear_view": False,
            "same_object_guard": False, "default_relation": "near", "pixel_lr": True,
            "tools": {"locate", "grasp", "lift", "move", "place", "carry", "push", "release", "home", "verify", "done",
                      "give_up"}}


def system_prompt():
    import guava_protocol as HG                    # the single-arm embodiment prompt used for Guava (no task content)
    lines = [ln for ln in HG.SYSTEM.split("\n") if not ln.startswith("- articulate(")]   # no articulated parts here
    i = next(k for k, ln in enumerate(lines) if ln.startswith("- release("))
    lines.insert(i, MOVE_DOC.rstrip("\n"))
    return "\n".join(lines)


def _yolo_weights():
    """EB's eb_man_utils loads YOLO("yolo11n.pt") from the working directory at import time (used only by EB's own
    native baseline, not by this driver). Link it into the run dir if the EB clone has it; otherwise Ultralytics fetches it."""
    for p in (EB / "_assets" / "yolo11n.pt", EB / "yolo11n.pt"):
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="base")
    ap.add_argument("--indexes", default="")
    ap.add_argument("--url", default=ZC.planner_url())
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--sam3", default=os.environ.get("EB_SAM3_URL"), required=not os.environ.get("EB_SAM3_URL"),
                    help="SAM3 server with the Guava API (POST /segment); default $EB_SAM3_URL")
    ap.add_argument("--tag", default="harness")
    ap.add_argument("--out-root", default=str(OUT))
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-calls", type=int, default=25)
    ap.add_argument("--max-env-steps", type=int, default=60)
    ap.add_argument("--xml-calls", type=int, default=1)
    ZC.add_gen_arg(ap)
    a = ap.parse_args()

    run = Path(a.out_root).resolve() / a.tag / a.eval_set
    run.mkdir(parents=True, exist_ok=True)
    for name, src in (("embodiedbench", EB / "embodiedbench"), ("yolo11n.pt", _yolo_weights())):
        if src is None:
            continue
        try:
            os.symlink(src, run / name)
        except FileExistsError:
            pass
    os.chdir(run)                                    # official relative paths (TTMS_FOLDER, running/)
    sys.path.insert(0, str(EB))
    # name clash: EB's amsolver does `from tools.grasploc import ...` (EB's tools package) and harness/ has tools.py.
    # Import the official env first (binds tools.grasploc), then drop 'tools' and put harness/ first on sys.path.
    from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
    sys.modules.pop("tools", None)
    HM = ZC.load_harness(BENCH, front=True)
    flags, gen_spec = ZC.gen_flags(HM, BENCH, a.gen)
    import planner as HU
    HU.XML_CALLS = bool(a.xml_calls)
    from adapter_eb import EBAdapter
    from PIL import Image

    idx = [int(x) for x in a.indexes.split(",") if x.strip()]
    env = EBManEnv(eval_set=a.eval_set, img_size=(500, 500), log_path=str(run / "_envlog"), selected_indexes=idx)
    system = HM.system_text(BENCH, system_prompt(), flags)

    class EBDriver(HU.Driver):
        keep_turns, attempts = 3, 3

        def __init__(self, A, instruction, folder):
            self.A, self.instruction, self.folder = A, instruction, folder
            (folder / "img").mkdir(parents=True, exist_ok=True)
            self.system = system
            self.cap, self.wall = a.max_calls, 1e9

        def sleep(self, attempt):
            return 2 + 10 * attempt

        def observe(self, dec, H):
            self.A.refresh()
            ip = self.folder / "img" / f"{dec:03d}.jpg"
            HU._jpg(self.A.rgb(), ip, 90)
            return [{"type": "image_url", "image_url": {"url": str(ip)}}], ip.name

        def encode(self, c):
            import base64
            return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(c["image_url"]["url"]).read_bytes()).decode()}}

        def first_text(self):
            return f"Task: {self.instruction}\n"

        def budget_text(self, el, S):
            return f"\nCalls used: {S['n_actions']}/{self.cap}. Arm motions used: {self.A.env_steps}/{self.A.max_env_steps}."

        def before_call(self):
            self.A.trace = []

        def trace(self):
            return list(self.A.trace)

        def success(self):
            return self.A.success()

    for k in range(env.number_of_episodes):
        ep_idx = idx[k] if idx else k
        d = run / f"ep{ep_idx:03d}"
        instr, obs = env.reset()
        if (d / "result.json").exists():
            continue
        try:
            A = EBAdapter(env, a.sam3, max_env_steps=a.max_env_steps)
            A.start_episode(obs)
            D = EBDriver(A, instr, d)
            H = HM.make_harness(BENCH, A, eb_profile(), flags)
            S = HM.run_loop(D, H, a.url, a.model, a.thinking, a.max_tokens)
            ZC.check_sent(S, system)
            Image.fromarray(A.rgb()).save(d / "img" / "final.jpg", quality=90)
            res = {"eval_set": a.eval_set, "index": ep_idx, "variation": env.current_task_variation, "instruction": instr,
                   "success_15": A.success_step is not None and A.success_step <= 15,
                   "success_any": A.success_step is not None, "success_step": A.success_step, "env_steps": A.env_steps,
                   "invalid_steps": A.invalid, "end_reason": S["end"], "calls": S["n_actions"], "rejects": S["rejects"],
                   "seconds": S["seconds"], "llm_s": round(S["llm_s"], 1), "table_z": A.table, "model": a.model,
                   "url": a.url, "thinking": a.thinking, "objects": list(A.names), "harness": "unified+adapter_eb",
                   "infra_error": S.get("infra_error")}
            steps = [{x: v for x, v in s.items() if x != "obs_text"} for s in S["steps"]]
            (d / "episode.json").write_text(json.dumps({**res, "system": system, "steps": steps, "usage": S["usage"]}, default=HU.jdefault))
            (d / "result.json").write_text(json.dumps(res, default=str))
            ZC.tag(d, HM, BENCH, flags, gen_spec, system, __file__)
            print(json.dumps({x: res[x] for x in ("index", "variation", "success_15", "success_any", "success_step", "env_steps", "end_reason", "calls", "seconds")}), flush=True)
        except Exception as e:  # noqa: BLE001
            d.mkdir(parents=True, exist_ok=True)
            (d / "error.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}))
            print("EPISODE_ERROR", ep_idx, type(e).__name__, str(e)[:300], flush=True)
    env.close()
    print("HARNESS_DONE", a.tag, a.eval_set, flush=True)


if __name__ == "__main__":
    main()
