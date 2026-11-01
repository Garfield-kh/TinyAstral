"""Toy benchmark + mock teacher, so the loop runs end-to-end on a CPU in seconds.

Ten pick-and-place tasks; four involve tall objects.  Outcomes are a deterministic function of (task, seed) plus a small
version-specific noise term, which mimics serving non-determinism and produces discordant pairs even between identical
harnesses.  The mock teacher stands in for the frozen teacher: it grasps, places, then declares done (also after a miss,
which is a false completion the success check catches)."""
import hashlib
import importlib.util
import json
import pathlib
import random
import re

OBJ = {"bottle": 0.24, "can": 0.12, "cup": 0.09, "vase": 0.30, "block": 0.05, "mug": 0.10, "thermos": 0.27, "apple": 0.08,
       "lamp": 0.35, "cube": 0.06}
TARGET = {"bottle": "bin", "can": "plate", "cup": "tray", "vase": "shelf", "block": "box", "mug": "tray", "thermos": "bag",
          "apple": "bowl", "lamp": "desk", "cube": "pad"}
TASKS = [f"put_{o}_on_{t}" for o, t in TARGET.items()]
NOISE = 0.08


class Env:
    def __init__(self, task, seed, version):
        self.obj = task.split("_")[1]
        self.target = TARGET[self.obj]
        r = random.Random(f"{task}|{seed}")
        self.u_grasp, self.u_place = r.random(), r.random()
        self.noise = random.Random(f"{task}|{seed}|{version}")
        self.holding, self.placed, self.lifted, self.last = False, False, False, "none"

    def _coin(self, p, u):
        return self.noise.random() < 0.5 if self.noise.random() < NOISE else u < p

    def observe(self):
        return (f"task: put the {self.obj} on the {self.target}. holding: {self.obj if self.holding else 'nothing'}. "
                f"placed: {self.placed}. last: {self.last}")

    def measure(self, obj):
        return {"height_m": OBJ[obj]}

    def grasp(self, obj, approach="top"):
        tall = OBJ[obj] > 0.18
        p = {("top", False): 0.80, ("top", True): 0.15, ("side", True): 0.80, ("side", False): 0.50}[(approach, tall)]
        self.holding = obj == self.obj and self._coin(p, self.u_grasp)
        return self.holding

    def lift(self, obj, height):
        self.lifted = height > 0.2

    def place(self, obj, target):
        if not self.holding:
            return False
        self.placed = target == self.target and self._coin(0.70 if self.lifted else 0.95, self.u_place)
        self.holding = False
        return self.placed


class MockTeacher:
    def chat(self, messages):
        obs = messages[-1]["content"]
        obj, tgt = re.search(r"put the (\w+) on the (\w+)", obs).groups()
        if "placed: True" in obs or "MISSED" in obs or "FAILED" in obs:
            return json.dumps({"tool": "done"})
        if "holding: nothing" in obs:
            return json.dumps({"tool": "grasp", "object": obj})
        return json.dumps({"tool": "place", "object": obj, "target": tgt})


_CACHE = {}


def _harness(d):
    if d not in _CACHE:
        src = (pathlib.Path(d) / "harness.py").read_bytes()
        spec = importlib.util.spec_from_file_location(f"toy_harness_{hashlib.sha1(d.encode()).hexdigest()[:8]}",
                                                      pathlib.Path(d) / "harness.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE[d] = (mod, hashlib.sha256(src).hexdigest()[:12])
    return _CACHE[d]


def run_episode(harness_dir, task, seed, out_dir, teacher, max_calls=None):
    mod, version = _harness(harness_dir)
    if teacher.get("model") != "mock":
        from rsi.teacher import Teacher
        tch = Teacher(teacher)
    else:
        tch = MockTeacher()
    env, log = Env(task, seed, version), []
    mod.run(env, tch, log, max_calls or 6)
    trace = pathlib.Path(out_dir) / "trace.txt"
    trace.write_text("\n".join(log) + "\n")
    return {"success": env.placed, "rejected": False, "trace": str(trace)}
