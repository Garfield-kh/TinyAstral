"""python -m unittest discover -s tests   (from the repository root)"""
import json
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rsi import gate, patch, seeds  # noqa: E402

G = dict(bootstrap_reps=1000, bootstrap_seed=7, alpha=0.025, veto_untriggered_pp=-2.0, interim_fraction=0.5,
         interim_planned_gain_pp=3.0, interim_min_power=0.10, screen_min_upper_pp=2.0)


def arms(effect_target, effect_other, fire_other=False, n_tasks=10, n_seeds=40, seed=0):
    """Synthetic paired outcomes: tasks t0..t3 are the target cluster."""
    R = random.Random(seed)
    p, c, tr = {}, {}, {}
    for i in range(n_tasks):
        for s in range(n_seeds):
            base = R.random() < 0.5
            eff = effect_target if i < 4 else effect_other
            cs = base if R.random() > 0.1 else R.random() < 0.5          # symmetric noise -> discordant pairs
            if eff > 0 and not base and R.random() < eff:
                cs = True
            if eff < 0 and base and R.random() < -eff:
                cs = False
            k = (f"t{i}", s)
            p[k], c[k] = dict(success=base, rejected=False), dict(success=cs, rejected=False)
            if i < 4 or fire_other:
                tr[("cand",) + k] = "[new_tool]"
    return p, c, tr


class TestGate(unittest.TestCase):
    targets = {"t0", "t1", "t2", "t3"}

    def test_accepts_real_gain(self):
        p, c, tr = arms(0.4, 0.0)
        self.assertEqual(gate.decide(p, c, self.targets, ["[new_tool]"], tr, G)["decision"], "ACCEPT")

    def test_rejects_null(self):              # false-accept rate under no effect should be about alpha (2.5%)
        g = dict(G, bootstrap_reps=400)
        acc = sum(gate.decide(*arms(0.0, 0.0, seed=sd)[:2], self.targets, ["[new_tool]"], arms(0.0, 0.0, seed=sd)[2], g)
                  ["decision"] == "ACCEPT" for sd in range(20))
        self.assertLessEqual(acc, 2)

    def test_vetoes_untriggered_regression(self):
        p, c, tr = arms(0.4, -0.3)              # target gains, but the untriggered pairs regress: the gating leaks
        out = gate.decide(p, c, self.targets, ["[new_tool]"], tr, G)
        self.assertTrue(out["veto"])
        self.assertEqual(out["decision"], "REJECT")

    def test_marker_in_parent_rejects(self):
        p, c, tr = arms(0.4, 0.0)
        tr[("parent", "t0", 0)] = "[new_tool]"
        self.assertIn("marker in parent arm", gate.decide(p, c, self.targets, ["[new_tool]"], tr, G)["reasons"])

    def test_interim_stops_harmful(self):
        p, c, tr = arms(-0.3, -0.3, fire_other=True)
        self.assertEqual(gate.decide(p, c, self.targets, ["[new_tool]"], tr, G, "interim")["decision"], "STOP")

    def test_screen(self):
        self.assertEqual(gate.decide(*arms(0.4, 0.0, n_seeds=8)[:2], self.targets, ["[new_tool]"],
                                     arms(0.4, 0.0, n_seeds=8)[2], G, "screen")["decision"], "CONTINUE")
        self.assertEqual(gate.decide(*arms(-0.4, 0.0, n_seeds=8)[:2], self.targets, ["[new_tool]"],
                                     arms(-0.4, 0.0, n_seeds=8)[2], G, "screen")["decision"], "DROP")


class TestSeeds(unittest.TestCase):
    cfg = dict(tasks=["a", "b"], seeds=dict(dev=[0, 10], selection=[10, 100], eval=[100, 200]))

    def test_eval_refused(self):
        with self.assertRaises(PermissionError):
            seeds.pairs(self.cfg, "eval", 0, 2)

    def test_overlap_refused(self):
        with self.assertRaises(ValueError):
            seeds.check_disjoint(dict(dev=[0, 20], selection=[10, 100], eval=[100, 200]))

    def test_blocks_fresh(self):
        a, b = seeds.pairs(self.cfg, "selection", 0, 3), seeds.pairs(self.cfg, "selection", 3, 3)
        self.assertFalse({s for _, s in a} & {s for _, s in b})


class TestPatch(unittest.TestCase):
    def test_apply_and_lint(self):
        import difflib
        d = pathlib.Path(tempfile.mkdtemp())
        (d / "p").mkdir()
        old = "def f(x):\n    return x\n"
        (d / "p" / "h.py").write_text(old)
        new = "def f(x):\n    log('[m1]')\n    return x\n"
        diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "a/h.py", "b/h.py"))
        patch.apply(d / "p", d / "c", diff)
        self.assertEqual((d / "c" / "h.py").read_text(), new)
        self.assertEqual((d / "p" / "h.py").read_text(), old)
        self.assertEqual(patch.lint(diff, ["task_x"], ["[m1]"]), [])
        self.assertTrue(patch.lint(diff.replace("[m1]", "task_x"), ["task_x"], ["[m1]"]))
        self.assertTrue(patch.lint(diff, ["task_x"], ["[m1]"], frozen=["h.py"]))           # frozen file edited
        mem = diff.replace("h.py", "memory/task_y.md")
        self.assertTrue(patch.lint(mem, ["task_x", "task_y"], ["[m1]"], ["task_x"], memory_dir="memory"))
        self.assertFalse(patch.lint(mem + diff, ["task_x", "task_y"], ["[m1]"], ["task_y"], memory_dir="memory"))
        diff2 = "".join(difflib.unified_diff([], ["log('[m2]')\n"], "a/g.py", "b/g.py"))     # merge two components
        patch.apply(d / "p", d / "m", [diff, diff2])
        self.assertIn("[m1]", (d / "m" / "h.py").read_text())
        self.assertIn("[m2]", (d / "m" / "g.py").read_text())
        shutil.rmtree(d)


class TestToyLoop(unittest.TestCase):
    def test_end_to_end(self):
        toy = ROOT / "examples" / "toy"
        shutil.rmtree(toy / "_work", ignore_errors=True)
        r = subprocess.run([sys.executable, "-m", "rsi.loop", "--config", str(toy / "rules.yaml")], cwd=ROOT,
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr)
        recs = [json.loads(l) for l in (toy / "_work" / "ledger.jsonl").read_text().splitlines()]
        dec = {x["summary"]: x["decision"] for x in recs if x["stage"] in ("screen", "merge")}
        self.assertEqual(dec["log a re-check after a missed grasp"], "DROP")     # survives the screen, conflicts at merge
        self.assertEqual(dec["lift every object high before placing"], "DROP")    # dropped by the screen
        self.assertEqual(dec["side grasp for tall objects (height from perception)"], "CONTINUE")   # survives the screen
        final = [x for x in recs if x["stage"] == "final"]
        self.assertEqual([x["decision"] for x in final], ["ACCEPT"])
        self.assertEqual(len(final[0]["components"]), 1)


if __name__ == "__main__":
    unittest.main()
