#!/usr/bin/env python
"""cot_v1 phase 1 (sim part): L2 execution verification of the L1-regenerated actions with the frozen V+ tools.
Per episode (data seed, demo_randomized):
  0 replay the scripted expert without failure branches (recorder hooks, no rendering) and keep an in-memory sim
    snapshot (snapshot.py: actor poses, articulation / robot joints, grippers) at every state; the k-th replay state is
    aligned with the k-th clean recorded state (branch states excluded) and checked (objects within 3 cm, held, gripper).
  1 per step: restore the snapshot of the step's observation, execute the regenerated call with the V+ tools; compare
    its effect with the recorded after-state (held objects, gripper states, object displacements within 3 cm,
    articulation within 0.15, task_ok, acting-arm tcp within 2 cm for pure motions). Also execute the GT call with the
    same tools from the same snapshot (GT-tool fidelity); a regenerated call whose effect equals the GT call's effect
    passes as well (pass_vs_gt_tool). 'skipped locate' (grasp where GT is locate) is compared with the grasp after it.
  2 chain: from the first observation execute the regenerated calls one after another (no restores); stop at the first
    step whose result no longer tracks the recorded after-state; chain = steps survived.
Recovery (branch) turns have no snapshots -> L1 only.
Usage: cot_l2.py --task <task> --worker k  (claims episodes of that task from cot_v1/l2/_todo.txt)"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "frozen"))
import cotlib as L  # noqa: E402
import ub_vplus2 as VP  # noqa: E402  (frozen copy; imports g50_cl -> g50_build patches, tools)
import snapshot as SN  # noqa: E402
import v3_record as V3R  # noqa: E402
from record import ArmTag, rename_thin, SimEnv  # noqa: E402
from envs.utils.actor_utils import Actor, ArticulationActor  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402

G, CL = VP.G, VP.CL
OUT = L.D / "l2"


def _actors(T):                      # as g50_record: articulated actors are restored through their joints
    out = {}
    for k, v in vars(T).items():
        if isinstance(v, Actor) and not isinstance(v, ArticulationActor) and id(v) not in {id(x) for x in out.values()}:
            out[k] = v
    return out


SN._actors = _actors


def r1(v):
    return [round(float(x), 1) for x in v]


def truth(sim, label, idx):
    import transforms3d as t3d
    T = sim.T
    tz = sim.table_z
    objs = {}
    for n in sim.names:
        lo, hi = sim.aabb(n)
        lo = (lo - [0, 0, tz]) * 100
        hi = (hi - [0, 0, tz]) * 100
        q = np.asarray(sim.names[n].get_pose().q)
        yaw = math.degrees(t3d.euler.quat2euler(q, axes="sxyz")[2])
        objs[n] = {"center": r1((lo + hi) / 2), "size": r1(hi - lo), "lo": r1(lo), "hi": r1(hi), "yaw": round(yaw, 1)}
    arms = {}
    for arm in ("left", "right"):
        tcp = sim.tcp(arm)
        arms[arm] = {"tcp": r1([tcp[0] * 100, tcp[1] * 100, (tcp[2] - tz) * 100]),
                     "gripper_cmd": round(sim.gripper_cmd(arm), 2), "opening": round(sim.opening(arm), 2)}
    held = sim.held()
    st = {"idx": idx, "label": label, "frames": {}, "objects": objs, "arms": arms, "visible": {},
          "held": {a: (held.get(a) if arms[a]["gripper_cmd"] <= 0.8 else None) for a in ("left", "right")},
          "contacts": [list(c) for c in sorted({(a, o) for a, o, _ in sim.contacts()})], "physics_steps": sim.physics_steps}
    try:
        st["task_ok"] = bool(T.check_success())
    except Exception:  # noqa: BLE001
        st["task_ok"] = None
    art = {}
    for n, v in sim.names.items():
        if isinstance(v, ArticulationActor):
            try:
                art[n] = [round(float(x), 4) for x in v.get_qpos()]
            except Exception:  # noqa: BLE001
                pass
    st["art"] = art
    return st


class SnapRec(V3R.Rec3):
    """expert replay without branches; truth-only snapshots + in-memory sim snapshot per state."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.snaps = []

    def snap(self, label):
        idx = len(self.states)
        self.states.append(truth(self.sim, label, idx))
        self.snaps.append(SN.save(self.T))
        return idx

    def branch(self, moment, injection, fn):
        return None


class Shim:
    """minimal Episode for the V+ tools: truth snapshots without rendering."""

    def __init__(self, sim, task):
        self.sim, self.task = sim, task
        self.states = []
        self.t0 = time.time()
        self.dir = OUT / "_tmp"
        self.tool_log = []

    def trace(self, rec):
        pass

    def snap(self, label):
        st = truth(self.sim, label, len(self.states))
        self.states.append(st)
        return st


def effect_diff(S0, A, B, arm=None, motion=False):
    """differences between two outcomes A, B of the same start S0 (empty list = same effect)."""
    d = []
    for a in ("left", "right"):
        if (A["held"].get(a) or None) != (B["held"].get(a) or None):
            d.append(f"held[{a}] {A['held'].get(a)} vs {B['held'].get(a)}")
        if (A["arms"][a]["gripper_cmd"] > 0.5) != (B["arms"][a]["gripper_cmd"] > 0.5):
            d.append(f"gripper[{a}]")
    for n in S0["objects"]:
        c0, ca, cb = (np.asarray(X["objects"][n]["center"]) for X in (S0, A, B))
        if np.linalg.norm(ca - c0) < 2 and np.linalg.norm(cb - c0) < 2:
            continue
        if np.linalg.norm(ca - cb) > 3:
            d.append(f"{n} off by {np.linalg.norm(ca - cb):.1f} cm")
    for n, q in (S0.get("art") or {}).items():
        qa, qb = (A.get("art") or {}).get(n), (B.get("art") or {}).get(n)
        if qa and qb and max(abs(x - y) for x, y in zip(qa, qb)) > 0.15:
            d.append(f"{n} joint differs")
    if A.get("task_ok") != B.get("task_ok"):
        d.append(f"task_ok {A.get('task_ok')} vs {B.get('task_ok')}")
    if motion and arm in ("left", "right"):
        e = np.linalg.norm(np.asarray(A["arms"][arm]["tcp"]) - np.asarray(B["arms"][arm]["tcp"]))
        if e > 2:
            d.append(f"{arm} tcp off by {e:.1f} cm")
    return d


def track(sim_st, rec_st):
    """does the sim still match the recorded observation? (objects within 3 cm, held, gripper)"""
    d = []
    for a in ("left", "right"):
        if (sim_st["held"].get(a) or None) != (rec_st["held"].get(a) or None):
            d.append(f"held[{a}]")
        if (sim_st["arms"][a]["gripper_cmd"] > 0.5) != (rec_st["arms"][a]["gripper_cmd"] > 0.5):
            d.append(f"gripper[{a}]")
    for n, o in rec_st["objects"].items():
        if n in sim_st["objects"] and np.linalg.norm(np.asarray(sim_st["objects"][n]["center"]) - np.asarray(o["center"])) > 3:
            d.append(f"{n} {np.linalg.norm(np.asarray(sim_st['objects'][n]['center']) - np.asarray(o['center'])):.1f} cm")
    return d


def execute(tools, ep, call):
    name, args = call.get("name"), call.get("arguments") or {}
    try:
        fb, S = tools.call(name, dict(args))
    except (TypeError, ValueError, KeyError) as exc:
        fb = {"ok": False, "stage": "error", "note": f"bad call: {exc}"}
    except Exception as exc:  # noqa: BLE001
        ep.sim.T.plan_success = True
        fb = {"ok": False, "stage": "error", "note": f"{name} failed: {type(exc).__name__}: {exc}"[:200]}
    St = ep.snap("after")
    return {k: fb.get(k) for k in ("ok", "stage", "note")}, St


def same_call(a, b):
    return L.normalise(a) == L.normalise(b)


def setup(sim, seed):
    T = sim.rd.env
    T.setup_demo(now_ep_num=0, seed=seed, is_test=True, **sim.rd.args)
    sim.rd._episode_open = True
    sim.T = T
    T.eval_success = False
    T.plan_success = True
    sim.rd._create_agent_camera()
    sim._fingers()
    sim._build_names()
    rename_thin(sim)
    return T


def replay_expert(sim, task, seed):
    T = sim.T
    rec = SnapRec(sim, OUT / "_tmp", task, seed, False, np.random.RandomState(0), task=task, p_branch=0.0, max_branches=0)
    rec.install()
    try:
        rec.snap("start")
        T.play_once()
        last = [e for e in rec.events if e.get("branch") is None]
        last = last[-1] if last else None
        if T.plan_success and last and last["api"] == "place_actor" and last.get("release"):
            V3R.RV2.RecorderV2.exec_one(rec, T.move_by_displacement(ArmTag(last["arm"]), z=0.05, move_axis="arm"))
    finally:
        rec.uninstall()
    return rec


def align(rec_states, branches, rp_states):
    """recorded clean state idx -> replay idx. Branch interiors (start+1..end) have no replay state; the 'restored'
    state (end+1) is the branch start re-instated. The k-th remaining recorded state = the k-th replay state."""
    inner = {i for b in branches for i in range(b["start_state"] + 1, b["end_state"] + 1)}
    restored = {b["end_state"] + 1: b["start_state"] for b in branches}
    m, bad, k = {}, {}, 0
    clean = []
    for i in range(len(rec_states)):
        if i in inner:
            continue
        if i in restored:
            j = restored[i]
            while j in restored:
                j = restored[j]
            if j in m:
                m[i] = m[j]
            continue
        clean.append(i)
        if k < len(rp_states):
            d = track(rp_states[k], rec_states[i])
            if d:
                bad[i] = d
            else:
                m[i] = k
        k += 1
    return m, bad, len(clean), len(rp_states)


class ToolCtx:
    def __init__(self, sim, task):
        self.ep = Shim(sim, task)
        self.tools = VP.VPTools(self.ep)

    def fresh(self, label="restored"):
        self.tools.last_arm, self.tools.unreach = None, 0
        VP.STATE.update(lock=False, refused=0)
        self.ep.sim.T.plan_success = True
        return self.ep.snap(label)


def run_episode(sim, eid):
    E = json.loads((L.D / "build/episodes" / f"{eid}.json").read_text())
    syn = json.loads((L.D / "synth" / f"{eid}.json").read_text())
    raw = json.loads(Path(E["src"]).read_text())
    rec_states = raw["states"]
    T = setup(sim, E["seed"])
    t0 = time.time()
    rp = replay_expert(sim, E["task"], E["seed"])
    amap, abad, n_clean, n_rp = align(rec_states, raw.get("branches", []), rp.states)
    out = {"episode_id": eid, "task": E["task"], "seed": E["seed"], "names_ok": sorted(sim.names) == sorted(E["objects"]),
           "replay": {"seconds": round(time.time() - t0, 1), "n_clean": n_clean, "n_replay": n_rp, "aligned": len(amap),
                      "bad": {str(k): v for k, v in list(abad.items())[:5]}}, "turns": [], "chain": None}
    ctx = ToolCtx(sim, E["task"])
    clean = sorted([c for c in E["convs"] if c["valid"] and not c["branch"]], key=lambda c: c["subgoal"])
    seq = []                                   # tool turns in episode order (for the chain)
    for c in clean:
        srec = syn["convs"][c["conv_id"]]["turns"]
        for i, t in enumerate(c["turns"]):
            r = srec[i]
            row = {"conv": c["conv_id"], "i": i, "gt": t["gt"], "regen": r["l1"]["call"], "l1": r["l1"]["cmp"][0],
                   "status": r["status"]}
            if t["kind"] == "check":
                row["l2"] = "n/a"
                out["turns"].append(row)
                continue
            seq.append((c, i, t, r))
            if t["obs"] not in amap:
                row.update(l2="no_snapshot", why="; ".join(abad.get(t["obs"], ["beyond replay"]))[:200])
                out["turns"].append(row)
                continue
            snap0 = rp.snaps[amap[t["obs"]]]
            gt, rg = t["gt"], r["l1"]["call"]
            motion = gt["name"] in ("move", "home", "rotate", "cmd")
            arm = (gt.get("arguments") or {}).get("arm")
            Sobs, Saft = rec_states[t["obs"]], rec_states[t["after"]]
            ref_after = Saft
            skip = False
            if rg is not None and gt["name"] == "locate" and L.normalise(rg).get("name") == "grasp" and i + 1 < len(c["turns"]) \
                    and c["turns"][i + 1]["gt"]["name"] == "grasp":
                skip = True
                ref_after = rec_states[c["turns"][i + 1]["after"]]
            # GT through the V+ tools from the same snapshot
            SN.restore(T, snap0)
            S0 = ctx.fresh("start_step")
            fb_g, Sg = execute(ctx.tools, ctx.ep, c["turns"][i + 1]["gt"] if skip else gt)
            fid = effect_diff(Sobs, Sg, ref_after, arm, motion)
            row.update(gt_fb=fb_g, gt_fidelity=not fid, gt_fid_why="; ".join(fid)[:200])
            if rg is None:
                row.update(l2="fail", why="no regenerated call")
            else:
                if same_call(rg, gt):
                    fb_r, Sr = fb_g, Sg
                else:
                    SN.restore(T, snap0)
                    ctx.fresh("start_step")
                    fb_r, Sr = execute(ctx.tools, ctx.ep, rg)
                d_rec = effect_diff(Sobs, Sr, ref_after, arm, motion)
                d_gt = effect_diff(S0, Sr, Sg, arm, motion)
                if not d_rec:
                    row.update(l2="pass", why="")
                elif not d_gt and same_call(rg, gt) is False and fid:
                    row.update(l2="pass_vs_gt_tool", why="; ".join(d_rec)[:200])
                elif not d_gt and fid:
                    row.update(l2="gt_irreproducible", why="; ".join(d_rec)[:200])
                else:
                    row.update(l2="fail", why="; ".join(d_rec)[:300])
                row.update(regen_fb=fb_r, skip_locate=skip)
            out["turns"].append(row)
    # chain: from the first observation, (1) the GT calls back to back through the V+ tools (reference trajectory),
    # (2) the regenerated calls back to back; the chain survives while (2) tracks (1) after every step.
    ch = {"steps": 0, "n": len(seq), "stop": None}
    if seq and seq[0][2]["obs"] in amap:
        s0 = rp.snaps[amap[seq[0][2]["obs"]]]
        SN.restore(T, s0)
        ctx.fresh("gtchain_start")
        ref = []
        for c, i, t, r in seq:
            fb, St = execute(ctx.tools, ctx.ep, t["gt"])
            ref.append(St)
        ch["gt_chain_task_ok"] = bool(ctx.ep.states[-1].get("task_ok"))
        SN.restore(T, s0)
        ctx.fresh("chain_start")
        k = 0
        while k < len(seq):
            c, i, t, r = seq[k]
            rg = r["l1"]["call"]
            if rg is None:
                ch["stop"] = {"k": k, "why": "no call"}
                break
            jump = 1
            if t["gt"]["name"] == "locate" and L.normalise(rg).get("name") == "grasp" and k + 1 < len(seq) and seq[k + 1][2]["gt"]["name"] == "grasp":
                jump = 2
            fb, St = execute(ctx.tools, ctx.ep, rg)
            d = track(St, ref[k + jump - 1])
            if d:
                ch["stop"] = {"k": k, "conv": c["conv_id"], "i": i, "why": ", ".join(d[:4]), "fb": fb}
                break
            ch["steps"] += jump
            k += jump
        ch["task_ok_end"] = bool(ctx.ep.states[-1].get("task_ok"))
    out["chain"] = ch
    try:
        sim.rd.close_episode()
    except Exception:  # noqa: BLE001
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--task", required=True, help="one sim per process, so one task per worker")
    a = ap.parse_args()
    VP.A = Namespace(guard=False, final_check=True, task_checker="v2", wall=720)
    G.G50Tools.V2 = True
    (OUT / "_claims").mkdir(parents=True, exist_ok=True)
    (OUT / "_tmp").mkdir(parents=True, exist_ok=True)
    sim = SimEnv(a.task, seed_dir=str(OUT / "_seedcache"), task_config="demo_randomized", seed_base_index=-1)
    orig = OUT / "_todo.txt"
    while True:
        eid, waiting = None, 0
        for e in [l.strip() for l in open(orig) if l.strip() and l.startswith(a.task + "__")]:
            if (OUT / f"{e}.json").exists() or (OUT / "_claims" / f"{e}.lock").exists():
                continue
            if not (L.D / "synth" / f"{e}.json").exists():
                waiting += 1
                continue
            lk = OUT / "_claims" / f"{e}.lock"
            try:
                fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.uname()[1]} {a.worker} {os.getpid()}".encode())
                os.close(fd)
                eid = e
                break
            except FileExistsError:
                continue
        if eid is None:
            if waiting:
                time.sleep(60)
                continue
            break
        t0 = time.time()
        try:
            res = run_episode(sim, eid)
            res["seconds"] = round(time.time() - t0, 1)
            (OUT / f"{eid}.json").write_text(json.dumps(res, ensure_ascii=False, default=str))
            print(time.strftime("%H:%M:%S"), eid, "replay", res["replay"]["aligned"], "/", res["replay"]["n_clean"], "chain",
                  res["chain"]["steps"], "/", res["chain"]["n"], f"{time.time() - t0:.0f}s", flush=True)
        except UnStableError as exc:
            (OUT / f"{eid}.json").write_text(json.dumps({"episode_id": eid, "error": f"unstable: {exc}"}))
        except Exception as exc:  # noqa: BLE001
            print(eid, "ERROR", traceback.format_exc()[-1500:], flush=True)
            (OUT / f"{eid}.json").write_text(json.dumps({"episode_id": eid, "error": f"{type(exc).__name__}: {exc}"}))
            try:
                sim.rd.close_episode()
            except Exception:  # noqa: BLE001
                pass
    try:
        sim.rd.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
