"""Append-only JSONL ledgers.

ledger.jsonl  the capability ledger: one record per stage of every candidate, accepted or not:
  {"time", "round", "candidate", "parent", "summary", "proposer", "failure_class", "failure_detail", "target_tasks",
   "markers", "capabilities", "components" (for a merged candidate), "diff_sha256", "rules_sha256", "selection_offset",
   "stage" (lint|screen|merge|interim|final|promote), "stats": {...gate output...},
   "decision" (ACCEPT|REJECT|STOP|DROP|CONTINUE|INADMISSIBLE), "reasons", "accepted_version" (new version id or null)}
state.json    current version, selection-seed cursor, episodes used
"""
import datetime
import json
import pathlib


class Ledger:
    def __init__(self, workdir):
        self.dir = pathlib.Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.jsonl"

    def records(self):
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()] if self.path.exists() else []

    def append(self, rec):
        rec = {"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), **rec}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def summary(self, last=30):
        """Short text of earlier attempts for the proposer, so rejected ideas are not re-proposed blindly."""
        rows = []
        for r in self.records()[-last:]:
            s = r.get("stats") or {}
            rows.append(f"- r{r['round']} {r['candidate']} ({r['summary']}): {r['decision']}"
                        + (f", J {s['J_gain_pp']}pp lower {s['J_lower_pp']}" if "J_gain_pp" in s else "")
                        + (f", target {s['target_gain_pp']}pp upper {s['target_upper_pp']}" if "target_upper_pp" in s else "")
                        + (f"; {'; '.join(r['reasons'])}" if r.get("reasons") else ""))
        return "\n".join(rows)

    def state(self, root_version):
        f = self.dir / "state.json"
        return json.loads(f.read_text()) if f.exists() else {"version": root_version, "sel_cursor": 0, "episodes": 0, "round": 0}

    def save_state(self, st):
        (self.dir / "state.json").write_text(json.dumps(st, indent=1))
