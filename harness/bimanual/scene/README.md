# bimanual/scene

RoboTwin scene construction used by the RoboTwin harness (`../planner.py`, through `../scene_runner.py` -> `cot_l2.py`).

The harness needs only a small part of this code: `SimEnv` (RoboDawn's `RoboTwinDiscreteEnv` with the official seed
protocol, cameras and motion execution, `oh/simenv.py` + `pilot/record.py`), scene-name handling (`rename_thin`),
snapshots and the shared tool vocabulary. These modules come from the tooling we used to build scenes and replay the
RoboTwin expert; they import each other at module level, so they are shipped together and unchanged in behaviour.
Functions for data recording, checker experiments and earlier closed-loop baselines are not used by the harness or the
evaluation scripts.

`frozen/executor_system_vp2f.txt` is read at import time by `cotlib.py`; it is not part of the H* prompt.
