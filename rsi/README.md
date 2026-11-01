# rsi: recursive self-improvement for agent harnesses

This package implements the harness-evolution protocol of the TinyAstral paper (Method, *Stage 1: Evolving the teacher
harness*; Supplementary, *RSI details*) as a small, model-agnostic loop that works with any harness and benchmark.
A frozen teacher acts inside a harness. A master reads the teacher's trajectories and proposes harness edits. Edits are
screened, merged and accepted only if a paired test on fresh selection seeds passes a gate that humans fixed before the
loop started. Every candidate, accepted or not, is written to a capability ledger.

Python 3.9+, standard library only (PyYAML is used if installed).

```
               rules.yaml (fixed by humans in advance: seeds, gate, budget)            teacher pi_T (frozen)
                                     |                                                          |
 H_k --run on fresh dev seeds--> failed trajectories --> master --> components (diff + target cluster + markers)
  ^                                                                         |
  |               per component: lint + sandbox copy + audit -> direction screen on a small fresh block
  |                                                                         |
  |               survivors merged into one candidate H'; H_k and H' interleaved on the SAME pre-registered
  |               (task, seed) selection pairs -> interim futility check -> pre-registered gate
  |                                                                         |
  +----------- ACCEPT: H_{k+1} = H'                 REJECT / STOP / DROP / INADMISSIBLE
                                    \______________________ ledger.jsonl ______________________/
 after the last round: H* is frozen; the eval split is used once, outside the loop (python -m rsi.evaluate)
```

## Mapping to the paper

| Paper (Method / Supplementary) | Here |
|---|---|
| The system is the frozen teacher $\pi_T$, the harness $H$ and the master $M$. Only $H$ changes. | `rules.yaml: teacher` (fixed endpoint and decoding); `runner.max_calls` (same per-episode budget for both arms); `harness/` is copied to `versions/H0`, and every accepted candidate becomes a new `versions/H{k}` |
| Humans specify the rules, including the pre-registered acceptance criteria, and the budget before the loop begins | `rules.yaml`; its SHA-256 is stored in every ledger record |
| The master is a set of frontier LLM agents that read trajectories, propose edits, implement them and audit one another's code | `rsi/proposer.py`: any OpenAI-compatible model (`kind: llm`), humans or CLI agents (`kind: manual`), or your own class; `audit_command` runs the audit. Brief template: `prompts/master_brief.md` |
| **1. Read trajectories** on development seeds; assign each failure to one of ten failure classes | `loop.py` (`dev` split, fresh block each round); the ten classes are listed in the brief |
| **2. Propose edits**: general (no task-name branching); system prompt and existing tool descriptions byte-identical to the parent; memory changes only for targeted tasks; every new component fires through an explicit trigger that writes a component-specific marker; an audit checks that the untriggered path matches the parent | `patch.lint` (task names may not appear in added code; `frozen_paths` must not change; notes in `memory_dir` may change only for target tasks; each marker must be written by the added code); `audit_command`; the gate also rejects a candidate whose marker appears in a parent trajectory |
| **3. Test and accept edits** on the same task/seed pairs; training, selection and evaluation seeds are disjoint; evaluation seeds are excluded | `seeds.py` (disjoint `train` / `dev` / `selection` / `eval` ranges, a forward-only cursor so every test uses fresh seeds, `train` and `eval` refused inside the loop); `evaluate.evaluate_paired` (both arms on the same pairs, interleaved) |
| *Direction screen*: discard a candidate if the task-averaged 95% upper bound of its gain on the target cluster is below +2 pp; survivors are merged into one candidate $H'$, which runs interleaved with $H_k$ on about 1,150 pre-registered pairs | `gate.decide(mode="screen")` per component (`screen_pairs_per_task`, `screen_min_upper_pp`); `loop.run_round` merges the survivors (a component that no longer applies on top of the others is dropped for this round); `selection.pairs_per_task` sets the pair count |
| *Statistical verdict*: $J$ = target pairs plus non-target pairs where a candidate marker fired, $U$ = the rest. Accept if the one-sided 97.5% lower bound on $J$ (task-stratified bootstrap) is > 0 and the target net wins are > 0. Veto if $U$ regresses by at least 2 pp with a two-sided 95% upper bound below 0. Futility stop at the halfway point if the net gain on $J$ is ≤ 0 or the conditional power is < 10%. The number of pairs is fixed in advance. | `gate.decide` (`alpha: 0.025`, `veto_untriggered_pp: -2`, `interim_fraction: 0.5`, `interim_min_power: 0.10`, `interim_planned_gain_pp: 3`) |
| *Capability ledger*: every candidate, with the failure that triggered it, its proposer, its acceptance evidence and its fate | `ledger.jsonl` (`rsi/ledger.py`): `failure_class`, `failure_detail`, `proposer`, `stats`, `decision`, `capabilities`; rejected candidates are kept |
| *From evolution to a frozen teacher system*: accepted snapshots $H_0 \to \dots \to H^*$ | `versions/H0, H1, ...`; `state.json` names the current version |
| *Task memory*: one short end-state note per task, refined on development seeds, frozen at evaluation | Notes live in the harness under `memory_dir`, so memory edits go through the same lint, screen, gate and ledger |

The bootstrap resamples seeds within each task and weights tasks equally. With $\alpha = 0.025$ the same quantiles give
the one-sided 97.5% lower bound on $J$ and the two-sided 95% bounds used by the screen and the veto.

## Layout
```
rules.yaml               template of the pre-registered rules (teacher endpoint, seeds, gate, budget)
rsi/config.py            load and hash the rules
rsi/seeds.py             train / dev / selection / eval splits; fresh blocks; train and eval refused inside the loop
rsi/teacher.py           client for the frozen teacher (OpenAI-compatible, images supported)
rsi/evaluate.py          run a harness version on (task, seed) pairs; paired, interleaved evaluation; final eval CLI
rsi/gate.py              pre-registered paired gate (screen / interim / final)
rsi/patch.py             apply unified diffs in a sandbox copy; lint (generality, frozen files, memory, markers)
rsi/proposer.py          master interface: llm / manual / custom
rsi/ledger.py            append-only capability ledger + loop state
rsi/loop.py              the loop
prompts/master_brief.md  brief given to the master each round
examples/toy/            mock benchmark, mock teacher, toy harness with a deliberate weakness, scripted master
tests/test_rsi.py        unit tests + end-to-end toy run
```

## Plug in your harness and benchmark
1. Put the starting harness in a directory (`harness: harness_v0`). Everything the master may edit (code, prompts, task
   memory) lives there. List files that must stay byte-identical in `frozen_paths`, and put task memory under
   `memory_dir` as `<task>.md`. The harness calls the teacher through `rsi.teacher.Teacher` or its own client.
2. Write a runner that plays one episode and writes `{out}/result.json`:
   `{"success": bool, "rejected": bool, "trace": "<path to a text log of tool calls, tool results and reasoning>"}`.
   Set `rejected: true` only when a seed is infeasible for reasons that have nothing to do with the harness.
   Configure it as `runner.command` (shell template) or `runner.callable` (`module:function`). Command runners receive
   `TEACHER_BASE_URL`, `TEACHER_MODEL`, `TEACHER_TEMPERATURE` and `MAX_CALLS` in the environment.
3. Fill in `rules.yaml`: tasks, disjoint seed ranges, pairs per test, gate, budget and the proposer.
   With 50 tasks, `selection.pairs_per_task: 24` gives about the 1,150 pairs per test used in the paper.
4. Run `python -m rsi.loop --config rules.yaml`. With `proposer.kind: manual`, each round writes
   `rounds/r{k}/BRIEF.md` and waits for `rounds/r{k}/candidates/<name>.json` + `<name>.diff` and a `DONE` file.
5. After the loop, evaluate the final version once:
   `python -m rsi.evaluate --config rules.yaml --version H3`.

The default teacher in `rules.yaml` is the open-weight model used in the paper (Qwen3.8-Flash-Next) behind an
OpenAI-compatible server. Any frozen model can take its place.

## Toy example (CPU, a few seconds)
```bash
python -m rsi.loop --config examples/toy/rules.yaml
python -m unittest discover -s tests
```
The toy benchmark is synthetic: ten pick-and-place tasks with hand-set success probabilities, so its numbers only show
how the loop behaves. The toy harness always grasps from the top, which rarely works on tall objects. In round 1 the
scripted master proposes three components: a neutral re-check log, a harmful "lift everything high", and a side grasp
for tall objects (height from perception). Output:
```
[round 1] H0: dev 49/100; 51 failed trajectories
  H0.r1c0 [log a re-check after a missed grasp]: screen CONTINUE  target +0.0pp (upper +12.5)
  H0.r1c1 [lift every object high before placing]: screen DROP  target -10.0pp (upper +0.0)
  H0.r1c2 [side grasp for tall objects (height from perception)]: screen CONTINUE  target +52.5pp (upper +67.5)
  H0.r1 = H0.r1c2: ACCEPT  J +50.0pp (lower +41.2), target net +80, U +2.9pp
  -> H1 = H0.r1
[round 2] H1: dev 70/100; 30 failed trajectories
[round 2] master proposed nothing; stopping
final harness H* = H1 after 2 rounds, 1600 episodes; ledger examples/toy/_work/ledger.jsonl
```
The screen drops the harmful component. The neutral one passes the screen but edits the same lines as the side grasp,
so it does not apply on top of it and is dropped from the merge (recorded as `stage: merge` in the ledger).

Markers should fire when a new component acts, not when an outcome is observed: pairs where a candidate marker fires
are judged in $J$, so a marker that fires only on failures pulls failed non-target pairs into $J$ and lowers its gain.

## Rules the code cannot enforce
- The harness must not read privileged simulator state at runtime (ground-truth poses, contact points, success
  signals).
- The audit must be done by someone other than the author of the candidate.
- Do not add episodes after a verdict to reach significance. Do not edit `rules.yaml` mid-run: the hash in the ledger
  will show it.
