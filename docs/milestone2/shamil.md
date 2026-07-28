# Milestone 2 — Shamil — Competition workbench: agents that run without you watching

My slice turns the repo from "a loop you press a key to run" into a workbench
where work starts on its own. One commit adds: project creation from a Kaggle
competition URL (metadata + train/test download via the Kaggle API), a
notebook **observer agent**, a conversational **EDA agent** with a live
dashboard, an **experiment agent** that plans with the human and then writes
and executes training code, and the FastAPI + React surface that shows all of
it (sidebar: Competition / EDA / Experiments / Discussions / Leaderboard).

## What runs without a user message

Every agent is a detached subprocess journaling to JSONL; the server holds no
job state and answers every poll from disk, so it can restart mid-run and lose
nothing.

- **Event trigger (background):** finishing the data download fires the next
  stage — `ui/run_setup.py` spawns the observer and the EDA bootstrap the
  moment `state: done` lands. Nobody sends a message; the project starts
  summarizing community notebooks and building its dashboard on its own.
- **Interactive trigger:** a chat message (EDA or experiment) spawns one
  stateless turn subprocess, re-primed from the durable conversation file.
- **The silence branch:** an observer refresh that finds nothing new
  deliberately produces nothing — and records why. Dedupe is keyed on kernel
  `ref`; the journal line `batch_selected {count: 0, skipped_known: N}` is the
  written record that silence was a decision, not a failure. Same idea in the
  EDA poll: `?version=N` answers `{unchanged: true}` instead of re-sending a
  payload nothing changed in.

## The queue decision

**Reject-while-busy.** Every agent has a mkdir-claim lock; a message arriving
mid-turn gets HTTP 409 and is *not* stored. Cost: the human must resend after
the turn ends (the UI disables the input so this is visible, not surprising).
I chose it over queuing because a queued instruction can be stale by the time
the turn it waited for finishes — with an agent that edits dashboards and runs
training, replaying stale intent is worse than asking again. A `spawn_pending`
marker closes the gap where a just-spawned turn hasn't claimed its lock yet.

## Privilege boundary

The ordinary path is the envelope-contract tool set (validated, capped,
reversible). Two things sit above it, with written boundaries:

- `run_bash` (experiment/EDA agents only): a real shell with network and the
  venv's pip — the capability the sandboxed script runners deliberately lack.
  Its description forbids using it for training runs, which must go through
  `run_experiment` where output is captured and scored.
- The **stop endpoint**: the server can kill a running turn by pid (written
  into the lock). The agent cannot stop itself or others; only the human's
  admin surface can.

The Milestone-1 gate is untouched: `submit_to_kaggle` stays irreversible and
human-gated; every new tool is reversible and ungated on purpose.

## How I tested it

- **Deterministic agent tests, no network:** `tests/ui/test_experiments.py`
  drives the full question → plan → approve → run → finished cycle with
  `FakeModel` — the "run" is a real subprocess whose script prints a CV score,
  and the test asserts consequences: state transitions, the plan file, the
  extracted code attempt, the leaderboard row. Premature `finalize` and
  thin plans are asserted to bounce.
- **Making the triggers fire on purpose:** `tests/ui/test_setup.py` fakes the
  Kaggle client and asserts the success path spawns exactly
  `[ui.run_observer, ui.run_eda]`; the observer tests prove a second refresh
  pulls an already-seen notebook zero times (the silence branch, asserted as
  "pulled exactly once across two refreshes").
- **Failure paths as first-class tests:** kaggle 403 → `rules_not_accepted`
  with an actionable retry, missing creds detected before any network call,
  crashed/оversized/timeout EDA scripts each mapping to their error kind, 409
  under a held lock with the consequence checked (nothing queued behind it).
- **Live end-to-end** on the Titanic competition: URL → download → observer
  summarized 8 notebooks → EDA built a 10-panel dashboard → experiment agent
  proposed a stacking plan citing the community's 0.808 LB claim, and after
  approval ran 5 script attempts to a recorded CV of 0.8417. The live run
  caught two real bugs the unit tests could not: token truncation inside
  `tool_use` blocks, and the loop's 2-consecutive-fails tool disable being
  wrong for a coding agent mid-debug (now a parameter, default unchanged).

337 tests green (`python -m pytest -q`); frontend `tsc`+`oxlint`+build clean.
