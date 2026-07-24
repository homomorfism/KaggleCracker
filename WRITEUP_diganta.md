# Milestone 1 — Diganta — Execute & Submit slice

**Commit:** https://github.com/homomorfism/KaggleCracker/commit/ab7a31feb79076fcbb1f7d8e07b4e6bdfebb0977

## What I built

Two tools. `run_experiment(code, dataset_ref, timeout_s, cv_folds)` executes a
training script in a subprocess confined to `workspace/`, enforces a wall-clock
timeout, and parses a required `CV_SCORE: <float>` line from stdout.
`record_experiment_result(experiment_id, cv_score, fold_scores, notes)` appends
to `leaderboard.jsonl` and returns the best score so far.

`timeout_s` is capped at 1800 **in the schema**, not in the function body, so an
over-long run is rejected by the validator before a subprocess is ever spawned.
I originally also bounded `cv_score` at `min=0.0` and removed it: our competition
metric is still unconfirmed, and valid metrics can be negative. A constraint that
rejects real input is worse than no constraint. I replaced it with
`fold_scores` length ≤ `_MAX_CV_FOLDS`, which is certainly wrong when violated
because you cannot score more folds than you ran — a bound derived from another
bound rather than a guess.

## The loop

Observe = the tool result appended to `messages`; reason = the model's next
choice; act = `dispatch`; verify = the parsed CV score checked against recorded
history. Three stopping conditions: a done-signal, a step cap that raises
`StepLimitReached` rather than returning silently, and plateau detection — stop
when the best CV has not improved by more than `min_delta` across the last N
experiments. The plateau detector reads `HIGHER_IS_BETTER` live, so one constant
flips the direction of "improvement" once the metric is confirmed.

## The gate

`submit_to_kaggle` is my irreversible action: it spends one of a capped daily
quota and is publicly recorded. The gate shows the file, its shape, its first
three rows, the CV score behind it, the message, and how many submissions remain,
then accepts only `yes`/`y` — `""`, `ok` and `sure` are denials. A denial returns
`denied_by_human` into the loop as an observation, so the model can revise rather
than crash. `run_experiment` and `record_experiment_result` stay ungated: a wasted
experiment costs time, and an appended JSONL line can be deleted.

Validation runs in a `precheck`, before the gate. Row count and column names are
compared against `sample_submission.csv`, and a mismatch returns `bad_input`
without ever prompting a human. A human approval gate is not a validation layer;
putting it where validation belongs makes the system *less* safe, because it
looks like a check.

## The error branch

`run_experiment` separates a crash (`exec_failed`), a hang (`timeout`,
`retryable=False` — an identical rerun cannot succeed), and the dangerous case:
a script that exits 0 and never prints the score line (`bad_input`,
`retryable=True` — the fix is one print statement). Collapsing that third case
into `exec_failed` would tell the model to debug a crash that never happened.

`break_it_on_purpose.py` runs the same scenario twice. The fixture prints
`mean auc = 0.8814`, a log line and not a cross-validation. In
`runs/without_branch.txt` the naive tool scrapes the last float from stdout and
returns it as success; the model records 0.8814 as its best score and marks it a
submission candidate. In `runs/with_branch.txt` the identical model, given the
identical script, is told `bad_input` and records nothing. The model's logic is
the same in both runs — only what the tool told it differs.

## How I tested it

A `FakeModel` harness replays scripted tool calls — no API key, deterministic,
and it reaches branches a real model would only hit by accident. 91 tests, every
one routed through the real `dispatch()` path so schema defaults and rejection
order match production. Coverage includes schema rejection asserting no script
file was written, one test per error kind with its `retryable` flag, gate approve
and deny both asserting on the filesystem, and plateau detection firing mid-run
rather than merely refusing to start.

**What surprised me:** `dispatch()` wrapped the tool body in `try/except` but not
the `precheck`. So a precheck that raised — the layer I had just added
specifically to make the system safer — escaped as a raw traceback into the
transcript, which is exactly the failure requirement 4 exists to prevent. I found
it while writing the submission precheck, not from a test.

A review of that commit turned up two more of the same shape, and between them
they sharpened the lesson. `_record()` and `tool_message()` read the envelope
*outside* the `try`, so a tool returning a bare value crashed the loop instead of
producing an error branch: the body was guarded, reading its result was not. And
`validate()`, whose docstring says it never raises, raised on arguments that were
not a dict, because every line below assumed a mapping. So the net has to cover
not just the layers that run untrusted code but the layers that read what those
layers returned — and a promise in a docstring is not a check until something
enforces it. In all three cases I had reasoned carefully about the dangerous path
and stopped one line short of it.

---

*Still unconfirmed pending the Day-0 checklist: the S6E7 metric direction
(`HIGHER_IS_BETTER`) and the daily submission quota (`DAILY_QUOTA`). Both are
single marked constants.*
