# HW2 — Shamil's slice: outcomes in shared memory, a monitor on its own clock, and tests for the judge

## What this commit does

Our HW2 base (stores, executor + critic, offline judge) landed in PR #6. This
commit closes the three gaps that were still open against the assignment, all
inside my slice of the repo (the exec path and `tests/`):

1. **Submission outcomes now live in the relational store.** Our own
   `CLAUDE.md` says SQLite holds "runs, CV scores, hyperparameters, submission
   outcomes" — but nothing ever recorded an outcome. `memory/relational.py`
   gains a `submission_outcomes` table plus `record_outcome` / `query_outcomes`,
   and the executor writes exactly one row (status + revise rounds) for every
   proposal it drives to a terminal state: `submitted`, `denied`, `escalated`,
   or `gave_up`. The status set is closed — an off-list status raises at the
   point of failure, the same posture as `core/contracts.err` refusing an
   unknown error kind.

2. **The monitor now has its own clock.** The assignment asks for a monitor
   "on its own schedule", not a manual step. `monitor/clock.py` runs the
   existing judge on an interval (`python monitor/clock.py --every 900`),
   entirely outside the request/response loop — it imports the judge and a
   sleep function, never the agents, tools, or model. The loop has an explicit
   stopping condition (`--max-runs`) and an injectable `sleep`, so tests can
   prove it ticks and stops without spending wall-clock time.

3. **The judge itself is now tested.** It is graded code that grades other
   code, and it had zero tests. `tests/hw2/test_monitor.py` feeds it
   transcripts engineered to earn each verdict and asserts on the rationale,
   not just the label — a verdict without its `{expected, got}` is
   indistinguishable from a hallucination, so the tests demand both.

Evidence: `runs/hw2/outcomes_recorded.txt` (new trace — the same proposal
driven to an approved dry-run and to a denial, then the outcomes table dumped
as the audit trail) and `monitor/report.json` (regenerated over all six
transcripts; it still reports the real problem it found — `rule_changes.txt`
contains an unguarded offer to submit, flagged as a `serious_violation` with
its expected/got rationale).

## Ideas from class this puts into practice

- **Pick the store by the shape of the thing.** An outcome is a status from a
  closed set, a round count, and a timestamp — fixed columns you filter on, so
  it belongs in SQLite next to the experiments it refers to, not in the
  free-form JSON store.
- **LLM-as-a-judge as a background job.** The judge grades on named values
  (`strictly_adheres` / `minor_violation` / `serious_violation`, plus a
  completion axis), every violation carries an expected/got rationale, and the
  clock makes it a job on its own schedule rather than a step inside the loop
  it grades — a grader inside the loop could be steered by the very run it is
  checking.
- **Coordination by field, evidenced by record.** The agents already branch on
  `{status, result, needs_approval}`, never on prose; this commit makes the
  *decisions* they reach queryable too.

## Why our agent architecture is shaped the way it is

We split executor and critic because the critic's job is to be a check the
executor cannot skip: submissions to a live competition are daily-capped and
irreversible, so the rubric (row counts, CV floor, recorded model type) runs
as a separate agent whose verdict the executor can only branch on — it cannot
argue with prose or grade its own work. A single agent would have cost us
exactly that: self-certification, where the same context that produced a weak
submission also talks itself into sending it, and every mistake spends a
capped, unrecoverable slot. (Of the four reasons to go multi-agent, ours is
the independent-verifier one.)

The two agents coordinate through shared memory, which we knew from class is
the hardest coordination to debug: there is no call stack connecting a write
to the read that acted on it, so "why did the executor do that?" has no single
transcript to replay. This commit is our mitigation — every terminal decision
leaves a row with its status and round count, so the history of the
coordination is a query (`query_outcomes`), and the offline monitor can be
checked against the same table the agents used, not against reconstructed
prose.

## How I tested it

`python -m pytest -q` — 229 passed (214 before this commit, 15 added).

- **Judge verdicts** (`tests/hw2/test_monitor.py`): an unguarded submit offer
  earns `serious_violation` and the rationale names the offending utterance; a
  guarded offer is clean with no rationale; a planted injection that the agent
  refuses is clean but carries an observation, while the same injection with
  no refusal is a `serious_violation`; a transcript with no final answer, or a
  step-limit stall, fails the completion axis; `build_report` over a directory
  flags only the bad transcript and the flag serializes to JSON.
- **The clock**: with an injected sleep, three scheduled runs tick exactly
  three times and nap exactly twice (immediately-first, stop-at-cap), and a
  single-run clock never sleeps at all.
- **Outcomes** (`tests/hw2/test_outcomes.py`): every test asserts the
  consequence in the table, not just the return value — an approval leaves a
  `submitted` row; a denial leaves a `denied` row *and* no quota file exists;
  the give-up path records exactly the 2-round effort budget; an escalation is
  on the record; an off-list status raises and leaves the table untouched.
- All state is redirected to tmp (`KC_DB`, `KC_WORKSPACE`); nothing touches
  the real `workspace/`, no network, no API key.
