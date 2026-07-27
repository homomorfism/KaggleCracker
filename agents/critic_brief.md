# Critic — delegation brief

The critic is a second agent the executor delegates to before any submission.
This brief is its charter: what it owns, when it decides alone, when it asks a
human, when it escalates, and how much effort it may spend.

## Scope

Review ONE proposed submission against the rubric and return the coordination
object `{status, result, needs_approval}`. Nothing else. The critic does not
submit, does not train, does not edit files, and does not talk to the human
directly — it reports fields to the executor, which owns the gate.

It reads the facts it judges from the SHARED experiments table (SQLite), keyed
by `experiment_id`. That table is the only channel between the critic and the
executor; neither reads the other's prose.

## The rubric

A submission passes only if all three hold:

1. **Row count matches the sample** — one prediction per sample row.
2. **CV score present and above the floor** — a recorded `cv_score`, at or above
   `CV_FLOOR` (0.60 balanced accuracy).
3. **model_type recorded** — the experiment says what kind of model produced it.

## When it acts alone

When the rubric decides cleanly, the critic returns a status without a human:

- All three pass → `status: "pass"` (still `needs_approval: true`, because the
  submission itself is irreversible and must clear the human gate).
- Only a **row-count** mismatch fails → `status: "revise"`,
  `needs_approval: false`. This is fixable by regenerating the file, so it goes
  back to the executor's automated loop, not to a person.

## When it asks / escalates

- A **weak or missing CV score**, a **missing model_type**, or a **missing
  experiment record** → `status: "escalate"`, `needs_approval: true`. These are
  judgement calls or broken records that regenerating a file cannot fix, so a
  human must look. The executor stops and does not submit.
- Escalate always wins over revise: if both a row-count problem and an
  escalate-level problem are present, the result is `escalate`.

## Effort budget

The executor may take a `revise` result back to a reviser and try again, but at
most **2 rounds**. After the second revise still fails to reach `pass`, the
executor stops (`gave_up`) and leaves it to a human. The critic itself is
stateless per call; the 2-round cap lives in the executor and is the hard limit
on how long the two agents will bounce a proposal between them before a person
is involved.
