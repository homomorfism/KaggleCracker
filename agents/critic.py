"""The critic agent: review a proposed submission against a written rubric.

The rubric it enforces is spelled out in agents/critic_brief.md. In code it is
three checks:

  R1  the submission's row count matches the sample's row count
  R2  a CV score is present AND at or above a floor
  R3  a model_type is recorded on the experiment

The facts for R2 and R3 are read from the SHARED experiments table (via
memory.relational), not from the proposal's prose — that table is the only
channel between this agent and the executor. The critic hands back the small
coordination object every agent here speaks in:

    {"status": "pass" | "revise" | "escalate", "result": <notes>,
     "needs_approval": bool}

`result` is prose for a human to read. The executor never parses it; it branches
on `status` and `needs_approval` only.
"""

from memory import relational

# Below this, a submission is not worth one of the day's capped live slots.
# Balanced accuracy of 0.5 is a coin flip, so the floor sits above it. One
# constant, easy to point at when explaining the rubric out loud.
CV_FLOOR = 0.60


def _obj(status, notes, needs_approval):
    return {"status": status, "result": "; ".join(notes), "needs_approval": needs_approval}


def review_submission(proposal, cv_floor=CV_FLOOR, get_experiment=relational.get_experiment):
    """Apply the rubric to one proposed submission and return the coordination
    object. `proposal` carries the two row counts and the experiment id; the
    scored facts are pulled from shared memory.

    Outcome priority is escalate > revise > pass: a problem a human must judge
    (a weak or unscored model, a missing record) is never quietly funnelled into
    the automated revise loop, and only a clean, fixable-by-regeneration issue
    (a wrong row count) is a revise.
    """
    exp = get_experiment(proposal["experiment_id"])
    if exp is None:
        # The record the two agents coordinate through is missing. The revise
        # loop cannot conjure it, so this is a human's problem.
        return _obj(
            "escalate",
            ["experiment %r not found in the shared experiments table; cannot "
             "verify cv_score or model_type" % proposal["experiment_id"]],
            needs_approval=True,
        )

    escalate_notes = []
    revise_notes = []

    # R2: CV score present and above the floor. A weak or missing score is a
    # judgement call about spending a live slot, so it escalates rather than
    # looping — regenerating the file will not raise the model's score.
    cv = exp.get("cv_score")
    if cv is None:
        escalate_notes.append("cv_score is missing from the experiment record")
    elif cv < cv_floor:
        escalate_notes.append(
            "cv_score %.4f is below the floor %.4f" % (cv, cv_floor)
        )

    # R3: model_type recorded. An unlabelled experiment is an incomplete record
    # a human/upstream must fix, not something the revise loop can add.
    if not exp.get("model_type"):
        escalate_notes.append("model_type is not recorded on the experiment")

    # R1: row count matches the sample. This IS fixable by regenerating the
    # file, so a mismatch is the one thing that earns a revise.
    if proposal["submission_rows"] != proposal["sample_rows"]:
        revise_notes.append(
            "submission has %d rows but the sample has %d"
            % (proposal["submission_rows"], proposal["sample_rows"])
        )

    if escalate_notes:
        return _obj("escalate", escalate_notes + revise_notes, needs_approval=True)
    if revise_notes:
        # No human needed yet — the executor should try to regenerate and resubmit.
        return _obj("revise", revise_notes, needs_approval=False)
    return _obj(
        "pass",
        ["rubric satisfied: rows match, cv_score %.4f >= %.4f, model_type %r recorded"
         % (cv, cv_floor, exp["model_type"])],
        # A pass still requires the human gate: the submission itself is
        # irreversible. needs_approval says so; the executor routes it there.
        needs_approval=True,
    )
