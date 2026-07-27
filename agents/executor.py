"""The executor agent: run a submission past the critic, then the human gate.

It wraps submit_to_kaggle but never calls it blindly. The critic runs FIRST, and
the executor branches purely on the coordination object's fields — never on its
prose:

    status == "pass"     -> proceed to the single human gate (dispatch the
                            irreversible submit, which stops for a yes/y)
    status == "revise"   -> ask the reviser for a corrected proposal and loop,
                            capped at MAX_ROUNDS attempts (the effort budget)
    status == "escalate" -> stop; a human must look. Do not submit.

The gate is not reimplemented here. The executor hands the call to core.loop
.dispatch, so the exact same precheck-then-gate machinery from HW1 runs, and the
irreversible flag on submit_to_kaggle is what makes a human get asked. cv_score
for the submit call is read from the SHARED experiments table, not from the
critic's notes.
"""

from collections import namedtuple

from core.loop import RunState, dispatch
from memory import relational
from agents.critic import review_submission

# Effort budget: at most this many revise attempts before the executor stops and
# leaves it to a human. Named so the cap is one obvious number, not a magic 2.
MAX_ROUNDS = 2

# dispatch() only needs .name and .args off the call object; a tiny local type
# keeps agents/ from importing the test helpers.
_Call = namedtuple("_Call", "name args")


def run_submission(proposal, registry, reviser=None, review=review_submission,
                   max_rounds=MAX_ROUNDS, input_fn=input, output_fn=print):
    """Drive one submission proposal to a terminal outcome.

    Returns a small record of what happened:
        {"status": "submitted" | "denied" | "escalated" | "gave_up",
         "rounds":  <revise attempts used>,
         "reviews": [<each critic object, in order>],
         "submit":  <the dispatch envelope, or None if we never submitted>}

    Whatever the terminal status is, it is also recorded as a row in the SHARED
    submission_outcomes table before this returns. The two agents coordinate
    through shared memory, which is easy to run and hard to debug — so every
    decision the pair reaches leaves a queryable row (status + rounds), and the
    offline monitor can check the table against the transcripts after the fact.
    """
    result = _drive(proposal, registry, reviser, review, max_rounds,
                    input_fn, output_fn)
    relational.record_outcome(
        proposal["experiment_id"], result["status"], result["rounds"]
    )
    return result


def _drive(proposal, registry, reviser, review, max_rounds, input_fn, output_fn):
    reviews = []
    rounds = 0
    current = proposal

    while True:
        obj = review(current)
        reviews.append(obj)
        status = obj["status"]  # branch on the FIELD only; obj["result"] is never read

        if status == "escalate":
            return {"status": "escalated", "rounds": rounds,
                    "reviews": reviews, "submit": None}

        if status == "revise":
            # Cap the loop before doing more work: this is the stopping condition
            # that makes an unfixable proposal terminate instead of spinning.
            if rounds >= max_rounds or reviser is None:
                return {"status": "gave_up", "rounds": rounds,
                        "reviews": reviews, "submit": None}
            rounds += 1
            current = reviser(current, obj)
            continue

        if status == "pass":
            # needs_approval is the field that says a human must sign off. It is
            # True on a pass, and submit_to_kaggle is irreversible, so dispatch
            # will run the gate. We never flip dry_run: the safe default holds
            # unless a human explicitly changes it elsewhere.
            exp = relational.get_experiment(current["experiment_id"])
            submit_args = {
                "submission_path": current["submission_path"],
                "sample_submission_path": current["sample_submission_path"],
                "message": current["message"],
                "cv_score": exp["cv_score"],  # from shared memory, not from prose
            }
            env = dispatch(_Call("submit_to_kaggle", submit_args), registry,
                           RunState(), input_fn=input_fn, output_fn=output_fn)
            return {"status": "submitted" if env["ok"] else "denied",
                    "rounds": rounds, "reviews": reviews, "submit": env}

        # The critic returned a status outside the contract. Stop loudly rather
        # than guess what it meant — a broken contract is not valid data.
        raise ValueError("unknown critic status: %r" % (status,))
