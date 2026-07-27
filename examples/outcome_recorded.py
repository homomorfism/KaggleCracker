"""One transcript showing terminal outcomes landing in the shared SQLite table.

The executor drives the same proposal to two different terminal states — an
approved dry-run submit, then a denial at the gate — and each one leaves a row
in submission_outcomes. The point of the trace is the last section: the table
IS the audit trail of the two-agent coordination, so "what did the agents
decide?" is answered by a query, not by rereading prose.

Deterministic and network-free: gate answers are scripted, the workspace and
the database live in a throwaway tmp dir.

Run from the repo root:  python examples/outcome_recorded.py
The transcript lands in runs/hw2/outcomes_recorded.txt.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running as `python examples/...` from root

from agents.executor import run_submission
from core.registry import Registry
from memory import relational
from tools.exec.submit import SUBMIT_TO_KAGGLE

OUT_PATH = ROOT / "runs" / "hw2" / "outcomes_recorded.txt"


def main():
    lines = []
    say = lines.append

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        os.environ["KC_DB"] = str(tmp / "experiments.db")
        os.environ["KC_WORKSPACE"] = str(tmp)
        relational.init_db()

        (tmp / "data").mkdir()
        (tmp / "data" / "sample_submission.csv").write_text("id,target\n1,0\n2,0\n3,0\n")
        (tmp / "submission.csv").write_text("id,target\n1,0.9\n2,0.1\n3,0.5\n")
        relational.add_experiment("exp-42", "alice", "lightgbm", 0.88, "done")

        registry = Registry()
        registry.register(SUBMIT_TO_KAGGLE)

        proposal = {
            "experiment_id": "exp-42",
            "submission_path": "submission.csv",
            "sample_submission_path": "data/sample_submission.csv",
            "message": "S6E7 candidate",
            "submission_rows": 3,
            "sample_rows": 3,
        }

        say("OUTCOMES RECORDED — every terminal decision leaves a shared SQLite row")
        say("=" * 72)
        say("Seeded experiment exp-42 (lightgbm, cv_score=0.88). Same proposal,")
        say("driven to two different terminal states through the executor + critic.")
        say("")

        say("[GATE] scripted human answers 'yes' -> approved dry-run submit")
        first = run_submission(proposal, registry,
                               input_fn=lambda _p="": "yes",
                               output_fn=lambda *_a, **_k: None)
        say("       executor returned status=%r (dry_run held, nothing uploaded)"
            % first["status"])

        say("[GATE] scripted human answers 'no'  -> denial")
        second = run_submission(proposal, registry,
                                input_fn=lambda _p="": "no",
                                output_fn=lambda *_a, **_k: None)
        say("       executor returned status=%r (%s)"
            % (second["status"], second["submit"]["error"]["kind"]))

        say("")
        say("submission_outcomes table, oldest first (the queryable audit trail):")
        for row in relational.query_outcomes(experiment_id="exp-42"):
            say("  experiment=%s status=%-9s rounds=%d at=%s"
                % (row["experiment_id"], row["status"], row["rounds"],
                   row["created_at"][:19]))

        say("")
        say("[FINAL] Both terminal decisions are rows in shared memory: the approved "
            "dry run and the denial each recorded with their round count, so the "
            "offline monitor and both agents read the same record instead of prose.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nwrote %s" % OUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
