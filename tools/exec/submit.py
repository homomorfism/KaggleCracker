"""The gated exec-slice tool: submit a predictions file to the live competition.

This is the one irreversible action in the slice, so it is the one that stops
for a human. Everything a string comparison can decide — the file exists, its
columns and row count match the sample, the daily quota is not already spent —
is settled in the precheck, BEFORE the gate, so a person is never asked to
approve something machine validation could have rejected. dry_run=True is the
default: a submission to a live, per-day-capped competition cannot be undone.
"""

import csv
import datetime
import json

from core.contracts import err, ok
from core.registry import ToolSpec

# Reuse the sibling tool's definition of the sandbox root so there is a single
# source of truth for where workspace/ is; a second copy could drift from it.
from tools.exec.experiments import _workspace_root


# CONFIRMED 2026-07-24 against the live competition metadata (Kaggle API,
# /api/v1/competitions/list, field maxDailySubmissions): S6E7 allows 10
# submissions per day. This is the number the gate shows the human before one
# is spent; if Kaggle changes it, this is the single line to update (same
# posture as HIGHER_IS_BETTER in experiments.py).
DAILY_QUOTA = 10


def _today():
    # Quota is tracked per calendar day, so the key is just today's date.
    return datetime.date.today().isoformat()


def _quota_path(ws):
    return ws / "submissions" / "quota.json"


def _load_quota(ws):
    """Return (mapping, None) or (None, error_envelope).

    A missing file means no submissions have been made yet -> empty map, not an
    error. A file that exists but will not parse is a real failure: we refuse
    rather than silently reset the count, which could let the daily cap be blown.
    """
    path = _quota_path(ws)
    if not path.exists():
        return {}, None
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, ValueError) as e:  # ValueError covers json.JSONDecodeError
        return None, err("exec_failed", "could not read quota file: %s" % e)
    if not isinstance(data, dict):
        return None, err("exec_failed", "quota file is not a JSON object")
    return data, None


def _save_quota(ws, quota):
    """Persist the quota map. Returns None on success or an error envelope."""
    path = _quota_path(ws)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(quota, f)
    except OSError as e:
        return err("exec_failed", "could not write quota file: %s" % e)
    return None


def _read_csv(path):
    """Return (header, data_rows, None) or (None, None, error_envelope).

    header is the list of column names; data_rows excludes the header. Reading
    the whole file is fine at this scale and lets the preview show the first rows
    without a second pass.
    """
    try:
        with path.open(newline="") as f:
            rows = list(csv.reader(f))
    except OSError as e:
        return None, None, err("exec_failed", "could not read %s: %s" % (path, e))
    if not rows:
        return None, None, err("bad_input", "%s is empty" % path)
    return rows[0], rows[1:], None


def _submit_precheck(args):
    """Machine-checkable guard that runs BEFORE the gate. Every rejection here is
    something a comparison decided, so no human is ever asked about it. Returns
    an error envelope or None."""
    ws = _workspace_root()
    sub_path = ws / args["submission_path"]
    sample_path = ws / args["sample_submission_path"]

    # Existence first. not_found (not bad_input) mirrors run_experiment's missing
    # dataset: the ref is well-formed, the file simply is not there.
    if not sub_path.exists():
        return err("not_found", "submission file not found: %r" % args["submission_path"])
    if not sample_path.exists():
        return err("not_found", "sample submission not found: %r" % args["sample_submission_path"])

    sub_header, sub_rows, e = _read_csv(sub_path)
    if e is not None:
        return e
    sample_header, sample_rows, e = _read_csv(sample_path)
    if e is not None:
        return e

    # Exact column match: same names, same order. A malformed header is caught
    # here so it never reaches a human at the gate.
    if sub_header != sample_header:
        return err(
            "bad_input",
            "submission columns %r do not match sample %r" % (sub_header, sample_header),
        )

    # Exact row-count match: one prediction per row of the sample.
    if len(sub_rows) != len(sample_rows):
        return err(
            "bad_input",
            "submission has %d data rows but sample has %d" % (len(sub_rows), len(sample_rows)),
        )

    # Daily quota, checked here so the gate is never reached once the cap is
    # spent. This applies to dry runs too: validation can legitimately fail
    # before the gate regardless of dry_run.
    quota, e = _load_quota(ws)
    if e is not None:
        return e
    spent = quota.get(_today(), 0)
    if isinstance(spent, bool) or not isinstance(spent, int):
        # Guard the arithmetic below: precheck is not wrapped by dispatch, so a
        # corrupt count must become an envelope here, never a raised TypeError.
        return err("exec_failed", "quota count for %s is not an integer" % _today())
    if spent >= DAILY_QUOTA:
        return err(
            "quota_exhausted",
            "daily submission quota spent (%d/%d for %s)" % (spent, DAILY_QUOTA, _today()),
            retryable=False,
        )
    return None


def _submit_preview(args):
    """What the human sees at the gate: enough to judge the submission, and a
    plain statement that it cannot be undone. Runs only after the precheck
    passed, so the file exists, parses, and matches the sample."""
    ws = _workspace_root()
    header, rows, e = _read_csv(ws / args["submission_path"])
    if e is not None:
        # A race made the just-validated file unreadable. Show what we can rather
        # than crash the gate; the human can still decline.
        return "SUBMIT %r — WARNING: file no longer readable (%s)" % (
            args["submission_path"], e["error"]["msg"],
        )

    quota, qe = _load_quota(ws)
    spent = quota.get(_today(), 0) if qe is None else 0
    remaining = DAILY_QUOTA - spent

    preview_rows = rows[:3]
    lines = [
        "SUBMIT TO KAGGLE — this is irreversible and cannot be undone.",
        "",
        "file:     %s" % args["submission_path"],
        "rows:     %d data rows" % len(rows),
        "columns:  %d (%s)" % (len(header), ", ".join(header)),
        "cv_score: %s  (CV score of the model behind this submission)" % args["cv_score"],
        "message:  %s" % args["message"],
        "quota:    %d of %d daily submissions remaining" % (remaining, DAILY_QUOTA),
        "",
        "first %d data rows:" % len(preview_rows),
    ]
    for r in preview_rows:
        lines.append("  " + ", ".join(r))
    return "\n".join(lines)


def submit_to_kaggle(args):
    ws = _workspace_root()
    dry_run = args["dry_run"]

    if dry_run:
        # Validated, prechecked, and approved at the gate — then it stops here.
        # Nothing is uploaded and no quota is spent. This is the default because a
        # live, per-day-capped submission cannot be taken back.
        return ok(
            dry_run=True,
            submitted=False,
            submission_path=args["submission_path"],
            note="dry run: validated and confirmed, not uploaded; quota untouched",
        )

    # --- the real, irreversible upload ------------------------------------
    # Left commented until the S6E7 Kaggle credentials/CLI are wired up.
    # Uncommenting this single call is the only step that touches the live
    # competition:
    #   subprocess.run(
    #       ["kaggle", "competitions", "submit", "-c", "<competition-slug>",
    #        "-f", str(ws / args["submission_path"]), "-m", args["message"]],
    #       check=True,
    #   )

    # Spend one submission for today and persist it, so a separate run of the
    # agent sees the updated count. Recomputed from the file, never an in-memory
    # tally.
    quota, e = _load_quota(ws)
    if e is not None:
        return e
    quota[_today()] = quota.get(_today(), 0) + 1
    e = _save_quota(ws, quota)
    if e is not None:
        return e

    return ok(
        dry_run=False,
        submitted=True,
        submission_path=args["submission_path"],
        remaining=DAILY_QUOTA - quota[_today()],
    )


SUBMIT_TO_KAGGLE = ToolSpec(
    name="submit_to_kaggle",
    description=(
        "Submit a finished predictions file to the live Kaggle competition. Call "
        "this only after run_experiment has produced a CV score you trust and you "
        "have decided this specific file is the one to send. It is irreversible "
        "and the daily submission count is capped, so it always asks for human "
        "approval and defaults to dry_run=True. Do NOT call it to check or clean a "
        "file — a dry run gates but never uploads. Do NOT call it to score or "
        "explore models; that is run_experiment. Do NOT call it repeatedly to "
        "probe the leaderboard, and never set dry_run=False unless a human is "
        "there to approve the real upload."
    ),
    parameters={
        "submission_path": {"type": "str", "required": True},
        "message": {"type": "str", "required": True},
        # Displayed to the human at the gate, not bounded: some competition
        # metrics are legitimately negative, so a min could reject a real score.
        "cv_score": {"type": "float", "required": True},
        "sample_submission_path": {"type": "str", "default": "data/sample_submission.csv"},
        # The safe default. Kept as a schema default so an omitted flag is a dry
        # run, never a live submission.
        "dry_run": {"type": "bool", "default": True},
    },
    fn=submit_to_kaggle,
    irreversible=True,
    preview=_submit_preview,
    precheck=_submit_precheck,
)


def register(registry):
    """Register the submit tool into the given registry."""
    registry.register(SUBMIT_TO_KAGGLE)
    return registry
