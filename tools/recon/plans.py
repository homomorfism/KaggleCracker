"""The recon slice's write pair: file a preprocessing plan, or replace one.

Same action, two specs, and that split IS the design: whether "write this
plan" is reversible depends on world state. Writing a NEW plan is undoable
(delete the file), so write_preprocessing_plan runs ungated. REPLACING an
existing plan destroys the version downstream experiments were built against,
so overwrite_preprocessing_plan is irreversible=True and stops for a human.
The frozen core reads irreversibility as a static ToolSpec flag, so the state
switch lives in the prechecks instead: each spec rejects the world state that
belongs to its sibling, and the rejection message names the right tool.
"""

from core.contracts import err, ok
from core.registry import ToolSpec
from tools.recon.paths import plan_path, plans_dir

# The dataset name doubles as the plan's filename, so it is constrained to
# characters that cannot climb out of workspace/plans or smuggle a path in.
_DATASET_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")

# How much of the doomed plan the human sees at the gate: enough to recognise
# what is being destroyed without scrolling a wall of text.
_PREVIEW_LINES = 10


def _name_precheck(args):
    """Guards shared by both specs. Returns an error envelope or None."""
    dataset = args["dataset"]
    if not dataset or not set(dataset) <= _DATASET_CHARS:
        return err(
            "bad_input",
            "dataset must be letters, digits, '-', '_' or '.', got %r" % dataset,
        )
    if set(dataset) <= {"."}:
        # "." and ".." pass the charset but name the directory or its parent.
        return err("bad_input", "dataset cannot be only dots: %r" % dataset)
    if not args["plan_markdown"].strip():
        return err(
            "bad_input",
            "plan_markdown is empty; profile the data first, then write what it needs",
        )
    return None


def _write_precheck(args):
    e = _name_precheck(args)
    if e is not None:
        return e
    if plan_path(args["dataset"]).exists():
        return err(
            "bad_input",
            "a plan for %r already exists; replacing it is overwrite_preprocessing_plan's "
            "job and needs human approval" % args["dataset"],
        )
    return None


def _overwrite_precheck(args):
    e = _name_precheck(args)
    if e is not None:
        return e
    if not plan_path(args["dataset"]).exists():
        # Nothing to destroy -> nothing to gate. Sending a human to approve a
        # no-op would be exactly the noise the precheck rule exists to prevent.
        return err(
            "not_found",
            "no plan for %r exists to overwrite; use write_preprocessing_plan" % args["dataset"],
        )
    return None


def _overwrite_preview(args):
    """What the human sees at the gate: the file, how much of it dies, and its
    opening lines. Runs only after the precheck passed, so the plan exists."""
    path = plan_path(args["dataset"])
    try:
        old = path.read_text()
    except OSError as e:
        # A race made the just-checked file unreadable. Show what we can rather
        # than crash the gate; the human can still decline.
        return "OVERWRITE plan for %r — WARNING: existing plan unreadable (%s)" % (
            args["dataset"], e,
        )
    old_lines = old.splitlines()
    shown = old_lines[:_PREVIEW_LINES]
    lines = [
        "OVERWRITE PREPROCESSING PLAN — the current plan is destroyed and cannot be recovered.",
        "",
        "plan file:    %s" % path,
        "existing:     %d lines, %d chars" % (len(old_lines), len(old)),
        "replacement:  %d chars" % len(args["plan_markdown"]),
        "",
        "first %d lines of the plan being destroyed:" % len(shown),
    ]
    lines += ["  " + l for l in shown]
    lines += ["", "Experiments derived from the old plan lose their reference."]
    return "\n".join(lines)


def _write_plan(args):
    """Shared body: both specs funnel here once their prechecks passed, so the
    only difference between them is the gate and which world state they accept."""
    path = plan_path(args["dataset"])
    try:
        plans_dir().mkdir(parents=True, exist_ok=True)
        path.write_text(args["plan_markdown"])
    except OSError as e:
        return err("exec_failed", "could not write %s: %s" % (path, e))
    return ok(path=str(path), dataset=args["dataset"], chars=len(args["plan_markdown"]))


_PARAMETERS = {
    "dataset": {"type": "str", "required": True},
    "plan_markdown": {"type": "str", "required": True},
}


WRITE_PLAN = ToolSpec(
    name="write_preprocessing_plan",
    description=(
        "Save a NEW preprocessing plan (markdown) for a dataset into "
        "workspace/plans. Call it once, after profile_dataset has shown what "
        "the data needs — a plan not grounded in a profile report is a guess. "
        "Do NOT call it to change an existing plan: it refuses if one exists; "
        "replacing a plan is overwrite_preprocessing_plan's job and needs "
        "human approval. Creating a new file is reversible (delete it), so "
        "this tool is ungated."
    ),
    parameters=_PARAMETERS,
    fn=_write_plan,
    precheck=_write_precheck,
    # Reversible half of the pair: no irreversible flag, no preview, no gate.
)


OVERWRITE_PLAN = ToolSpec(
    name="overwrite_preprocessing_plan",
    description=(
        "Replace the EXISTING preprocessing plan for a dataset. The old plan "
        "is destroyed and cannot be recovered, so this always stops for human "
        "approval. Call it only when new profiling evidence contradicts the "
        "current plan. Do NOT call it for a first plan — that is "
        "write_preprocessing_plan; this tool refuses if no plan exists. Do NOT "
        "call it for cosmetic edits; every overwrite costs a human a decision."
    ),
    parameters=_PARAMETERS,
    fn=_write_plan,
    irreversible=True,
    preview=_overwrite_preview,
    precheck=_overwrite_precheck,
)


def register(registry):
    """Register both plan tools into the given registry."""
    registry.register(WRITE_PLAN)
    registry.register(OVERWRITE_PLAN)
    return registry
