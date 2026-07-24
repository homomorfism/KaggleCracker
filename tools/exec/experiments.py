"""The two exec-slice tools: run an experiment, then record its result.

Both return the standard envelope from core.contracts. run_experiment is the
observe/act workhorse; record_experiment_result is a deliberately reversible,
ungated write. The contrast between them is the point — one produces a score by
executing code, the other only files away a number that already exists.
"""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from core.contracts import err, ok
from core.registry import ToolSpec


# Metric direction is not yet confirmed — pending the S6E7 Day-0 checklist. Until
# then we assume higher-is-better; if the confirmed metric is lower-is-better,
# this is the single line to flip and every best-score comparison follows it.
HIGHER_IS_BETTER = True

# You cannot score more folds than you ran. cv_folds is capped at this value in
# the run_experiment schema, so both places read one constant and stay in step.
_MAX_CV_FOLDS = 20

# The only parent environment variables an experiment subprocess inherits. PATH
# so the interpreter can find its tools, HOME because several libraries write
# caches under it, LANG for text encoding. Everything else — credentials, tokens,
# PYTHONPATH — is deliberately left behind (see the env build in run_experiment).
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG")


def _workspace_root():
    # Tests point KC_WORKSPACE at a tmp dir (see tests/conftest.py); real runs
    # use ./workspace. Reading it here means no test ever writes into the real
    # workspace/.
    return Path(os.environ.get("KC_WORKSPACE", "workspace"))


def _tail(text, limit=800):
    """Keep the end of captured output — that is where a traceback or the score
    line lives — without letting an unbounded log into the transcript."""
    text = (text or "").strip()
    return text[-limit:]


def _parse_cv_score(stdout):
    """Pull the required 'CV_SCORE: <float>' line out of stdout.

    Returns (score, None) or (None, error_envelope). Both failure branches are
    bad_input/retryable=True on purpose: the process ran clean and only broke the
    output contract, so the model's fix is to print the line (or print it as a
    real float) and try again — not a crash to debug.
    """
    marker = "CV_SCORE:"
    # Scan from the bottom: the final printed score is the authoritative one.
    for line in reversed(stdout.splitlines()):
        s = line.strip()
        if s.startswith(marker):
            raw = s[len(marker):].strip()
            try:
                return float(raw), None
            except ValueError:
                # #7: the line is there but the value is not a float.
                return None, err(
                    "bad_input",
                    "CV_SCORE printed but %r is not a float; print exactly 'CV_SCORE: <number>'" % raw,
                    retryable=True,
                )
    # #6: it exited 0 and simply never printed the score.
    return None, err(
        "bad_input",
        "no 'CV_SCORE: <float>' line in stdout; add a print of the final CV score",
        retryable=True,
    )


def _run_precheck(args):
    """Machine-checkable guard that runs before the body: keep dataset_ref inside
    workspace/data. A string comparison rejects a traversal attempt so the
    subprocess never sees a path pointing out of the sandbox. Returns an error
    envelope or None."""
    dataset_ref = args["dataset_ref"]
    if not dataset_ref.strip():
        return err("bad_input", "dataset_ref is empty")

    data_root = (_workspace_root() / "data").resolve()
    # resolve() collapses '..' and follows symlinks, and joining an absolute path
    # discards data_root entirely — both then land outside data_root and are
    # rejected here rather than in the body.
    candidate = (_workspace_root() / "data" / dataset_ref).resolve()
    if candidate != data_root and data_root not in candidate.parents:
        return err("bad_input", "dataset_ref escapes workspace/data: %r" % dataset_ref)
    return None


def run_experiment(args):
    code = args["code"]
    dataset_ref = args["dataset_ref"]
    timeout_s = args["timeout_s"]
    cv_folds = args["cv_folds"]

    ws = _workspace_root()
    data_path = ws / "data" / dataset_ref

    # #3: the ref is in-bounds (precheck proved that) but nothing is there.
    # Distinct from the traversal case above, which is a malformed ref, not a
    # missing file.
    if not data_path.exists():
        return err("not_found", "dataset not found under workspace/data: %r" % dataset_ref)

    exp_id = uuid.uuid4().hex[:12]
    exp_dir = ws / "experiments"
    script = exp_dir / (exp_id + ".py")
    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
        script.write_text(code)
    except OSError as e:
        # #8: could not even stage the script (disk full, permissions).
        return err("exec_failed", "could not write experiment script: %s" % e)

    # Run confined to workspace/ so the script addresses its data relatively and
    # cannot reach above the sandbox. The dataset and fold count are handed over
    # via the environment so the script does not have to hardcode them.
    #
    # The environment is built from an allowlist rather than copied from
    # os.environ: this subprocess runs model-written code, and inheriting our
    # whole environment would hand it every ambient secret this process happens to
    # hold — Kaggle credentials above all — which it could then print straight
    # into stdout and from there into the transcript. Only what a Python process
    # needs to start, plus our own KC_* sandbox settings, crosses the boundary.
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    env.update({k: v for k, v in os.environ.items() if k.startswith("KC_")})
    env["KC_DATASET_REF"] = str(Path("data") / dataset_ref)
    env["KC_CV_FOLDS"] = str(cv_folds)
    try:
        proc = subprocess.run(
            [sys.executable, str(script.relative_to(ws))],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=timeout_s,  # the cap lives in the schema; the body trusts the value
            env=env,
        )
    except subprocess.TimeoutExpired:
        # #4: wall-clock cap hit. retryable=False: the identical call will time
        # out again, so the message tells the model the two real fixes instead.
        return err(
            "timeout",
            "experiment exceeded %ss; raise timeout_s or reduce the work" % timeout_s,
            retryable=False,
        )

    # #5: the experiment itself crashed. This is a genuine execution failure,
    # kept separate from the clean-run/bad-output cases below.
    if proc.returncode != 0:
        return err(
            "exec_failed",
            "experiment exited %d: %s" % (proc.returncode, _tail(proc.stderr)),
        )

    # #6 / #7: exited 0 but broke the output contract -> bad_input, above.
    score, score_err = _parse_cv_score(proc.stdout)
    if score_err is not None:
        return score_err

    return ok(
        experiment_id=exp_id,
        cv_score=score,
        cv_folds=cv_folds,
        stdout_tail=_tail(proc.stdout),
    )


def _record_precheck(args):
    """Machine-checkable guard before the ungated write: you cannot have more
    fold scores than the maximum number of folds an experiment can run. A string
    comparison rejects an impossible list here rather than filing it into the
    leaderboard. Returns an error envelope or None."""
    fold_scores = args["fold_scores"]
    if len(fold_scores) > _MAX_CV_FOLDS:
        return err(
            "bad_input",
            "fold_scores has %d entries but at most %d folds can be run"
            % (len(fold_scores), _MAX_CV_FOLDS),
        )
    return None


def record_experiment_result(args):
    ws = _workspace_root()
    exp_dir = ws / "experiments"
    leaderboard = exp_dir / "leaderboard.jsonl"

    row = {
        "experiment_id": args["experiment_id"],
        "cv_score": args["cv_score"],
        "fold_scores": args["fold_scores"],
        "notes": args["notes"],
    }
    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
        with leaderboard.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        return err("exec_failed", "could not append to leaderboard: %s" % e)

    # Recompute count and best from the file rather than trusting an in-memory
    # tally, so the answer is correct even across separate runs of the agent.
    count = 0
    best = None
    try:
        with leaderboard.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                count += 1
                s = rec.get("cv_score")
                # Direction comes from HIGHER_IS_BETTER (see top of file), not a
                # hardcoded '>'; flipping that one constant flips this comparison.
                if isinstance(s, (int, float)) and (
                    best is None
                    or (s > best if HIGHER_IS_BETTER else s < best)
                ):
                    best = s
    except OSError as e:
        return err("exec_failed", "could not read leaderboard: %s" % e)

    return ok(count=count, best_score=best)


RUN_EXPERIMENT = ToolSpec(
    name="run_experiment",
    description=(
        "Run a self-contained Python training script in the sandbox and return "
        "its cross-validated score. Call this to test one concrete modelling idea "
        "whose code is ready to run; the script must print exactly one line "
        "'CV_SCORE: <float>' to stdout. Do NOT call it to inspect data, list "
        "files, or read docs — it executes code, it does not explore. Do NOT call "
        "it to re-run an experiment you have already scored; read the leaderboard "
        "instead. Do NOT use it to submit to Kaggle — it never touches the "
        "competition."
    ),
    parameters={
        "code": {"type": "str", "required": True},
        "dataset_ref": {"type": "str", "required": True},
        # Capped here in the schema (not in the body) so an over-long budget is a
        # bad_input rejection before anything runs.
        "timeout_s": {"type": "int", "min": 1, "max": 1800, "default": 600},
        "cv_folds": {"type": "int", "min": 2, "max": _MAX_CV_FOLDS, "default": 5},
    },
    fn=run_experiment,
    precheck=_run_precheck,
    # Reversible: a training run writes a throwaway script and computes a number.
    # Nothing external happens, so it is ungated.
)


RECORD_RESULT = ToolSpec(
    name="record_experiment_result",
    description=(
        "Append one already-computed experiment result to the local leaderboard "
        "and return the running count and best score. Call this once, after "
        "run_experiment has returned a CV score you want to keep. Do NOT call it "
        "to compute or estimate a score — it only files a number you already "
        "have. Do NOT call it before an experiment has actually run, and do NOT "
        "call it twice for the same experiment_id. It writes a local file only; "
        "it is reversible and submits nothing."
    ),
    parameters={
        "experiment_id": {"type": "str", "required": True},
        # No bound: some competition metrics are legitimately negative (e.g. a
        # signed or log-loss-style score), so any min risks rejecting a real
        # value. Impossible fold counts are caught in _record_precheck instead.
        "cv_score": {"type": "float", "required": True},
        "fold_scores": {"type": "list", "required": True},
        "notes": {"type": "str", "default": ""},
    },
    fn=record_experiment_result,
    precheck=_record_precheck,
    # No irreversible flag and no preview: this is the reversible half of the
    # pair, so it stays ungated by design.
)


def register(registry):
    """Register both exec-slice tools into the given registry."""
    registry.register(RUN_EXPERIMENT)
    registry.register(RECORD_RESULT)
    return registry
