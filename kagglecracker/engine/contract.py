"""The output contract, and the two validators that enforce it.

This module is the single source of truth for what a node's `train.py` must
produce. Three consumers import from here and they must never drift apart:

    OUTPUT_CONTRACT ──► PromptBuilder   (tells the agent what to write)
                    ├──► parse_metrics  (checks metrics.json after the run)
                    └──► SubmissionValidator (checks submission.csv after the run)

If the wording of the contract changes, it changes in one place and all three
consumers follow. That is the whole reason this is a module constant rather
than three f-strings in three files.

Both validators raise `ContractError`. The caller turns that into a node with
status `contract_violation` and the error message as its `error_text`, so a bad
submission becomes a debuggable node the agent can fix — the same escape hatch
as a crash, rather than a silent scoring failure.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

METRICS_FILENAME = "metrics.json"
SUBMISSION_FILENAME = "submission.csv"

#: Below this, fold scores are treated as identical and the run is degenerate.
#:
#: Measured across 67 legitimate nodes from four runs, fold std ranged 0.00381 to
#: 0.01615 — the tightest real node is ~380x above this bound, so a false
#: positive is not a realistic concern. A leaked target produces exactly 0.0.
#:
#: Five folds of ~1,700 rows landing on identical scores does not happen to a
#: model that is actually learning; it happens to one that is reading the answer.
#: The check needs no metric, no direction, and no per-competition threshold,
#: which is why it is a constant rather than a protocol field.
DEGENERATE_FOLD_STD = 1e-9

# Mounted read-only into every container.
DATA_MOUNT = "/data"
WORKSPACE_MOUNT = "/workspace"


class ContractError(Exception):
    """A run completed but broke the output contract."""


@dataclass(frozen=True, slots=True)
class Metrics:
    cv_score: float
    fold_scores: list[float]
    extra: dict

    @property
    def fold_mean(self) -> float:
        return sum(self.fold_scores) / len(self.fold_scores)

    @property
    def fold_std(self) -> float:
        if len(self.fold_scores) < 2:
            return 0.0
        m = self.fold_mean
        return math.sqrt(sum((s - m) ** 2 for s in self.fold_scores) / (len(self.fold_scores) - 1))

    @property
    def fold_min(self) -> float:
        return min(self.fold_scores)


# Observed API drift, fed back into the prompt.
#
# A model writes idioms for the version it saw most in training, not the version
# installed — and stating the version number is not enough, because it does not
# know what changed. Every entry here is a failure this project actually hit, so
# the list grows from evidence rather than speculation. Keep it short: it is
# spent context on every single node.
LIBRARY_API_NOTES = """\
## Installed-version gotchas

These are real failures previous attempts hit against these exact versions. The
versions above are recent; APIs you remember may have changed.

  - lightgbm: early stopping and logging are callbacks imported from the package
    root — `from lightgbm import early_stopping, log_evaluation`. They are NOT
    bare names. `eval_set=` is deprecated in 4.x in favour of `eval_X=`/`eval_y=`.
  - catboost: pass every hyper-parameter as a keyword. Mixing positional and
    keyword arguments raises "got multiple values for keyword argument".
  - pandas: do not call `.astype("category")` on a column containing nulls, and
    do not assume a string column is object-dtype — check before reaching for
    `.cat` or `.str` accessors.
  - Prefer plain, stable APIs over recently-added convenience arguments. You
    cannot install anything, so a wrong guess costs the entire attempt."""


def render_output_contract(
    *,
    protocol_description: str,
    package_versions: str,
    timeout_s: int,
    api_notes: bool = False,
) -> str:
    """The contract block injected verbatim into every prompt.

    Kept as a function rather than a bare string because three facts differ per
    run — the CV protocol, the installed package versions, and the time budget —
    and every one of them is something the agent gets wrong if left to guess.
    """
    return f"""\
## Output contract (hard requirements)

Write a SINGLE self-contained Python file. It runs as `python train.py` with the
working directory at `{WORKSPACE_MOUNT}`.

INPUT — `{DATA_MOUNT}` is mounted READ-ONLY:
  {DATA_MOUNT}/train.csv
  {DATA_MOUNT}/test.csv
  {DATA_MOUNT}/sample_submission.csv

OUTPUT — write exactly these two files into `{WORKSPACE_MOUNT}`:

  1. `{METRICS_FILENAME}`
     {{"cv_score": <float>, "fold_scores": [<float>, ...]}}
     `cv_score` is the mean across folds under the protocol below. `fold_scores`
     has one entry per fold, in fold order. Both must be finite — no NaN, no inf.

  2. `{SUBMISSION_FILENAME}`
     Same columns, in the same order, as `{DATA_MOUNT}/sample_submission.csv`.
     One row per row of sample_submission.csv, same id set, no duplicates, no
     nulls. Write it with `index=False` — a stray index column fails validation.

CROSS-VALIDATION PROTOCOL (use exactly this; do not invent your own):
  {protocol_description}

ENVIRONMENT:
  - NO network access. `pip install` will fail. Import only from the list below.
  - NO writes outside `{WORKSPACE_MOUNT}`. `{DATA_MOUNT}` is read-only.
  - The process is killed at {timeout_s}s wall clock. Budget your work: a model
    that has not written both output files by then scores nothing at all.
  - Installed packages:
{package_versions}

Print whatever you find useful to stdout; it is captured and shown to you if the
run fails.{_api_notes_block(api_notes)}"""


def _api_notes_block(enabled: bool) -> str:
    return f"\n\n{LIBRARY_API_NOTES}" if enabled else ""


# --------------------------------------------------------------------------
# metrics.json
# --------------------------------------------------------------------------


def parse_metrics(workspace: Path, *, expected_folds: int | None = None) -> Metrics:
    path = workspace / METRICS_FILENAME
    if not path.is_file():
        raise ContractError(
            f"{METRICS_FILENAME} was not written. The script must write "
            f"{WORKSPACE_MOUNT}/{METRICS_FILENAME} before exiting."
        )

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ContractError(f"{METRICS_FILENAME} is not valid JSON: {exc}") from None

    if not isinstance(raw, dict):
        raise ContractError(
            f"{METRICS_FILENAME} must contain a JSON object, got {type(raw).__name__}"
        )

    if "cv_score" not in raw:
        raise ContractError(
            f"{METRICS_FILENAME} has no 'cv_score' key. Keys present: {sorted(raw)}"
        )

    cv_score = raw["cv_score"]
    if isinstance(cv_score, bool) or not isinstance(cv_score, (int, float)):
        raise ContractError(f"'cv_score' must be a number, got {type(cv_score).__name__}")
    cv_score = float(cv_score)
    if not math.isfinite(cv_score):
        raise ContractError(f"'cv_score' must be finite, got {cv_score}")

    folds_raw = raw.get("fold_scores")
    if folds_raw is None:
        raise ContractError(f"{METRICS_FILENAME} has no 'fold_scores' key")
    if not isinstance(folds_raw, list) or not folds_raw:
        raise ContractError("'fold_scores' must be a non-empty list of numbers")

    fold_scores: list[float] = []
    for i, s in enumerate(folds_raw):
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(float(s)):
            raise ContractError(f"'fold_scores[{i}]' must be a finite number, got {s!r}")
        fold_scores.append(float(s))

    if expected_folds is not None and len(fold_scores) != expected_folds:
        raise ContractError(
            f"'fold_scores' has {len(fold_scores)} entries but the protocol uses "
            f"{expected_folds} folds — the script did not follow the CV protocol."
        )

    extra = {k: v for k, v in raw.items() if k not in ("cv_score", "fold_scores")}
    metrics = Metrics(cv_score=cv_score, fold_scores=fold_scores, extra=extra)

    if len(fold_scores) > 1 and metrics.fold_std < DEGENERATE_FOLD_STD:
        raise ContractError(
            f"Every fold scored identically ({fold_scores[0]:.6f}), so cross-validation "
            f"has zero variance. Real models do not do this — across folds of this size "
            f"the score always moves. It almost always means the target leaked into the "
            f"features: check that the label column is dropped from X before fitting, and "
            f"that no feature is derived from the label (including group or aggregate "
            f"features computed over the full frame before splitting). A perfect or "
            f"near-perfect score here is a bug, not a result."
        )

    return metrics


# --------------------------------------------------------------------------
# submission.csv
# --------------------------------------------------------------------------


class SubmissionValidator:
    """Diffs a generated submission against the competition's own sample.

    Every rule here is derived from sample_submission.csv, so this is generic
    across competitions — there is nothing Spaceship-Titanic-specific in it.

    This is the cheapest failure to prevent and the most expensive to hit: an
    LLM-written script can produce a perfectly good CV score and a submission
    Kaggle rejects, and you only find out when you spend one of the day's
    submissions on it.
    """

    def __init__(self, sample_submission_path: Path) -> None:
        self.sample = pd.read_csv(sample_submission_path)
        self.id_column = self.sample.columns[0]
        self.prediction_columns = list(self.sample.columns[1:])
        self._sample_ids = set(self.sample[self.id_column])

    def validate(self, workspace: Path) -> pd.DataFrame:
        path = workspace / SUBMISSION_FILENAME
        if not path.is_file():
            raise ContractError(
                f"{SUBMISSION_FILENAME} was not written. The script must write "
                f"{WORKSPACE_MOUNT}/{SUBMISSION_FILENAME} before exiting."
            )
        try:
            sub = pd.read_csv(path)
        except Exception as exc:
            raise ContractError(
                f"{SUBMISSION_FILENAME} could not be parsed as CSV: {exc}"
            ) from None
        return self.validate_frame(sub)

    def validate_frame(self, sub: pd.DataFrame) -> pd.DataFrame:
        expected = list(self.sample.columns)
        actual = list(sub.columns)

        if actual != expected:
            # The single most common way this fails: `to_csv()` without
            # index=False, which prepends an unnamed column. Name it explicitly
            # so the debug prompt gets a fix rather than a puzzle.
            stray = [c for c in actual if str(c).startswith("Unnamed:")]
            if stray:
                raise ContractError(
                    f"{SUBMISSION_FILENAME} has an index column {stray} — it was written "
                    f"without index=False. Use `df.to_csv(path, index=False)`. "
                    f"Expected columns {expected}, got {actual}."
                )
            raise ContractError(
                f"{SUBMISSION_FILENAME} columns must be exactly {expected} in that order, "
                f"got {actual}."
            )

        if len(sub) != len(self.sample):
            raise ContractError(
                f"{SUBMISSION_FILENAME} has {len(sub)} rows, expected {len(self.sample)} "
                f"(one per row of sample_submission.csv)."
            )

        ids = sub[self.id_column]
        if ids.isna().any():
            raise ContractError(f"'{self.id_column}' contains null values.")

        dupes = ids[ids.duplicated()].unique()
        if len(dupes):
            raise ContractError(
                f"'{self.id_column}' has {len(dupes)} duplicate value(s), "
                f"e.g. {list(dupes[:5])}."
            )

        actual_ids = set(ids)
        missing = self._sample_ids - actual_ids
        extra = actual_ids - self._sample_ids
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"{len(missing)} missing (e.g. {sorted(missing)[:3]})")
            if extra:
                parts.append(f"{len(extra)} unexpected (e.g. {sorted(extra)[:3]})")
            raise ContractError(
                f"'{self.id_column}' does not match sample_submission.csv: {', '.join(parts)}."
            )

        for col in self.prediction_columns:
            if sub[col].isna().any():
                n = int(sub[col].isna().sum())
                raise ContractError(f"Prediction column '{col}' has {n} null value(s).")
            if pd.api.types.is_numeric_dtype(sub[col]):
                finite = pd.to_numeric(sub[col], errors="coerce")
                if not finite.map(lambda v: math.isfinite(v)).all():
                    raise ContractError(f"Prediction column '{col}' contains inf or -inf.")

        return sub
