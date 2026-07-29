"""Scripted experiment agent for demo mode — no API key, real training.

The prose is canned; the numbers are not. run_experiment genuinely executes
the training script in the exec sandbox, and the score the demo records is
whatever the sandbox printed — this model reads it back out of the transcript,
it never invents one. Phase logic mirrors the live agent's brief:
draft -> propose plan -> (human approval) -> implement -> record -> finalize.
"""

import ast
import json
import re

from tests.fakemodel import Reply, ToolCall

_OK_RE = re.compile(r"^\[TOOL OK (?P<name>\S+)\] (?P<data>.*)$", re.DOTALL)

# Deliberately narrow, mirroring the spirit of core/gate.py: approving a plan
# should be an explicit act, not something a vague "hm, maybe" triggers.
_APPROVAL_WORDS = ("yes", "approve", "approved", "go ahead", "accept", "looks good", "run it")


def approves(text):
    t = (text or "").strip().lower()
    return t == "y" or any(w in t for w in _APPROVAL_WORDS)


def _last_ok(messages, tool):
    for m in reversed(messages):
        if not isinstance(m, str):
            continue
        match = _OK_RE.match(m)
        if match and match.group("name") == tool:
            try:
                return ast.literal_eval(match.group("data"))
            except (ValueError, SyntaxError):
                return None
    return None


def _fold_scores(result):
    """The per-fold scores the training script printed, out of stdout_tail."""
    tail = result.get("stdout_tail") or ""
    match = re.search(r"FOLD_SCORES: (\[[^\]]*\])", tail)
    if not match:
        return None
    try:
        scores = json.loads(match.group(1))
    except ValueError:
        return None
    return scores if isinstance(scores, list) and scores else None


_PLAN = """\
# Baseline: gradient boosting on the profiled features

## Data preprocessing
- drop identifier-like columns (unique per row) — the profile flags these
- coerce every feature to numeric; failures become missing (the model handles NaN natively)

## Model
- HistGradientBoostingClassifier, default parameters, fixed random_state

## Validation
- StratifiedKFold (shuffled), scored with balanced accuracy — the competition metric

## Expected outcome
- an honest, reproducible baseline score to beat; nothing tuned yet
"""


def _script(target):
    return """\
import json
import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv(os.environ.get("KC_DATASET_REF", "data/train.csv"))
target = %r
y = df[target].astype(str)
X = df.drop(columns=[target])
# Identifier-like columns (unique per row) carry no signal — drop them.
X = X.loc[:, [c for c in X.columns if X[c].nunique(dropna=True) < len(X)]]
X = X.apply(lambda s: pd.to_numeric(s, errors="coerce"))

folds = int(os.environ.get("KC_CV_FOLDS", "3"))
cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
# min_samples_leaf=5: the sklearn default of 20 cannot split small folds at
# all (it silently predicts the majority class); 5 behaves on small demo data
# and stays a sane, lightly-regularised baseline on big competition files.
scores = cross_val_score(
    HistGradientBoostingClassifier(random_state=42, min_samples_leaf=5), X, y, cv=cv,
    scoring="balanced_accuracy",
)
print("FOLD_SCORES:", json.dumps([round(float(s), 6) for s in scores]))
print("CV_SCORE: %%.6f" %% scores.mean())
""" % target


class DemoExperimentModel:
    """model(messages, tools) callable with staged, deterministic behaviour."""

    def __init__(self, meta, last_user_text, target, train_file):
        self._state = meta["state"]
        self._exp_id = meta["id"]
        self._prior_score = meta.get("cv_score")
        self._approved = approves(last_user_text)
        self._target = target
        self._train = train_file
        self._step = 0
        self._score = None
        self._folds = None

    def __call__(self, messages, tools):
        self._step += 1

        if not self._target:
            return Reply(text=(
                "(demo) No target column is set on this project — set one, then "
                "start a new experiment."
            ))
        if not self._train:
            return Reply(text="(demo) No csv files uploaded yet — add data first.")

        if self._state == "draft":
            if self._step == 1:
                return Reply(
                    text=(
                        "(demo agent) Proposing a baseline experiment grounded in "
                        "the dataset profile."
                    ),
                    tool_calls=[ToolCall(
                        "propose_experiment_plan",
                        {"name": "baseline gradient boosting", "plan_markdown": _PLAN},
                    )],
                )
            return Reply(text=(
                "Plan proposed — reply 'yes' to approve and I will run it. "
                "(demo mode: scripted agent, real sandbox training)"
            ))

        if self._state == "plan_proposed" and not self._approved:
            return Reply(text=(
                "(demo) Waiting for approval — reply 'yes' to run the plan. The "
                "demo agent cannot redesign the plan; that needs live mode."
            ))

        if self._state == "plan_proposed":
            if self._step == 1:
                return Reply(
                    text="(demo) Plan approved — training in the sandbox now.",
                    tool_calls=[ToolCall(
                        "run_experiment",
                        {
                            "code": _script(self._target),
                            "dataset_ref": self._train,
                            "cv_folds": 3,
                            "timeout_s": 300,
                        },
                    )],
                )
            if self._step == 2:
                result = _last_ok(messages, "run_experiment")
                if not result or "cv_score" not in result:
                    return Reply(text=(
                        "(demo) The training run failed — the journal above has "
                        "the sandbox error. Fix the data (running data prep "
                        "usually helps) and send 'yes' to retry."
                    ))
                self._score = float(result["cv_score"])
                self._folds = _fold_scores(result) or [self._score]
                return Reply(
                    text="(demo) CV score %.4f — filing it on the leaderboard." % self._score,
                    tool_calls=[ToolCall(
                        "record_experiment_result",
                        {
                            "experiment_id": str(result.get("experiment_id", self._exp_id)),
                            "cv_score": self._score,
                            "fold_scores": [float(s) for s in self._folds],
                            "notes": "demo baseline (scripted agent, real training)",
                        },
                    )],
                )
            if self._step == 3:
                return Reply(
                    text="",
                    tool_calls=[ToolCall(
                        "finalize_experiment",
                        {
                            "cv_score": self._score,
                            "summary": (
                                "Demo baseline: HistGradientBoosting, 3-fold "
                                "balanced accuracy %.4f." % self._score
                            ),
                        },
                    )],
                )
            return Reply(text=(
                "Baseline finished with balanced-accuracy CV %.4f. Beat it with a "
                "live-mode experiment or better features." % self._score
            ))

        score = " (CV %.4f)" % self._prior_score if isinstance(self._prior_score, float) else ""
        return Reply(text=(
            "(demo) This experiment is already finished%s — its score is on the "
            "leaderboard. Start a new experiment to try another idea." % score
        ))
