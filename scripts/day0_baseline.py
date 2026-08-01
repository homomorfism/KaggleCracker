"""Day-0 boundary walk: prove the whole external loop before any engine exists.

This is deliberately the dumbest model that isn't trivial. Its job is not to score
well — it is to walk every external dependency once, on the day there is still
time to react:

    download  ->  train  ->  submission.csv  ->  kaggle submit  ->  public score

It also produces the baseline number that success criterion 3 is measured against.

The rule: passengers in CryoSleep were transported. One feature, no libraries
beyond pandas, no sklearn on the host (all real ML runs in the sandbox).

    CryoSleep=True   -> transported 81.8% of the time
    CryoSleep=False  -> transported 32.9% of the time
    CryoSleep=NaN    -> transported 48.8% of the time  (predict False, the majority)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "runs" / "day0_baseline"


def predict(df: pd.DataFrame) -> pd.Series:
    return df["CryoSleep"].fillna(False).astype(bool)


def group_of(df: pd.DataFrame) -> pd.Series:
    """Travelling party: PassengerId is 'gggg_pp'."""
    return df["PassengerId"].str.split("_").str[0]


def grouped_holdout_accuracy(train: pd.DataFrame, seed: int = 42) -> float:
    """Score on a group-disjoint holdout, matching how Kaggle split train/test.

    A random row-level holdout would let the rule benefit from groupmates, which
    do not exist at test time. This is the same reasoning that put
    StratifiedGroupKFold in the protocol.
    """
    groups = group_of(train)
    rng = np.random.default_rng(seed)
    uniq = groups.unique()
    holdout = set(rng.choice(uniq, size=len(uniq) // 5, replace=False))
    mask = groups.isin(holdout)
    return float((predict(train[mask]) == train.loc[mask, "Transported"]).mean())


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    all_false = float((~train["Transported"]).mean())
    cryo = grouped_holdout_accuracy(train)
    print(f"sample_submission baseline (all False): {all_false:.4f}")
    print(f"CryoSleep rule, group-disjoint holdout: {cryo:.4f}")

    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Transported": predict(test)}
    )

    # The same checks SubmissionValidator will enforce on every generated node.
    # Getting them wrong is the cheapest way to waste a submission.
    assert list(submission.columns) == list(sample.columns), "column names differ"
    assert len(submission) == len(sample), "row count differs"
    assert set(submission["PassengerId"]) == set(sample["PassengerId"]), "id set differs"
    assert not submission["PassengerId"].duplicated().any(), "duplicate ids"
    assert submission["Transported"].notna().all(), "null predictions"
    assert submission["Transported"].dtype == bool, "Transported is not boolean"

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "submission.csv"
    submission.to_csv(path, index=False)  # index=False or Kaggle sees a third column
    print(f"wrote {path} ({len(submission)} rows)")
    print(submission.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
