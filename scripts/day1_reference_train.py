"""Reference solution — a hand-written node, used to prove the pipeline before any LLM exists.

This is exactly what the agent will be asked to produce: one self-contained file
that reads /data, honours the CV protocol, and writes metrics.json and
submission.csv into /workspace. If this runs green through LocalDockerExecutor
and passes both validators, the sandbox contract works and Day 2 can point an
LLM at it.

It is also the honest baseline for what "the search has to beat" means: a
competent-but-unremarkable LightGBM, no feature engineering worth the name.
"""

import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold

DATA = "/data"
OUT = "/workspace"
N_FOLDS = 5
SEED = 42


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # Cabin is "deck/num/side" — three separate signals crammed into one string.
    cabin = X["Cabin"].str.split("/", expand=True)
    X["Deck"] = cabin[0]
    X["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    X["Side"] = cabin[2]

    # Party size: passengers travelling together share an id prefix.
    X["GroupSize"] = X.groupby("Group")["Group"].transform("size")

    spend = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    X["TotalSpend"] = X[spend].fillna(0).sum(axis=1)
    X["NoSpend"] = (X["TotalSpend"] == 0).astype(int)

    for col in ("HomePlanet", "Destination", "Deck", "Side"):
        X[col] = X[col].astype("category")
    for col in ("CryoSleep", "VIP"):
        # Keep the missing value distinguishable from False — nulls here are
        # informative, and collapsing them to False throws that away.
        X[col] = X[col].map({True: 1, False: 0}).astype("float64")

    drop = ["PassengerId", "Name", "Cabin", "Group", "Transported"]
    return X.drop(columns=[c for c in drop if c in X.columns])


def main() -> None:
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    sample = pd.read_csv(f"{DATA}/sample_submission.csv")

    for df in (train, test):
        df["Group"] = df["PassengerId"].str.split("_").str[0]

    y = train["Transported"].astype(int).to_numpy()
    groups = train["Group"].to_numpy()

    # Featurize train and test together so categorical levels line up.
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    features = featurize(combined)
    X = features.iloc[: len(train)].reset_index(drop=True)
    X_test = features.iloc[len(train) :].reset_index(drop=True)

    # Grouped folds are not optional here: Kaggle split train/test by party, so
    # a fold that splits a party lets the model read a groupmate's label that
    # will not exist at test time.
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_scores: list[float] = []
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=SEED,
            verbose=-1,
        )
        model.fit(X.iloc[tr_idx], y[tr_idx])
        acc = float((model.predict(X.iloc[va_idx]) == y[va_idx]).mean())
        fold_scores.append(acc)
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"fold {fold}: accuracy {acc:.5f}", flush=True)

    cv_score = float(np.mean(fold_scores))
    print(f"cv_score {cv_score:.5f}  std {np.std(fold_scores):.5f}")

    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump({"cv_score": cv_score, "fold_scores": fold_scores}, f)

    submission = pd.DataFrame(
        {
            sample.columns[0]: test["PassengerId"],
            sample.columns[1]: (test_pred > 0.5),
        }
    )
    submission.to_csv(f"{OUT}/submission.csv", index=False)
    print(f"wrote submission.csv ({len(submission)} rows)")


if __name__ == "__main__":
    main()
