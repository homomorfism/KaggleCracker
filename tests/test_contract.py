"""Every branch of the two validators.

These are the checks standing between an LLM-written script and a wasted
Kaggle submission, so each failure mode gets its own test rather than a
happy-path smoke test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kagglecracker.engine.contract import (
    ContractError,
    SubmissionValidator,
    parse_metrics,
    render_output_contract,
)

# --------------------------------------------------------------------------
# parse_metrics
# --------------------------------------------------------------------------


def test_valid_metrics(workspace, write_metrics):
    write_metrics({"cv_score": 0.81, "fold_scores": [0.80, 0.82, 0.81, 0.79, 0.83]})
    m = parse_metrics(workspace, expected_folds=5)
    assert m.cv_score == pytest.approx(0.81)
    assert m.fold_min == pytest.approx(0.79)
    assert m.fold_mean == pytest.approx(0.81)
    assert m.fold_std > 0


def test_metrics_extra_keys_are_kept(workspace, write_metrics):
    write_metrics({"cv_score": 0.5, "fold_scores": [0.5], "n_estimators": 400})
    assert parse_metrics(workspace).extra == {"n_estimators": 400}


def test_metrics_file_missing(workspace):
    with pytest.raises(ContractError, match="was not written"):
        parse_metrics(workspace)


def test_metrics_malformed_json(workspace, write_metrics):
    write_metrics("{not json")
    with pytest.raises(ContractError, match="not valid JSON"):
        parse_metrics(workspace)


def test_metrics_not_an_object(workspace, write_metrics):
    write_metrics([1, 2, 3])
    with pytest.raises(ContractError, match="must contain a JSON object"):
        parse_metrics(workspace)


def test_metrics_missing_cv_score(workspace, write_metrics):
    write_metrics({"fold_scores": [0.8]})
    with pytest.raises(ContractError, match="no 'cv_score' key"):
        parse_metrics(workspace)


@pytest.mark.parametrize("bad", ["0.81", None, True])
def test_metrics_cv_score_not_a_number(workspace, write_metrics, bad):
    write_metrics({"cv_score": bad, "fold_scores": [0.8]})
    with pytest.raises(ContractError, match="must be a number"):
        parse_metrics(workspace)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_metrics_cv_score_not_finite(workspace, write_metrics, bad):
    # json.loads accepts these non-standard literals, so they really can arrive.
    write_metrics(f'{{"cv_score": {bad}, "fold_scores": [0.8]}}')
    with pytest.raises(ContractError, match="must be finite"):
        parse_metrics(workspace)


def test_metrics_missing_fold_scores(workspace, write_metrics):
    write_metrics({"cv_score": 0.81})
    with pytest.raises(ContractError, match="no 'fold_scores' key"):
        parse_metrics(workspace)


@pytest.mark.parametrize("bad", [[], "0.8", {}])
def test_metrics_fold_scores_not_a_list(workspace, write_metrics, bad):
    write_metrics({"cv_score": 0.81, "fold_scores": bad})
    with pytest.raises(ContractError, match="non-empty list"):
        parse_metrics(workspace)


def test_metrics_fold_score_not_finite(workspace, write_metrics):
    write_metrics('{"cv_score": 0.81, "fold_scores": [0.8, NaN]}')
    with pytest.raises(ContractError, match=r"fold_scores\[1\]"):
        parse_metrics(workspace)


def test_metrics_fold_count_must_match_protocol(workspace, write_metrics):
    """A 3-fold answer to a 5-fold protocol is not comparable to the rest of the tree."""
    write_metrics({"cv_score": 0.81, "fold_scores": [0.8, 0.81, 0.82]})
    with pytest.raises(ContractError, match="did not follow the CV protocol"):
        parse_metrics(workspace, expected_folds=5)


# --------------------------------------------------------------------------
# SubmissionValidator
# --------------------------------------------------------------------------


def test_valid_submission(sample_submission, workspace, write_submission, valid_submission_df):
    write_submission(valid_submission_df)
    out = SubmissionValidator(sample_submission).validate(workspace)
    assert len(out) == 5


def test_submission_file_missing(sample_submission, workspace):
    with pytest.raises(ContractError, match="was not written"):
        SubmissionValidator(sample_submission).validate(workspace)


def test_submission_written_with_index_column(
    sample_submission, workspace, write_submission, valid_submission_df
):
    """`to_csv()` without index=False — the single most common way this fails."""
    write_submission(valid_submission_df, index=True)
    with pytest.raises(ContractError, match="index=False"):
        SubmissionValidator(sample_submission).validate(workspace)


def test_submission_wrong_column_names(
    sample_submission, workspace, write_submission, valid_submission_df
):
    write_submission(valid_submission_df.rename(columns={"Transported": "prediction"}))
    with pytest.raises(ContractError, match="columns must be exactly"):
        SubmissionValidator(sample_submission).validate(workspace)


def test_submission_column_order_matters(
    sample_submission, workspace, write_submission, valid_submission_df
):
    write_submission(valid_submission_df[["Transported", "PassengerId"]])
    with pytest.raises(ContractError, match="in that order"):
        SubmissionValidator(sample_submission).validate(workspace)


def test_submission_row_count_differs(
    sample_submission, workspace, write_submission, valid_submission_df
):
    write_submission(valid_submission_df.head(3))
    with pytest.raises(ContractError, match="has 3 rows, expected 5"):
        SubmissionValidator(sample_submission).validate(workspace)


def test_submission_duplicate_ids(sample_submission, workspace, write_submission):
    df = pd.DataFrame(
        {"PassengerId": ["0013_01"] * 5, "Transported": [True] * 5}
    )
    write_submission(df)
    with pytest.raises(ContractError, match="duplicate value"):
        SubmissionValidator(sample_submission).validate(workspace)


def test_submission_id_set_differs(
    sample_submission, workspace, write_submission, valid_submission_df
):
    df = valid_submission_df.copy()
    df.loc[0, "PassengerId"] = "9999_99"
    write_submission(df)
    with pytest.raises(ContractError, match="does not match sample_submission"):
        SubmissionValidator(sample_submission).validate(workspace)


def test_submission_null_prediction(
    sample_submission, workspace, write_submission, valid_submission_df
):
    df = valid_submission_df.astype({"Transported": "object"})
    df.loc[2, "Transported"] = None
    write_submission(df)
    with pytest.raises(ContractError, match="null value"):
        SubmissionValidator(sample_submission).validate(workspace)


def test_submission_infinite_prediction(sample_submission, workspace, write_submission):
    df = pd.DataFrame(
        {"PassengerId": ["0013_01", "0018_01", "0019_01", "0021_01", "0023_01"],
         "Transported": [0.1, 0.2, float("inf"), 0.4, 0.5]}
    )
    write_submission(df)
    with pytest.raises(ContractError, match="inf"):
        SubmissionValidator(sample_submission).validate(workspace)


def test_submission_unparseable(sample_submission, workspace):
    (workspace / "submission.csv").write_bytes(b"\x00\x01\x02 not,a,csv\n\"unclosed")
    with pytest.raises(ContractError):
        SubmissionValidator(sample_submission).validate(workspace)


# --------------------------------------------------------------------------
# the contract text itself
# --------------------------------------------------------------------------


def test_api_notes_are_off_by_default():
    """They cost context on every node, so they stay off until measured to help."""
    text = render_output_contract(
        protocol_description="p", package_versions="  - pandas 2.3.3", timeout_s=900
    )
    assert "gotchas" not in text


def test_api_notes_name_the_specific_failures_when_enabled():
    """Stating a version number is not enough — the model does not know what changed."""
    text = render_output_contract(
        protocol_description="p",
        package_versions="  - pandas 2.3.3",
        timeout_s=900,
        api_notes=True,
    )
    assert "gotchas" in text
    assert "early_stopping" in text  # the lightgbm failure from day 2
    assert "multiple values for keyword" in text  # the catboost failure from run 4
    assert "astype(\"category\")" in text  # the pandas failure from run 4


def test_contract_states_the_facts_the_agent_cannot_guess():
    """Protocol, packages and time budget are the three things it gets wrong if left to infer."""
    text = render_output_contract(
        protocol_description="accuracy, higher_is_better, 5-fold StratifiedGroupKFold, seed=42",
        package_versions="    - lightgbm 4.7.0",
        timeout_s=900,
    )
    assert "StratifiedGroupKFold" in text
    assert "lightgbm 4.7.0" in text
    assert "900s" in text
    assert "index=False" in text
    assert "NO network access" in text
    assert "/data" in text and "/workspace" in text


# --------------------------------------------------------------------------
# degenerate cross-validation — the leak guard
# --------------------------------------------------------------------------


def test_identical_fold_scores_are_rejected_as_a_leak(workspace, write_metrics):
    """Observed live: a node scored 1.0 on all five folds because the target
    column was never dropped from the feature matrix. It was accepted, ranked
    first, and would have been offered for submission."""
    write_metrics({"cv_score": 1.0, "fold_scores": [1.0] * 5})
    with pytest.raises(ContractError, match="zero variance"):
        parse_metrics(workspace, expected_folds=5)


def test_the_leak_message_names_the_likely_cause(workspace, write_metrics):
    """A debug prompt built from this has to point at the leak, not at a bug."""
    write_metrics({"cv_score": 1.0, "fold_scores": [1.0] * 5})
    with pytest.raises(ContractError) as exc:
        parse_metrics(workspace, expected_folds=5)
    assert "leaked" in str(exc.value)
    assert "label column is dropped" in str(exc.value)


def test_a_degenerate_score_short_of_perfect_is_still_caught(workspace, write_metrics):
    """The signal is zero variance, not a perfect score — a leak that happens to
    score 0.87 identically on every fold is the same bug."""
    write_metrics({"cv_score": 0.87, "fold_scores": [0.87] * 5})
    with pytest.raises(ContractError, match="zero variance"):
        parse_metrics(workspace, expected_folds=5)


def test_real_fold_variance_passes(workspace, write_metrics):
    """The tightest of 67 legitimate nodes had std 0.00381 — ~380x the bound."""
    write_metrics(
        {"cv_score": 0.82135, "fold_scores": [0.8102, 0.8338, 0.8286, 0.8222, 0.8118]}
    )
    assert parse_metrics(workspace, expected_folds=5).cv_score == pytest.approx(0.82135)


def test_a_single_fold_is_not_treated_as_degenerate(workspace, write_metrics):
    """One fold has no variance by definition; the protocol's fold count is the
    check for that, not this one."""
    write_metrics({"cv_score": 0.81, "fold_scores": [0.81]})
    assert parse_metrics(workspace).cv_score == pytest.approx(0.81)
