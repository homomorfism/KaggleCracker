from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kagglecracker.config import Settings

SAMPLE_IDS = ["0013_01", "0018_01", "0019_01", "0021_01", "0023_01"]


@pytest.fixture
def sample_submission(tmp_path: Path) -> Path:
    """A miniature stand-in for the competition's sample_submission.csv.

    The validator derives every rule from this file, so tests never need the
    real competition data.
    """
    path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"PassengerId": SAMPLE_IDS, "Transported": [False] * 5}).to_csv(
        path, index=False
    )
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def write_metrics(workspace: Path):
    def _write(payload) -> Path:
        path = workspace / "metrics.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        return path

    return _write


@pytest.fixture
def write_submission(workspace: Path):
    def _write(df: pd.DataFrame, *, index: bool = False) -> Path:
        path = workspace / "submission.csv"
        df.to_csv(path, index=index)
        return path

    return _write


@pytest.fixture
def valid_submission_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"PassengerId": SAMPLE_IDS, "Transported": [True, False, True, False, True]}
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Every path points into tmp_path.

    context_dir in particular: it defaults to the project's real `context/`,
    and a test that writes an eda_findings.md there both pollutes the repo and
    leaks into the next test's assertions.
    """
    return Settings(
        competition="test-comp",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        context_dir=tmp_path / "context",
        db_path=tmp_path / "test.db",
        deepseek_api_key="test-key",
    )
