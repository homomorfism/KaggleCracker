"""The findings half of the recon SQLite store, driven through dispatch().

profile_dataset is the only writer: every profile run lands as rows in the
findings table, replace-per-check, so the table always holds the LATEST
profile of each dataset. Numbers asserted here are hand-computed from the
5-row fixture, same discipline as test_profile.py.

The workspace is redirected to tmp_path via the `store_ws` fixture, so the
default DB path (KC_WORKSPACE/experiments.db) lands in tmp and no test ever
touches the real workspace/ (the KC_WORKSPACE rule in AGENTS.md).
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from core.loop import RunState, dispatch
from core.registry import Registry
from tools.recon import store
from tools.recon.profile import register


@dataclass
class Call:
    name: str
    args: dict


def _mute(*a, **k):
    return None


# income: 2 of 5 missing (flagged). id: every row distinct (likely_id).
# city: constant 'riga' (flagged constant; in drift, test adds unseen 'vilnius').
# age and income each repeat a value so only id is all-distinct — otherwise
# every column of a 5-row fixture would trip the likely_id heuristic.
TRAIN = """id,age,income,city
1,34,52000,riga
2,29,N/A,riga
3,41,61000,riga
4,34,,riga
5,52,52000,riga
"""

# age means 38.0 vs 38.5 keep age under the SMD threshold on purpose, so the
# drift assertions below stay about city (unseen category) and income (shift).
TEST = """id,age,income,city
6,33,50000,vilnius
7,44,57000,riga
"""

TEST_MISSING_CITY = """id,age,income
6,33,50000
7,44,57000
"""


@pytest.fixture
def store_ws(tmp_path, monkeypatch):
    """Point KC_WORKSPACE at a fresh tmp dir and seed the CSVs."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("KC_DB", raising=False)
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "train.csv").write_text(TRAIN)
    (data / "test.csv").write_text(TEST)
    (data / "test_missing_city.csv").write_text(TEST_MISSING_CITY)
    return tmp_path


def _profile(args):
    return dispatch(
        Call("profile_dataset", args), register(Registry()), RunState(),
        input_fn=_mute, output_fn=_mute,
    )


THREE_CHECKS = {"path": "train.csv", "checks": ["missingness", "dtypes", "cardinality"]}


def test_a_profile_run_lands_as_findings_rows(store_ws):
    result = _profile(dict(THREE_CHECKS))
    assert result["ok"] is True
    # 4 columns x 3 per-column checks; the report says what the store received.
    assert result["data"]["findings_recorded"] == 12
    rows = store.query_findings("train.csv")
    assert len(rows) == 12
    # One row spot-checked end to end: the JSON value round-trips intact.
    income = [
        r for r in rows
        if r["check_name"] == "missingness" and r["column_name"] == "income"
    ]
    assert len(income) == 1
    assert income[0]["value"] == {"missing": 2, "pct": 0.4}
    assert income[0]["flagged"] is True
    assert income[0]["rows_profiled"] == 5


def test_flagged_rows_are_exactly_the_decisions_a_plan_must_make(store_ws):
    _profile(dict(THREE_CHECKS))
    flagged = store.query_findings("train.csv", flagged_only=True)
    assert {(r["check_name"], r["column_name"]) for r in flagged} == {
        ("missingness", "income"),   # 2 missing -> impute or encode
        ("cardinality", "id"),       # likely_id -> not a feature
        ("cardinality", "city"),     # constant -> carries no signal
    }


def test_reprofiling_replaces_rows_instead_of_duplicating_them(store_ws):
    _profile(dict(THREE_CHECKS))
    _profile(dict(THREE_CHECKS))
    assert len(store.query_findings("train.csv")) == 12


def test_drift_rows_flag_the_unseen_category(store_ws):
    result = _profile({
        "path": "train.csv",
        "checks": ["train_test_drift"],
        "compare_path": "test.csv",
    })
    assert result["ok"] is True
    rows = store.query_findings("train.csv", check_name="train_test_drift")
    city = [r for r in rows if r["column_name"] == "city"][0]
    assert city["flagged"] is True
    assert city["value"]["unseen_in_compare"] == ["vilnius"]
    # Both files agree on the column set, so the file-level row is not flagged.
    star = [r for r in rows if r["column_name"] == "*"][0]
    assert star["flagged"] is False


def test_a_column_set_mismatch_flags_the_file_level_row(store_ws):
    _profile({
        "path": "train.csv",
        "checks": ["train_test_drift"],
        "compare_path": "test_missing_city.csv",
    })
    rows = store.query_findings("train.csv", check_name="train_test_drift")
    star = [r for r in rows if r["column_name"] == "*"][0]
    assert star["flagged"] is True
    assert star["value"]["only_in_this_file"] == ["city"]


def test_kc_db_override_is_honored(store_ws, tmp_path, monkeypatch):
    override = tmp_path / "elsewhere" / "recon.db"
    monkeypatch.setenv("KC_DB", str(override))
    _profile(dict(THREE_CHECKS))
    assert override.exists()
    assert not (Path(str(store_ws)) / "experiments.db").exists()
