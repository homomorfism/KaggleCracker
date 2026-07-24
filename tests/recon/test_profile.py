"""profile_dataset, all driven through the real dispatch() order.

The tests pin the slice's central distinction: dirty VALUES come back as ok
findings (a mixed column is exactly what a preprocessing plan needs to know),
while broken STRUCTURE — ragged rows, empty files, duplicate headers — comes
back as an error envelope, because no statistic computed from a non-rectangle
can be trusted. Happy-path numbers are hand-computed from tiny fixtures.

The workspace is redirected to tmp_path via the `workspace` fixture, so no
test reads or writes the real workspace/ (the KC_WORKSPACE rule in AGENTS.md).
"""

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.loop import RunState, dispatch
from core.registry import Registry
from tools.recon import profile as profile_mod
from tools.recon.profile import PROFILE_DATASET, register

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

CLEAN = (
    "id,age,income,city,risk\n"
    "1,20,1000,A,0\n"
    "2,30,2000,B,1\n"
    "3,40,3000,A,0\n"
    "4,50,4000,B,1\n"
    "5,60,,A,0\n"
    "6,70,6000,B,1\n"
    "7,80,7000,A,0\n"
    "8,,8000,B,1\n"
    "9,25,9000,,0\n"
    "10,35,NA,A,1\n"
)

# y is exactly 2x, so r(x, y) must be exactly 1; noise is constant, so its
# correlation is undefined and it must be absent from the report.
CORR = "x,noise,y\n1,5,2\n2,5,4\n3,5,6\n4,5,8\n"

BIN = "f,label\n1,no\n2,no\n3,yes\n4,yes\n"
MULTI = "f,label\n1,a\n2,b\n3,c\n4,a\n"

# Drift pair: 'a' shifts its mean by 100, 'b' is the same constant on both
# sides (exercises the std=0 fallthrough), 'cat' gains an unseen category 'z',
# and 'risk' exists only in the train file.
TRAIN_D = "a,b,cat,risk\n1,10,x,0\n2,10,x,1\n3,10,y,0\n4,10,y,1\n"
TEST_D = "a,b,cat\n101,10,x\n102,10,y\n103,10,z\n104,10,z\n"


@dataclass
class Call:
    name: str
    args: dict


def _mute(*a, **k):
    return None


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point KC_WORKSPACE at a fresh tmp dir and seed the fixture CSVs under
    data/. Returns the workspace root."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "train.csv").write_text(CLEAN)
    (data / "corr.csv").write_text(CORR)
    (data / "bin.csv").write_text(BIN)
    (data / "multi.csv").write_text(MULTI)
    (data / "train_d.csv").write_text(TRAIN_D)
    (data / "test_d.csv").write_text(TEST_D)
    shutil.copy(FIXTURES / "ragged.csv", data / "ragged.csv")
    return tmp_path


def _profile(args):
    reg = Registry()
    reg.register(PROFILE_DATASET)
    return dispatch(Call("profile_dataset", args), reg, RunState(),
                    input_fn=_mute, output_fn=_mute)


def _forbid_read(monkeypatch):
    """Make any attempt to read a table fail the test loudly: used to prove a
    rejection happened before the body ran."""
    def boom(*a, **k):
        raise AssertionError("the tool body must not run")
    monkeypatch.setattr(profile_mod, "_read_table", boom)


# --- registration / shape ----------------------------------------------------


def test_profile_registers_under_its_action_name():
    assert register(Registry()).get("profile_dataset") is PROFILE_DATASET


def test_profile_is_reversible_and_ungated():
    # Read-only tool: no irreversible flag, no preview, never a gate. The
    # contrast with overwrite_preprocessing_plan is what requirement 3 grades.
    assert PROFILE_DATASET.irreversible is False
    assert PROFILE_DATASET.preview is None
    assert PROFILE_DATASET.precheck is not None


# --- schema rejection: a bad arg stops before the body -----------------------


def test_off_enum_check_is_rejected_before_any_file_is_read(workspace, monkeypatch):
    _forbid_read(monkeypatch)
    result = _profile({"path": "train.csv", "checks": ["correlation"]})  # not in the enum
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_oversized_sample_rows_is_rejected_before_any_file_is_read(workspace, monkeypatch):
    _forbid_read(monkeypatch)
    result = _profile({"path": "train.csv", "checks": ["dtypes"], "sample_rows": 10**9})
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


# --- precheck: still before the body ------------------------------------------


def test_path_traversal_is_rejected_before_any_file_is_read(workspace, monkeypatch):
    _forbid_read(monkeypatch)
    result = _profile({"path": "../../etc/passwd", "checks": ["dtypes"]})
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert "escapes" in result["error"]["msg"]


def test_target_check_without_target_is_bad_input(workspace, monkeypatch):
    _forbid_read(monkeypatch)
    result = _profile({"path": "train.csv", "checks": ["target_balance"]})
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert "target" in result["error"]["msg"]


def test_drift_without_compare_path_is_bad_input(workspace, monkeypatch):
    _forbid_read(monkeypatch)
    result = _profile({"path": "train.csv", "checks": ["train_test_drift"]})
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert "compare_path" in result["error"]["msg"]


def test_empty_checks_list_is_bad_input(workspace, monkeypatch):
    # validate() lets an empty list through (no items to check against the
    # enum); the precheck is the layer that knows empty is useless.
    _forbid_read(monkeypatch)
    result = _profile({"path": "train.csv", "checks": []})
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


# --- error branches: structure is broken -> error, not data -------------------


def test_missing_file_is_not_found(workspace):
    result = _profile({"path": "nope.csv", "checks": ["dtypes"]})
    assert result["ok"] is False
    assert result["error"]["kind"] == "not_found"


def test_empty_file_is_bad_input(workspace):
    (workspace / "data" / "empty.csv").write_text("")
    result = _profile({"path": "empty.csv", "checks": ["dtypes"]})
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_header_only_file_is_bad_input(workspace):
    (workspace / "data" / "header.csv").write_text("a,b,c\n")
    result = _profile({"path": "header.csv", "checks": ["dtypes"]})
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_ragged_csv_is_bad_input_and_names_the_broken_line(workspace):
    result = _profile({"path": "ragged.csv", "checks": ["missingness"]})
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    # The message must point at the exact line so the failure is actionable,
    # and no statistic from the broken file may leak into the envelope.
    assert "line 5" in result["error"]["msg"]
    assert "missingness" not in repr(result.get("data"))


def test_duplicate_column_names_is_bad_input(workspace):
    (workspace / "data" / "dup.csv").write_text("a,a\n1,2\n")
    result = _profile({"path": "dup.csv", "checks": ["dtypes"]})
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_target_column_absent_from_file_is_bad_input(workspace):
    result = _profile({"path": "train.csv", "checks": ["target_balance"], "target": "label"})
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert "label" in result["error"]["msg"]


# --- happy path: hand-computed numbers ----------------------------------------


def test_missingness_cardinality_and_balance_match_hand_counts(workspace):
    result = _profile({
        "path": "train.csv",
        "checks": ["missingness", "cardinality", "target_balance"],
        "target": "risk",
    })
    assert result["ok"] is True
    data = result["data"]
    assert data["rows_profiled"] == 10
    # Hand-counted: age misses row 8, income misses rows 5 ('') and 10 ('NA').
    assert data["missingness"]["age"] == {"missing": 1, "pct": 0.1}
    assert data["missingness"]["income"] == {"missing": 2, "pct": 0.2}
    assert data["missingness"]["id"]["missing"] == 0
    assert data["cardinality"]["id"]["likely_id"] is True
    assert data["cardinality"]["risk"]["distinct"] == 2
    assert data["target_balance"]["0"] == {"count": 5, "ratio": 0.5}
    assert data["target_balance"]["1"] == {"count": 5, "ratio": 0.5}


def test_dirty_column_is_a_finding_not_an_error(workspace):
    # 3.7 / N/A / '' / 12.0 / null / abc: junk-riddled, but structurally sound.
    # The profile must SUCCEED and flag the mix — dirty values are data.
    (workspace / "data" / "dirty.csv").write_text(
        "id,amount\n1,3.7\n2,N/A\n3,\n4,12.0\n5,null\n6,abc\n"
    )
    result = _profile({"path": "dirty.csv", "checks": ["dtypes"]})
    assert result["ok"] is True
    amount = result["data"]["dtypes"]["amount"]
    assert amount["mixed"] is True
    assert amount["missing"] == 3   # N/A, '', null
    assert amount["float"] == 2 and amount["text"] == 1


def test_numeric_summary_matches_hand_computation(workspace):
    result = _profile({"path": "corr.csv", "checks": ["numeric_summary"]})
    assert result["ok"] is True
    x = result["data"]["numeric_summary"]["x"]
    assert x["count"] == 4 and x["min"] == 1.0 and x["max"] == 4.0
    assert x["mean"] == 2.5
    assert x["std"] == round(math.sqrt(1.25), 6)  # population variance
    assert x["median"] == 2.0                     # nearest-rank, not interpolated


def test_correlation_is_exact_and_skips_constant_columns(workspace):
    result = _profile({
        "path": "corr.csv",
        "checks": ["correlation_with_target"],
        "target": "y",
    })
    assert result["ok"] is True
    corr = result["data"]["correlation_with_target"]
    assert corr["x"] == 1.0        # y = 2x exactly
    assert "noise" not in corr     # constant: correlation undefined, so absent


def test_two_class_text_target_is_encoded_and_correlated(workspace):
    result = _profile({
        "path": "bin.csv",
        "checks": ["correlation_with_target"],
        "target": "label",
    })
    assert result["ok"] is True
    # no/no/yes/yes against 1/2/3/4: r = 2 / sqrt(5), hand-computed.
    assert result["data"]["correlation_with_target"]["f"] == round(2 / math.sqrt(5), 4)


def test_multiclass_text_target_is_skipped_with_a_note_not_an_error(workspace):
    result = _profile({
        "path": "multi.csv",
        "checks": ["correlation_with_target"],
        "target": "label",
    })
    # An unhelpful target is a true fact about the data, not a failure.
    assert result["ok"] is True
    assert "skipped" in result["data"]["correlation_with_target"]["note"]


# --- train_test_drift ----------------------------------------------------------


def test_drift_flags_shifted_and_unseen_but_not_stable_columns(workspace):
    result = _profile({
        "path": "train_d.csv",
        "checks": ["train_test_drift"],
        "compare_path": "test_d.csv",
    })
    assert result["ok"] is True
    drift = result["data"]["train_test_drift"]
    # 'a' shifted its mean by 100 -> smd = 100 / sqrt(1.25), hand-computed.
    assert drift["columns"]["a"]["smd"] == round(100 / math.sqrt(1.25), 4)
    assert drift["columns"]["a"]["drifted"] is True
    # 'b' is the same constant on both sides: std=0 falls through to the
    # frequency comparison and must NOT be flagged.
    assert drift["columns"]["b"]["type"] == "categorical"
    assert drift["columns"]["b"]["drifted"] is False
    # 'cat' gains category 'z' never seen in train — the encoder-breaking case.
    assert drift["columns"]["cat"]["unseen_in_compare"] == ["z"]
    assert drift["columns"]["cat"]["drifted"] is True
    assert drift["flagged"] == ["a", "cat"]
    # The target column exists only in train; that is expected and reported.
    assert drift["only_in_this_file"] == ["risk"]
    assert drift["only_in_compare"] == []


def test_identical_files_show_no_drift(workspace):
    result = _profile({
        "path": "train_d.csv",
        "checks": ["train_test_drift"],
        "compare_path": "train_d.csv",
    })
    assert result["ok"] is True
    assert result["data"]["train_test_drift"]["flagged"] == []


def test_missing_compare_file_is_not_found(workspace):
    result = _profile({
        "path": "train_d.csv",
        "checks": ["train_test_drift"],
        "compare_path": "nope.csv",
    })
    assert result["ok"] is False
    assert result["error"]["kind"] == "not_found"
