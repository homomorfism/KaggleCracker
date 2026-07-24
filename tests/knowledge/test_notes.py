"""Wiring and behaviour of save_technique_note, driven through dispatch().

The tool is the deliberately reversible, ungated half of the knowledge pair:
these tests pin that it never stops for a human, that its enums reject junk
before the body runs, and that a duplicate note leaves the file untouched.
"""

import json

import pytest

from core.loop import RunState, dispatch
from core.registry import Registry
from tools.knowledge.notes import SAVE_TECHNIQUE_NOTE, register


# --- registration / shape ---------------------------------------------------


def _registered():
    return register(Registry()).get("save_technique_note")


def test_note_tool_registers_under_its_action_name():
    assert _registered() is SAVE_TECHNIQUE_NOTE


def test_note_is_reversible_and_ungated():
    spec = _registered()
    assert spec.irreversible is False
    assert spec.preview is None


def test_confidence_and_effort_are_constrained_enums():
    params = SAVE_TECHNIQUE_NOTE.parameters
    assert set(params["confidence"]["enum"]) == {"confirmed_cv_gain", "claimed", "speculative"}
    assert set(params["est_effort"]["enum"]) == {"low", "medium", "high"}


def test_description_says_when_not_to_call():
    assert "Do NOT" in SAVE_TECHNIQUE_NOTE.description


# --- behaviour: helpers -----------------------------------------------------


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Redirect KC_WORKSPACE at a fresh tmp dir. Returns the workspace root."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    return tmp_path


def _args(**overrides):
    args = {
        "competition_slug": "playground-series-s6e7",
        "technique": "train on synthetic + original data combined",
        "evidence_url": "https://www.kaggle.com/code/alice/eda-notebook",
        "confidence": "claimed",
        "est_effort": "low",
    }
    args.update(overrides)
    return args


def _never(_p=""):
    raise AssertionError("gate must not be reached for an ungated tool")


def _save(args):
    registry = register(Registry())

    class Call:
        name = "save_technique_note"

    call = Call()
    call.args = args
    # input_fn=_never proves the ungated path: if the gate were ever consulted
    # for this reversible tool, the test fails loudly.
    return dispatch(call, registry, RunState(), input_fn=_never)


def _note_file(ws):
    return ws / "knowledge" / "playground-series-s6e7.jsonl"


# --- behaviour: schema and precheck reject before the body -------------------


def test_out_of_enum_confidence_is_rejected_before_the_body(ws):
    result = _save(_args(confidence="pretty_sure"))
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert not _note_file(ws).exists()  # the body never ran


def test_missing_required_confidence_is_rejected(ws):
    args = _args()
    del args["confidence"]
    result = _save(args)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_non_http_evidence_url_is_rejected_by_the_precheck(ws):
    result = _save(_args(evidence_url="ftp://example.com/paper.pdf"))
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert not _note_file(ws).exists()


# --- behaviour: happy path and idempotence -----------------------------------


def test_happy_path_appends_one_line_and_reports_the_count(ws):
    result = _save(_args())
    assert result["ok"] is True
    assert result["data"]["count"] == 1
    assert result["data"]["duplicate"] is False

    lines = _note_file(ws).read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["confidence"] == "claimed"
    assert rec["evidence_url"] == "https://www.kaggle.com/code/alice/eda-notebook"


def test_a_second_distinct_technique_appends(ws):
    _save(_args())
    result = _save(_args(technique="target encoding on school_id", est_effort="medium"))
    assert result["ok"] is True
    assert result["data"]["count"] == 2


def test_duplicate_note_changes_nothing(ws):
    _save(_args())
    before = _note_file(ws).read_text()
    result = _save(_args())
    assert result["ok"] is True
    assert result["data"]["duplicate"] is True
    assert result["data"]["count"] == 1
    # Consequence, not just a return value: the file's contents are unchanged.
    assert _note_file(ws).read_text() == before
