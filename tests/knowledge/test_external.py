"""Wiring and behaviour of fetch_external_dataset, the knowledge slice's gated
irreversible action, driven through dispatch().

The behaviour tests exercise the real order — validate, precheck, gate, body —
and assert on the consequence each time: after a denial nothing was downloaded
and nothing was disclosed; after an approval the manifest records exactly what
the human said yes to; after a failed download no half-fetched directory or
manifest line survives.
"""

import json

import pytest

from core.contracts import err
from core.loop import RunState, dispatch
from core.registry import Registry
from tools.knowledge import external
from tools.knowledge.external import FETCH_EXTERNAL_DATASET, register


# --- registration / shape ---------------------------------------------------


def _registered():
    return register(Registry()).get("fetch_external_dataset")


def test_fetch_tool_registers_under_its_action_name():
    assert _registered() is FETCH_EXTERNAL_DATASET


def test_fetch_is_irreversible_and_carries_a_preview():
    # irreversible=True is what sends this tool through the gate; the registry
    # refuses an irreversible tool with no preview, so both must hold.
    spec = _registered()
    assert spec.irreversible is True
    assert spec.preview is not None


def test_fetch_has_a_precheck_that_runs_before_the_gate():
    assert _registered().precheck is not None


def test_merge_defaults_to_false():
    # An omitted flag downloads for inspection only; it never silently rewires
    # the training set.
    assert FETCH_EXTERNAL_DATASET.parameters["merge_into_training"]["default"] is False


def test_description_says_when_not_to_call():
    assert "Do NOT" in FETCH_EXTERNAL_DATASET.description


# --- behaviour: helpers -----------------------------------------------------


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Redirect KC_WORKSPACE at a fresh tmp dir. Returns the workspace root."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    return tmp_path


REF = "worlddata/student-health-original"


def _args(**overrides):
    args = {"dataset_ref": REF}
    args.update(overrides)
    return args


def _approve(_p=""):
    return "yes"


def _deny(_p=""):
    return "no"


def _never(_p=""):
    # Used where the precheck must reject before the gate: if the gate is ever
    # reached, this fires and the test fails loudly instead of silently passing.
    raise AssertionError("gate must not be reached")


def _mute(*a, **k):
    return None


def _fetch(args, input_fn, monkeypatch, download=None, output_fn=_mute):
    """Run one fetch through the real dispatch order with a fake downloader."""
    if download is None:
        def download(ref, dest):
            raise AssertionError("download must not run")
    monkeypatch.setattr(external, "_download", download)
    registry = register(Registry())

    class Call:
        name = "fetch_external_dataset"

    call = Call()
    call.args = args
    return dispatch(call, registry, RunState(), input_fn=input_fn, output_fn=output_fn)


def _dest(ws):
    return ws / "data" / "external" / "student-health-original"


def _manifest(ws):
    return ws / "data" / "external_manifest.jsonl"


def _fake_download_ok(ref, dest):
    """Stands in for the kaggle CLI: drop one file where the download would."""
    (dest / "original.csv").write_text("id,target\n1,0\n")
    return None


# --- behaviour: precheck rejects before the gate -----------------------------


def test_malformed_ref_is_rejected_before_the_gate(ws, monkeypatch):
    result = _fetch(_args(dataset_ref="not-a-kaggle-ref"), _never, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_traversal_shaped_ref_is_rejected_before_the_gate(ws, monkeypatch):
    result = _fetch(_args(dataset_ref="../../etc/passwd"), _never, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_already_fetched_dataset_is_rejected_before_the_gate(ws, monkeypatch):
    _dest(ws).mkdir(parents=True)
    result = _fetch(_args(), _never, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert "already fetched" in result["error"]["msg"]


def test_non_bool_merge_flag_is_rejected_by_the_schema(ws, monkeypatch):
    result = _fetch(_args(merge_into_training="yes"), _never, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


# --- behaviour: the gate -----------------------------------------------------


def test_denial_downloads_nothing_and_discloses_nothing(ws, monkeypatch):
    result = _fetch(_args(), _deny, monkeypatch)  # download stub would raise if run
    assert result["ok"] is False
    assert result["error"]["kind"] == "denied_by_human"
    assert result["error"]["retryable"] is False
    # Consequence: the filesystem is exactly as it was.
    assert not _dest(ws).exists()
    assert not _manifest(ws).exists()


def test_empty_input_is_a_denial(ws, monkeypatch):
    result = _fetch(_args(), lambda _p="": "", monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "denied_by_human"
    assert not _dest(ws).exists()


def test_approval_downloads_and_records_the_disclosure(ws, monkeypatch):
    result = _fetch(
        _args(merge_into_training=True), _approve, monkeypatch, download=_fake_download_ok
    )
    assert result["ok"] is True
    assert result["data"]["files"] == ["original.csv"]
    assert (_dest(ws) / "original.csv").exists()

    # The disclosure record carries exactly what the human approved.
    lines = _manifest(ws).read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["dataset_ref"] == REF
    assert rec["merged_into_training"] is True


def test_preview_names_the_dataset_and_the_disclosure_obligation(ws, monkeypatch):
    shown = []
    _fetch(_args(), _approve, monkeypatch, download=_fake_download_ok,
           output_fn=shown.append)
    assert len(shown) == 1
    preview = shown[0]
    assert REF in preview
    assert "disclosed" in preview
    assert "leak" in preview


# --- behaviour: a failed download leaves no trace ----------------------------


def test_missing_dataset_maps_to_not_found_and_cleans_up(ws, monkeypatch):
    def download(ref, dest):
        return err("not_found", "no such dataset on Kaggle: %r" % ref)

    result = _fetch(_args(), _approve, monkeypatch, download=download)
    assert result["ok"] is False
    assert result["error"]["kind"] == "not_found"
    # Consequence: no half-fetched directory survives to trip the
    # already-fetched precheck on a later, corrected attempt; no disclosure
    # line exists for data we do not have.
    assert not _dest(ws).exists()
    assert not _manifest(ws).exists()


def test_partial_download_is_removed_on_failure(ws, monkeypatch):
    def download(ref, dest):
        (dest / "partial.csv").write_text("id\n")  # half-written, then dies
        return err("exec_failed", "connection dropped mid-download")

    result = _fetch(_args(), _approve, monkeypatch, download=download)
    assert result["ok"] is False
    assert result["error"]["kind"] == "exec_failed"
    assert not _dest(ws).exists()
