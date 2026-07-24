"""The plan pair, all driven through the real dispatch() order.

The shape tests pin the design: write_preprocessing_plan is the reversible,
ungated half; overwrite_preprocessing_plan is irreversible, previewed and
gated. The behaviour tests then assert on consequences, not just envelopes:
after every rejection or denial the old plan's contents AND mtime are
unchanged, and where the precheck must fire first, reaching the gate at all
fails the test loudly.

The workspace is redirected to tmp_path via the `plans_ws` fixture, so no test
writes into the real workspace/ (the KC_WORKSPACE rule in AGENTS.md).
"""

from dataclasses import dataclass

import pytest

from core.loop import RunState, dispatch
from core.registry import Registry
from tools.recon.paths import plan_path
from tools.recon.plans import OVERWRITE_PLAN, WRITE_PLAN, register


@dataclass
class Call:
    name: str
    args: dict


def _mute(*a, **k):
    return None


def _answer(text):
    """An input_fn that always returns `text`, ignoring the prompt."""
    return lambda _p="": text


def _never(_p=""):
    # Used where no human may be consulted: for the ungated write tool, and for
    # overwrite calls the precheck must reject BEFORE the gate. If the gate is
    # ever reached, this fires and the test fails loudly.
    raise AssertionError("gate must not be reached")


OLD_PLAN = "# plan v1\n- drop id\n- impute age with median\n"
NEW_PLAN = "# plan v2\n- drop id\n- impute age with median\n- clip income at p95\n"


@pytest.fixture
def plans_ws(tmp_path, monkeypatch):
    """Point KC_WORKSPACE at a fresh tmp dir. Returns the workspace root."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    return tmp_path


def _call(name, args, input_fn):
    reg = register(Registry())
    return dispatch(Call(name, args), reg, RunState(),
                    input_fn=input_fn, output_fn=_mute)


def _args(**over):
    a = {"dataset": "train.csv", "plan_markdown": NEW_PLAN}
    a.update(over)
    return a


def _seed_old_plan():
    path = plan_path("train.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OLD_PLAN)
    return path


# --- registration / shape -----------------------------------------------------


def test_both_tools_register_under_their_action_names():
    reg = register(Registry())
    assert reg.get("write_preprocessing_plan") is WRITE_PLAN
    assert reg.get("overwrite_preprocessing_plan") is OVERWRITE_PLAN


def test_write_is_the_reversible_ungated_half():
    assert WRITE_PLAN.irreversible is False
    assert WRITE_PLAN.preview is None
    assert WRITE_PLAN.precheck is not None


def test_overwrite_is_irreversible_with_preview_and_precheck():
    assert OVERWRITE_PLAN.irreversible is True
    assert OVERWRITE_PLAN.preview is not None
    assert OVERWRITE_PLAN.precheck is not None


# --- write: the ungated path ----------------------------------------------------


def test_writing_a_new_plan_needs_no_human(plans_ws):
    # input_fn=_never proves the point: if anything consults a human here, the
    # test blows up. Reversible actions stay ungated.
    result = _call("write_preprocessing_plan", _args(), input_fn=_never)
    assert result["ok"] is True
    assert plan_path("train.csv").read_text() == NEW_PLAN


def test_writing_over_an_existing_plan_is_refused_and_redirected(plans_ws):
    path = _seed_old_plan()
    before = path.stat().st_mtime_ns
    result = _call("write_preprocessing_plan", _args(), input_fn=_never)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    # The message must hand the model its next move, not a dead end.
    assert "overwrite_preprocessing_plan" in result["error"]["msg"]
    assert path.read_text() == OLD_PLAN
    assert path.stat().st_mtime_ns == before


def test_missing_plan_markdown_is_rejected_and_nothing_is_written(plans_ws):
    result = _call("write_preprocessing_plan", {"dataset": "train.csv"}, input_fn=_never)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    # validate() rejected before the body: the plans dir was never created.
    assert not plan_path("train.csv").parent.exists()


def test_empty_plan_markdown_is_rejected(plans_ws):
    result = _call("write_preprocessing_plan", _args(plan_markdown="  \n"), input_fn=_never)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


@pytest.mark.parametrize("dataset", ["../evil", "a/b.csv", "..", ""])
def test_unsafe_dataset_names_are_rejected_with_nothing_written(plans_ws, dataset):
    result = _call("write_preprocessing_plan", _args(dataset=dataset), input_fn=_never)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert not plan_path("evil").exists()
    assert not (plans_ws / "plans").exists()


# --- overwrite: precheck before gate, then the gate itself -----------------------


def test_overwrite_with_no_existing_plan_is_not_found_before_the_gate(plans_ws):
    # Nothing to destroy -> nothing to approve. _never proves the human was
    # never consulted about a no-op.
    result = _call("overwrite_preprocessing_plan", _args(), input_fn=_never)
    assert result["ok"] is False
    assert result["error"]["kind"] == "not_found"
    assert "write_preprocessing_plan" in result["error"]["msg"]
    assert not plan_path("train.csv").exists()


@pytest.mark.parametrize("answer", ["yes", "y", "YES", " y "])
def test_approval_replaces_the_plan(plans_ws, answer):
    _seed_old_plan()
    result = _call("overwrite_preprocessing_plan", _args(), input_fn=_answer(answer))
    assert result["ok"] is True
    assert plan_path("train.csv").read_text() == NEW_PLAN


@pytest.mark.parametrize("answer", ["", "ok", "sure", "go ahead", "no", "n"])
def test_everything_else_denies_and_the_old_plan_survives_untouched(plans_ws, answer):
    path = _seed_old_plan()
    before = path.stat().st_mtime_ns
    result = _call("overwrite_preprocessing_plan", _args(), input_fn=_answer(answer))
    assert result["ok"] is False
    assert result["error"]["kind"] == "denied_by_human"
    assert result["error"]["retryable"] is False
    # The consequence, not just the envelope: contents AND mtime unchanged.
    assert path.read_text() == OLD_PLAN
    assert path.stat().st_mtime_ns == before


def test_preview_shows_the_human_what_dies(plans_ws):
    _seed_old_plan()
    shown = []
    reg = register(Registry())
    dispatch(Call("overwrite_preprocessing_plan", _args()), reg, RunState(),
             input_fn=_answer("yes"), output_fn=shown.append)
    assert len(shown) == 1
    preview = shown[0]
    assert "cannot be recovered" in preview      # the irreversibility line
    assert "# plan v1" in preview                # the doomed plan's own first line
    assert str(plan_path("train.csv")) in preview  # exactly which file
