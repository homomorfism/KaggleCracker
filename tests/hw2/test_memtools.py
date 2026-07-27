"""Tests for the memory-slice tools (tools/memory/memtools.py).

KC_DOCS and KC_USER are redirected to tmp via the `mem_env` fixture, so no test
writes into the real workspace/ and every save/retrieve is scoped to a known
user. Each test asserts on a consequence — the note is (or is not) actually in
the store afterwards — not merely on a return value.
"""

import pytest

from core.loop import RunState, dispatch, run_agent
from core.registry import Registry
from memory import documents
from tests.fakemodel import FakeModel, Reply, ToolCall
from tools.memory import memtools

USER = "alice"


def _mute(*a):
    return None


@pytest.fixture
def mem_env(tmp_path, monkeypatch):
    """Point the document store and current user at tmp. Returns a registry with
    both memory tools registered."""
    monkeypatch.setenv("KC_DOCS", str(tmp_path / "documents.jsonl"))
    monkeypatch.setenv("KC_USER", USER)
    reg = Registry()
    memtools.register(reg)
    return reg


def test_saving_a_real_preference_persists(mem_env):
    model = FakeModel([
        Reply(tool_calls=[ToolCall("save_memory", {
            "text": "always use lightgbm for tabular data",
            "cue": "model preference",
            "kind": "fact",
        })]),
        Reply(text="noted", tool_calls=[]),
    ])
    out = run_agent([], model, mem_env, input_fn=_mute, output_fn=_mute)
    assert out == "noted"

    # Consequence: the preference is actually retrievable afterwards, and the
    # kind was folded into the cue so it persisted alongside the text.
    docs = documents.retrieve_documents(USER, "lightgbm")
    assert len(docs) == 1
    assert docs[0]["text"] == "always use lightgbm for tabular data"
    assert docs[0]["cue"].startswith("fact ")


def test_greeting_is_noise_the_boundary_is_the_description(mem_env):
    """Documents the intended boundary for save_memory.

    You cannot reliably decide "is this a greeting" with a string comparison, so
    there is deliberately NO machine guard that inspects the text — that would be
    a brittle judgement dressed up as a check. Instead the boundary lives in the
    tool's description, which tells the model not to save greetings, and a
    compliant model declines. This test pins both halves of that contract:
    """
    # (a) The guard text exists in the description the model is given.
    desc = mem_env.get("save_memory").description.lower()
    assert "do not save greetings" in desc

    # (b) A compliant model, offered the tool for a bare 'hello', saves nothing.
    #     It was OFFERED save_memory (so the choice was real) and chose not to.
    model = FakeModel([Reply(text="Hello! What would you like to run?", tool_calls=[])])
    run_agent([], model, mem_env, input_fn=_mute, output_fn=_mute)

    assert model.seen, "model should have been called at least once"
    offered = {t["name"] for t in model.seen[0]}
    assert "save_memory" in offered            # the tempting tool was available
    assert documents.retrieve_documents(USER, "hello") == []  # yet nothing saved


def test_off_list_kind_is_rejected_before_the_body_runs(mem_env):
    # "note" is not in the {fact, rule} enum, so validate() rejects it and the
    # tool body never executes — nothing is written.
    state = RunState()
    result = dispatch(
        ToolCall("save_memory", {"text": "x", "cue": "y", "kind": "note"}),
        mem_env, state, input_fn=_mute, output_fn=_mute,
    )
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"
    assert documents.retrieve_documents(USER, "x") == []  # body never ran


def test_retrieve_memory_returns_shared_note_from_another_user(mem_env):
    # A note bob shared should reach alice through the retrieve tool as DATA.
    documents.save_document("bob", "start from last week's blend", "blend", shared=True)
    state = RunState()
    result = dispatch(
        ToolCall("retrieve_memory", {"query": "blend"}),
        mem_env, state, input_fn=_mute, output_fn=_mute,
    )
    assert result["ok"] is True
    assert result["data"]["count"] == 1
    assert result["data"]["documents"][0]["text"] == "start from last week's blend"


def test_private_note_of_another_user_is_not_retrieved(mem_env):
    # bob's PRIVATE note must never surface for alice, even on an exact match.
    documents.save_document("bob", "secret sauce", "secret", shared=False)
    state = RunState()
    result = dispatch(
        ToolCall("retrieve_memory", {"query": "secret"}),
        mem_env, state, input_fn=_mute, output_fn=_mute,
    )
    assert result["ok"] is True
    assert result["data"]["count"] == 0
