"""The judged scorers against a FAKE judge — no network, deterministic.

The assignment's rule: a scorer nobody tested is a number nobody should
trust. These tests pin the parsing, clamping and failure behaviour of the
judge plumbing using canned replies, plus a two-case fixture run.
"""

import pytest

from evals import generation_metrics as gm


def test_scores_parse_and_clamp(monkeypatch):
    replies = iter(
        ['{"score": 0.8, "reason": "grounded"}',
         'noise before {"score": 1.7, "reason": "over"} noise after',
         '{"score": -3, "reason": "under"}']
    )
    monkeypatch.setattr(gm, "complete", lambda s, u, **kw: next(replies))
    assert gm.faithfulness("q", "a", "ctx")["score"] == pytest.approx(0.8)
    # Out-of-range judge scores clamp instead of poisoning the average.
    assert gm.answer_relevance("q", "a")["score"] == 1.0
    assert gm.context_recall("golden", "ctx")["score"] == 0.0


def test_malformed_judge_reply_raises(monkeypatch):
    monkeypatch.setattr(gm, "complete", lambda s, u, **kw: "I think it is fine.")
    with pytest.raises(ValueError):
        gm.faithfulness("q", "a", "ctx")


def test_two_case_fixture_run(monkeypatch):
    """A miniature end-to-end of the scorer set over fixture cases: the
    grounded answer must outscore the invented one on faithfulness."""

    def fake_judge(system, user, **kw):
        if "FAITHFULNESS" in system:
            good = "gate approves" in user and "invented" not in user
            return '{"score": %s, "reason": "fixture"}' % ("1.0" if good else "0.1")
        return '{"score": 0.9, "reason": "fixture"}'

    monkeypatch.setattr(gm, "complete", fake_judge)
    context = "the gate approves irreversible actions"
    grounded = gm.faithfulness("who approves?", "the gate approves them", context)
    invented = gm.faithfulness("who approves?", "an invented committee votes", context)
    assert grounded["score"] > invented["score"]


def test_generate_answer_packs_retriever_order(monkeypatch):
    seen = {}

    def fake_complete(system, user, **kw):
        seen["user"] = user
        return "answer"

    monkeypatch.setattr(gm, "complete", fake_complete)
    chunks = [{"text": "FIRST"}, {"text": "SECOND"}]
    answer, context = gm.generate_answer("q", chunks)
    assert answer == "answer"
    # Retriever order preserved — the metrics upstream read this order too.
    assert context.index("FIRST") < context.index("SECOND")
    assert "FIRST" in seen["user"] and "q" in seen["user"]
