"""The knowledge slice's stopping condition: stall detection over the durable
search log, plus the run_agent integration proving the loop actually stops.
"""

import json

import pytest

from core.loop import run_agent
from core.registry import Registry
from tests.fakemodel import FakeModel
from tools.knowledge.paths import knowledge_dir, search_log_path
from tools.knowledge.stopping import stall_detector


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Redirect KC_WORKSPACE at a fresh tmp dir. Returns the workspace root."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    return tmp_path


def _log(*entries):
    """Seed the search log as search_kaggle_discussions would have written it."""
    knowledge_dir().mkdir(parents=True, exist_ok=True)
    with search_log_path().open("a") as f:
        for query, urls in entries:
            f.write(json.dumps({"slug": "s6e7", "query": query, "urls": urls}) + "\n")


# --- unit: when the detector fires -------------------------------------------


def test_no_log_means_no_stall(ws):
    assert stall_detector()(None, []) is None


def test_a_single_search_is_never_a_stall(ws):
    _log(("ensembling", ["https://k/a"]))
    assert stall_detector()(None, []) is None


def test_progress_is_not_a_stall(ws):
    _log(("ensembling", ["https://k/a"]), ("target encoding", ["https://k/b"]))
    assert stall_detector()(None, []) is None


def test_repeated_query_is_a_stall(ws):
    # Same question twice — even though it keeps "finding" the same source, the
    # model is going in circles and should summarise instead.
    _log(("ensembling", ["https://k/a"]), ("ensembling", ["https://k/a"]))
    assert "stall" in stall_detector()(None, [])


def test_repeated_query_matches_case_insensitively(ws):
    _log(("Ensembling", ["https://k/a"]), ("ensembling  ", ["https://k/a"]))
    assert "stall" in stall_detector()(None, [])


def test_two_searches_with_no_new_sources_is_a_stall(ws):
    # Different queries, but the last two only re-surfaced what the first had
    # already found (or nothing at all): the corpus is mined out for now.
    _log(
        ("ensembling", ["https://k/a", "https://k/b"]),
        ("stacking", ["https://k/b"]),
        ("blending", []),
    )
    assert "stall" in stall_detector()(None, [])


def test_one_dry_search_after_a_fresh_one_is_not_yet_a_stall(ws):
    # One empty answer is normal early in a Playground month; two in a row is
    # the signal. Patience of exactly two, pinned.
    _log(("ensembling", ["https://k/a"]), ("stacking", []))
    assert stall_detector()(None, []) is None


# --- integration: run_agent stops without consulting the model ---------------


def test_run_agent_stops_on_stall_before_calling_the_model(ws):
    _log(("ensembling", ["https://k/a"]), ("ensembling", ["https://k/a"]))
    # FakeModel with no scripted replies: if run_agent asked it anything, it
    # would raise. Returning None proves the stall fired first.
    model = FakeModel([])
    assert run_agent([], model, Registry(), should_stop=stall_detector()) is None
    assert model.seen == []
