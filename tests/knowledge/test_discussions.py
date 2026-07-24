"""Wiring and behaviour of search_kaggle_discussions, driven through dispatch().

The heart of this file is the graded contrast: zero results is a SUCCESS the
model can reason about, a 429 is a FAILURE it must react to. A naive tool
collapses both into an empty list; these tests pin that ours never does.

No test touches the network: _http_get_json is monkeypatched with canned
payloads or made to raise the exact exceptions urllib would.
"""

import json
import socket
import urllib.error

import pytest

from core.loop import RunState, dispatch
from core.registry import Registry
from tools.knowledge import discussions
from tools.knowledge.discussions import SEARCH_DISCUSSIONS, register
from tools.knowledge.paths import search_log_path


# --- registration / shape ---------------------------------------------------


def _registered():
    return register(Registry()).get("search_kaggle_discussions")


def test_search_tool_registers_under_its_action_name():
    assert _registered() is SEARCH_DISCUSSIONS


def test_search_is_reversible_and_ungated():
    # The contrast with fetch_external_dataset is the point: reading public
    # pages must never stop for a human.
    spec = _registered()
    assert spec.irreversible is False
    assert spec.preview is None


def test_sort_is_a_constrained_enum_and_max_results_is_capped():
    params = SEARCH_DISCUSSIONS.parameters
    assert set(params["sort"]["enum"]) == {"hot", "recent", "votes"}
    assert params["max_results"]["max"] == 50


def test_description_says_when_not_to_call():
    assert "Do NOT" in SEARCH_DISCUSSIONS.description


# --- behaviour: helpers -----------------------------------------------------


CANNED = [
    {"ref": "alice/eda-notebook", "title": "EDA and baseline", "author": "alice", "totalVotes": 41},
    {"ref": "bob/gbm-tricks", "title": "GBM tricks that worked", "author": "bob", "totalVotes": 17},
]


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Redirect KC_WORKSPACE at a fresh tmp dir. Returns the workspace root."""
    monkeypatch.setenv("KC_WORKSPACE", str(tmp_path))
    return tmp_path


def _args(**overrides):
    args = {"competition_slug": "playground-series-s6e7", "query": "feature engineering"}
    args.update(overrides)
    return args


def _search(args, fetch, monkeypatch):
    """Run one search through the real dispatch order with a fake HTTP layer."""
    monkeypatch.setattr(discussions, "_http_get_json", fetch)
    registry = register(Registry())

    class Call:
        name = "search_kaggle_discussions"

    call = Call()
    call.args = args
    return dispatch(call, registry, RunState())


def _no_network(url):
    raise AssertionError("the network must not be touched")


def _log_lines():
    path = search_log_path()
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --- behaviour: schema and precheck reject before any network traffic --------


def test_out_of_enum_sort_is_rejected_before_the_body(ws, monkeypatch):
    result = _search(_args(sort="best"), _no_network, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_oversized_max_results_is_rejected_before_the_body(ws, monkeypatch):
    result = _search(_args(max_results=51), _no_network, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_malformed_slug_is_rejected_by_the_precheck(ws, monkeypatch):
    # A traversal-shaped slug never reaches the network (_no_network would fire).
    result = _search(_args(competition_slug="../../../etc"), _no_network, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


def test_blank_query_is_rejected_by_the_precheck(ws, monkeypatch):
    result = _search(_args(query="   "), _no_network, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "bad_input"


# --- behaviour: happy path ---------------------------------------------------


def test_happy_path_returns_parsed_results_and_logs_the_search(ws, monkeypatch):
    result = _search(_args(), lambda url: list(CANNED), monkeypatch)
    assert result["ok"] is True
    data = result["data"]
    assert data["count"] == 2
    assert data["results"][0] == {
        "title": "EDA and baseline",
        "url": "https://www.kaggle.com/code/alice/eda-notebook",
        "author": "alice",
        "votes": 41,
    }
    # Consequence: the durable search log the stall detector reads gained a row.
    lines = _log_lines()
    assert len(lines) == 1
    assert lines[0]["query"] == "feature engineering"
    assert lines[0]["urls"] == [r["url"] for r in data["results"]]


def test_max_results_truncates_an_overlong_page(ws, monkeypatch):
    result = _search(_args(max_results=1), lambda url: list(CANNED), monkeypatch)
    assert result["ok"] is True
    assert result["data"]["count"] == 1


# --- behaviour: THE graded contrast — empty is data, 429 is an error ---------


def _http_429(url):
    raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)


def test_zero_results_is_valid_data_not_an_error(ws, monkeypatch):
    result = _search(_args(), lambda url: [], monkeypatch)
    assert result["ok"] is True
    assert result["data"]["results"] == []
    assert "valid result" in result["data"]["note"]
    # An empty answer still counts as a completed search for stall detection.
    assert len(_log_lines()) == 1


def test_rate_limit_is_an_error_not_an_empty_list(ws, monkeypatch):
    result = _search(_args(), _http_429, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "rate_limited"
    assert result["error"]["retryable"] is True
    # Consequence: a search that never ran is NOT logged, so a flaky network
    # can never trip the stall condition.
    assert _log_lines() == []


def test_the_two_branches_diverge_visibly(ws, monkeypatch):
    """The side-by-side the one-pager is built on: identical query, one gets a
    truthful 'nothing published yet', the other a failure to retry later. A
    naive tool would hand the loop the same empty list for both."""
    empty = _search(_args(), lambda url: [], monkeypatch)
    limited = _search(_args(), _http_429, monkeypatch)
    assert empty["ok"] is True and limited["ok"] is False
    assert limited["error"]["kind"] == "rate_limited"


# --- behaviour: the remaining network failure kinds --------------------------


def test_404_maps_to_not_found(ws, monkeypatch):
    def fetch(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    result = _search(_args(), fetch, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "not_found"


def test_socket_timeout_maps_to_timeout(ws, monkeypatch):
    def fetch(url):
        raise socket.timeout("timed out")

    result = _search(_args(), fetch, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "timeout"
    assert result["error"]["retryable"] is True


def test_unreachable_network_maps_to_exec_failed(ws, monkeypatch):
    def fetch(url):
        raise urllib.error.URLError("no route to host")

    result = _search(_args(), fetch, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "exec_failed"


def test_unparseable_payload_maps_to_exec_failed(ws, monkeypatch):
    result = _search(_args(), lambda url: {"unexpected": "shape"}, monkeypatch)
    assert result["ok"] is False
    assert result["error"]["kind"] == "exec_failed"
