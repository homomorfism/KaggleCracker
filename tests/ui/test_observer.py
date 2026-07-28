"""The observer pipeline with a fake kaggle client and a fake LLM: dedupe is
what makes refresh affordable, and one bad notebook never sinks the batch."""

import json

import pytest
from fastapi.testclient import TestClient

import ui.app
import ui.run_observer as run_observer
from ui import projects
from ui.kaggle_client import KaggleError


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def project():
    return projects.create_project("titanic", competition="titanic")


def _kernel(ref, votes=10):
    return {
        "ref": ref,
        "title": "NB %s" % ref,
        "author": "A",
        "votes": votes,
        "language": "python",
        "url": "https://www.kaggle.com/code/%s" % ref,
    }


class FakeClient:
    KaggleError = KaggleError

    def __init__(self, kernels, sources=None, fail_pull=()):
        self.kernels = kernels
        self.sources = sources or {}
        self.fail_pull = set(fail_pull)
        self.pulled = []

    def list_kernels(self, slug, page_size=20):
        return self.kernels

    def pull_kernel(self, ref, scratch):
        self.pulled.append(ref)
        if ref in self.fail_pull:
            raise KaggleError("not_found", "gone")
        return {"ref": ref, "source": self.sources.get(ref, "import pandas as pd")}


def _summary_json(name="target encoding"):
    return json.dumps(
        {
            "summary": "does things",
            "models": ["xgboost"],
            "cv_claim": "0.81",
            "lb_claim": "",
            "techniques": [
                {
                    "name": name,
                    "why_it_matters": "leak-free categorials",
                    "code_snippet": "import pandas as pd",
                    "snippet_explanation": "the import",
                }
            ],
        }
    )


def _llm_ok(system, user, max_tokens=2000):
    if "aggregate" in system.lower():
        return "# Community briefing\n## Technique frequency\n- target encoding — 1"
    return _summary_json()


def _store(slug):
    path = projects.project_dir(slug) / "knowledge" / "notebooks.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_happy_path_writes_store_and_summary(project):
    client = FakeClient([_kernel("a/one"), _kernel("b/two", votes=5)])
    assert run_observer.run_observer("titanic", llm=_llm_ok, client=client) == 0

    records = _store("titanic")
    assert [r["ref"] for r in records] == ["a/one", "b/two"]
    assert records[0]["techniques"][0]["code_snippet"] == "import pandas as pd"
    summary = (projects.project_dir("titanic") / "knowledge" / "summary.md").read_text()
    assert "Community briefing" in summary
    from ui.journal import run_status

    assert run_status(projects.project_dir("titanic") / "observer" / "journal.jsonl") == "finished"


def test_refresh_skips_already_summarized(project):
    client = FakeClient([_kernel("a/one")])
    run_observer.run_observer("titanic", llm=_llm_ok, client=client)
    client.kernels = [_kernel("a/one"), _kernel("c/new")]
    run_observer.run_observer("titanic", llm=_llm_ok, client=client)
    # a/one was pulled exactly once across both refreshes.
    assert client.pulled.count("a/one") == 1
    assert [r["ref"] for r in _store("titanic")] == ["a/one", "c/new"]


def test_bad_json_gets_one_retry_then_skip(project):
    calls = {"n": 0}

    def flaky_llm(system, user, max_tokens=2000):
        if "aggregate" in system.lower():
            return "# Community briefing"
        calls["n"] += 1
        if calls["n"] == 1:
            return "sorry, no JSON today"
        return _summary_json()

    client = FakeClient([_kernel("a/one")])
    assert run_observer.run_observer("titanic", llm=flaky_llm, client=client) == 0
    assert calls["n"] == 2  # the retry happened
    assert [r["ref"] for r in _store("titanic")] == ["a/one"]


def test_one_bad_notebook_does_not_sink_the_batch(project):
    client = FakeClient(
        [_kernel("a/one"), _kernel("bad/pull"), _kernel("c/three")],
        fail_pull=["bad/pull"],
    )
    assert run_observer.run_observer("titanic", llm=_llm_ok, client=client) == 0
    assert [r["ref"] for r in _store("titanic")] == ["a/one", "c/three"]
    events = (projects.project_dir("titanic") / "observer" / "journal.jsonl").read_text()
    assert "kernel_failed" in events


def test_missing_api_key_fails_early(project, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    client = FakeClient([_kernel("a/one")])
    assert run_observer.run_observer("titanic", llm=_llm_ok, client=client) == 1
    assert client.pulled == []  # nothing was fetched before the honest failure


def test_observer_routes(project, monkeypatch):
    api = TestClient(ui.app.app, raise_server_exceptions=False)

    body = api.get("/api/projects/titanic/observer").json()
    assert body == {"status": "none", "notebooks": [], "summary_md": ""}

    client = FakeClient([_kernel("a/one", votes=3), _kernel("b/two", votes=9)])
    run_observer.run_observer("titanic", llm=_llm_ok, client=client)

    body = api.get("/api/projects/titanic/observer").json()
    assert body["status"] == "finished"
    # Sorted by votes, best first, regardless of summarize order.
    assert [n["ref"] for n in body["notebooks"]] == ["b/two", "a/one"]
    assert "Community briefing" in body["summary_md"]

    spawned = []
    monkeypatch.setattr(ui.app, "_spawn", lambda mod, args, log: spawned.append(mod))
    assert api.post("/api/projects/titanic/observer/refresh").status_code == 200
    assert spawned == ["ui.run_observer"]

    lock = projects.project_dir("titanic") / "observer" / "lock"
    lock.mkdir()
    assert api.post("/api/projects/titanic/observer/refresh").status_code == 409

    events = api.get("/api/projects/titanic/observer/events?since=0").json()
    assert events["status"] == "finished"
    assert [e["type"] for e in events["events"]][0] == "run_started"
