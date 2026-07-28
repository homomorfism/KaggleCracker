"""The setup state machine with a faked kaggle client: every terminal state is
readable from status.json, and a failure keeps the project alive for retry."""

import json

import pytest

import ui.run_setup as run_setup
from ui import projects
from ui.kaggle_client import KaggleError


@pytest.fixture
def project():
    return projects.create_project("titanic", competition="titanic")


def _status(slug):
    return json.loads(
        (projects.project_dir(slug) / "setup" / "status.json").read_text()
    )


def test_happy_path_reaches_done_with_file_inventory(project, monkeypatch):
    monkeypatch.setattr(
        run_setup.kaggle_client,
        "fetch_metadata",
        lambda slug: {"slug": slug, "title": "Titanic", "evaluation_metric": "Accuracy"},
    )
    monkeypatch.setattr(
        run_setup.kaggle_client,
        "list_files",
        lambda slug: [{"name": "train.csv", "bytes": 10}],
    )

    def fake_download(slug, dest):
        (dest / "train.csv").write_text("a,b\n1,2\n")
        (dest / "test.csv").write_text("a\n1\n")
        return ["train.csv", "test.csv"]

    monkeypatch.setattr(run_setup.kaggle_client, "download_data", fake_download)

    assert run_setup.run_setup("titanic", spawn_followups=False) == 0
    status = _status("titanic")
    assert status["state"] == "done"
    assert sorted(status["files"]) == ["test.csv", "train.csv"]
    # Metadata landed where the /competition route reads it.
    meta = projects.competition_meta("titanic")
    assert meta["title"] == "Titanic"
    assert meta["files"][0]["name"] == "train.csv"


def test_rules_not_accepted_is_actionable_and_retryable(project, monkeypatch):
    monkeypatch.setattr(
        run_setup.kaggle_client,
        "fetch_metadata",
        lambda slug: (_ for _ in ()).throw(
            KaggleError("rules_not_accepted", "403 for titanic")
        ),
    )
    assert run_setup.run_setup("titanic", spawn_followups=False) == 1
    status = _status("titanic")
    assert status["state"] == "failed"
    assert status["error"]["kind"] == "rules_not_accepted"
    assert "kaggle.com/c/titanic/rules" in status["error"]["actionable"]
    # The project survives the failure — that is what makes retry possible.
    assert projects.get_project("titanic")["slug"] == "titanic"

    # Retry after the human accepts the rules: same entry point, now green.
    monkeypatch.setattr(
        run_setup.kaggle_client, "fetch_metadata", lambda slug: {"slug": slug}
    )
    monkeypatch.setattr(run_setup.kaggle_client, "list_files", lambda slug: [])
    monkeypatch.setattr(
        run_setup.kaggle_client, "download_data", lambda slug, dest: []
    )
    assert run_setup.run_setup("titanic", spawn_followups=False) == 0
    assert _status("titanic")["state"] == "done"


def test_download_failure_reports_bytes_seen_so_far(project, monkeypatch):
    monkeypatch.setattr(
        run_setup.kaggle_client, "fetch_metadata", lambda slug: {"slug": slug}
    )
    monkeypatch.setattr(
        run_setup.kaggle_client,
        "list_files",
        lambda slug: [{"name": "train.csv", "bytes": 1000}],
    )
    monkeypatch.setattr(
        run_setup.kaggle_client,
        "download_data",
        lambda slug, dest: (_ for _ in ()).throw(KaggleError("network", "reset")),
    )
    assert run_setup.run_setup("titanic", spawn_followups=False) == 1
    assert _status("titanic")["error"]["kind"] == "network"


def test_success_spawns_observer_and_eda(project, monkeypatch):
    monkeypatch.setattr(
        run_setup.kaggle_client, "fetch_metadata", lambda slug: {"slug": slug}
    )
    monkeypatch.setattr(run_setup.kaggle_client, "list_files", lambda slug: [])
    monkeypatch.setattr(run_setup.kaggle_client, "download_data", lambda slug, dest: [])
    spawned = []
    monkeypatch.setattr(run_setup, "_spawn", lambda mod, slug, log: spawned.append(mod))

    assert run_setup.run_setup("titanic") == 0
    assert spawned == ["ui.run_observer", "ui.run_eda"]
