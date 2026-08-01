"""Dashboard.

Two properties matter beyond "the page renders": the viewer must not be able to
write to a running search, and the approve path must not be able to spend a
submission twice or send an invalid one.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from kagglecracker.db.store import NodeStore, ProtocolStore, init_db
from kagglecracker.web.app import create_app

SAMPLE_IDS = ["0013_01", "0018_01", "0019_01", "0021_01", "0023_01"]


@pytest.fixture
def app_env(settings, tmp_path):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PassengerId": SAMPLE_IDS, "Transported": [False] * 5}).to_csv(
        settings.data_dir / "sample_submission.csv", index=False
    )
    conn = init_db(settings.db_path)
    protocol = ProtocolStore(conn).set_current(
        metric_name="accuracy", direction="higher_is_better",
        fold_scheme="StratifiedGroupKFold", n_folds=5, seed=42,
        rationale="Kaggle split train/test by party.",
    )
    run_id = conn.execute(
        "INSERT INTO runs (protocol_id, max_nodes, max_wall_clock_s, max_usd, "
        "max_consecutive_failures, max_consecutive_infra_errors) VALUES (?,20,3600,5.0,5,3)",
        (protocol.id,),
    ).lastrowid
    store = NodeStore(conn)

    def add(**kw):
        base = dict(
            run_id=run_id, protocol_id=protocol.id, parent_id=None, action="draft",
            depth=0, code="print('hello')", status="ok", cv_score=0.81,
            fold_scores=[0.80, 0.81, 0.82, 0.80, 0.82], usd=0.0017,
            tokens_in=900, tokens_out=2400, prompt_chars=8900,
            run_flags=["--network", "none"],
        )
        return store.insert(**{**base, **kw})

    return settings, conn, protocol, add


@pytest.fixture
def client(app_env):
    settings, *_ = app_env
    return TestClient(create_app(settings))


def write_submission(settings, node, *, valid: bool = True):
    """Put a submission.csv where the approve handler will look for it."""
    d = settings.runs_dir / f"n{node.id}"
    d.mkdir(parents=True, exist_ok=True)
    cols = {"PassengerId": SAMPLE_IDS, "Transported": [True] * 5}
    if not valid:
        cols = {"PassengerId": SAMPLE_IDS, "WrongName": [True] * 5}
    pd.DataFrame(cols).to_csv(d / "submission.csv", index=False)
    (d / "stdout.log").write_text("fold 0: 0.80\n")
    return d


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_dashboard_renders_all_four_numbered_sections(client, app_env):
    _, _, _, add = app_env
    add(cv_score=0.815)
    body = client.get("/").text
    for heading in ("1. Solution tree", "2. Leaderboard", "3. Node detail", "4. Context"):
        assert heading in body


def test_tree_is_rendered_server_side_as_node_cards(client, app_env):
    """The tree is plain server-rendered HTML — no client-side renderer to fail.
    A child node is nested inside its parent's list item."""
    _, _, _, add = app_env
    parent = add(cv_score=0.80)
    child = add(cv_score=0.82, action="improve", parent_id=parent.id, depth=1)
    body = client.get("/").text
    assert 'class="tree"' in body
    assert f"draft #{parent.id}" in body
    assert f"improve #{child.id}" in body


def _node(i, parent, status="ok", cv=0.8):
    from kagglecracker.db.store import Node

    return Node(
        id=i, run_id=1, protocol_id=1, parent_id=parent, action="improve", depth=0,
        code="", status=status, error_text=None, cv_score=cv,
        fold_scores=[cv] * 5 if cv else [], stdout_path=None, prompt=None, prompt_chars=0,
        tokens_in=0, tokens_out=0, usd=0.0, created_at="",
    )


def test_tree_nests_a_child_under_its_parent():
    from kagglecracker.web.app import build_tree

    roots = build_tree([_node(1, None), _node(2, 1), _node(3, 2)])
    assert len(roots) == 1
    assert roots[0]["node"].id == 1
    assert roots[0]["children"][0]["node"].id == 2
    assert roots[0]["children"][0]["children"][0]["node"].id == 3


def test_tree_marks_the_best_node_distinctly():
    from kagglecracker.web.app import build_tree

    roots = build_tree([_node(1, None, cv=0.80), _node(2, 1, cv=0.85)], best_id=2)
    assert roots[0]["cls"] == "ok"
    assert roots[0]["children"][0]["cls"] == "best"


def test_tree_never_treats_a_failed_node_as_scored():
    """A node that failed must never display a score, however it came to have
    one — the same rule ranking.py enforces."""
    from kagglecracker.web.app import build_tree

    [entry] = build_tree([_node(1, None, status="error", cv=0.99)])
    assert entry["scored"] is False
    assert entry["cls"] == "bad"


def test_tree_distinguishes_infra_failures():
    """Not the agent's fault, so it should not read as a failed attempt."""
    from kagglecracker.web.app import build_tree

    [entry] = build_tree([_node(1, None, status="infra_error", cv=None)])
    assert entry["cls"] == "infra"


def test_tree_handles_an_empty_tree():
    from kagglecracker.web.app import build_tree

    assert build_tree([]) == []


def test_leaderboard_shows_fold_spread_not_just_the_mean(client, app_env):
    """The CV-overfit mitigation: the top row is partly the luckiest draw."""
    _, _, _, add = app_env
    add(cv_score=0.82, fold_scores=[0.70, 0.94, 0.82, 0.82, 0.82])
    body = client.get("/").text
    assert "fold std" in body
    assert "fold min" in body
    assert "0.70000" in body


def test_a_failed_node_never_displays_a_score(client, app_env):
    """Anywhere — leaderboard or tree. A score on a failed node is a lie about
    what happened, and the tree is the panel a demo actually shows."""
    _, _, _, add = app_env
    add(cv_score=0.99, status="error", fold_scores=None, error_text="boom")
    add(cv_score=0.50)
    body = client.get("/").text
    assert "0.99000" not in body
    assert "error" in body


def test_a_node_can_be_selected_from_the_leaderboard(client, app_env):
    _, _, _, add = app_env
    add(cv_score=0.80)
    second = add(cv_score=0.70)
    body = client.get(f"/?node={second.id}").text
    assert f"#{second.id}" in body


def test_failure_text_is_shown_for_a_failed_node(client, app_env):
    _, _, _, add = app_env
    node = add(status="error", cv_score=None, fold_scores=None,
               error_text="ValueError: Categorical categories cannot be null")
    body = client.get(f"/?node={node.id}").text
    assert "Categorical categories cannot be null" in body


def test_context_panel_explains_why_it_is_empty(client):
    """Rather than showing a blank box — the emptiness is a measured decision."""
    body = client.get("/").text
    assert "No context documents" in body
    assert "kagglecracker eda" in body


def test_empty_database_does_not_crash(settings, tmp_path):
    init_db(settings.db_path)
    body = TestClient(create_app(settings)).get("/")
    assert body.status_code == 200
    assert "No CV protocol seeded" in body.text


# --------------------------------------------------------------------------
# approve — the only write the viewer may make
# --------------------------------------------------------------------------


def test_approve_validates_before_queueing(client, app_env):
    settings, conn, _, add = app_env
    node = add(cv_score=0.82)
    d = write_submission(settings, node)
    conn.execute("UPDATE nodes SET stdout_path = ? WHERE id = ?", (str(d / "stdout.log"), node.id))

    r = client.post(f"/approve/{node.id}")
    assert r.status_code == 200
    assert "validated and queued" in r.text
    assert conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 1


def test_approve_refuses_an_invalid_submission(client, app_env):
    settings, conn, _, add = app_env
    node = add(cv_score=0.82)
    d = write_submission(settings, node, valid=False)
    conn.execute("UPDATE nodes SET stdout_path = ? WHERE id = ?", (str(d / "stdout.log"), node.id))

    r = client.post(f"/approve/{node.id}")
    assert "failed validation" in r.text
    assert conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0


def test_a_double_click_cannot_spend_two_submissions(client, app_env):
    """The quota is ~10/day and irreplaceable; the button is clickable twice."""
    settings, conn, _, add = app_env
    node = add(cv_score=0.82)
    d = write_submission(settings, node)
    conn.execute("UPDATE nodes SET stdout_path = ? WHERE id = ?", (str(d / "stdout.log"), node.id))

    first = client.post(f"/approve/{node.id}")
    second = client.post(f"/approve/{node.id}")
    assert "validated and queued" in first.text
    assert "already submitted" in second.text
    assert conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 1


def test_approve_refuses_a_node_that_never_scored(client, app_env):
    _, conn, _, add = app_env
    node = add(status="error", cv_score=None, fold_scores=None, error_text="boom")
    r = client.post(f"/approve/{node.id}")
    assert "only a scored node" in r.text
    assert conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0


def test_approve_refuses_when_the_file_is_missing(client, app_env):
    _, conn, _, add = app_env
    node = add(cv_score=0.82)
    r = client.post(f"/approve/{node.id}")
    assert "no submission.csv" in r.text
    assert conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0


def test_the_viewer_opens_the_database_read_only(settings, app_env):
    """A wrong handler must not be able to corrupt a running search."""
    from kagglecracker.db.store import connect

    conn = connect(settings.db_path, read_only=True)
    with pytest.raises(Exception, match="readonly|read-only|attempt to write"):
        conn.execute("INSERT INTO protocols (metric_name, direction, fold_scheme, "
                     "n_folds, seed) VALUES ('x','higher_is_better','k',5,1)")
