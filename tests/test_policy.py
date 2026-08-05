"""Search policy and ranking.

The policy must be *total*: there is no tree state in which it has nothing to
do, or the loop stalls. Both fallbacks land on `draft` because draft needs no
parent — these tests exist to prove that holds for every degenerate tree.
"""

from __future__ import annotations

import pytest

from kagglecracker.db.store import NodeStore, ProtocolStore, init_db
from kagglecracker.engine.policy import SearchPolicy
from kagglecracker.engine.ranking import best_node, leaderboard, most_recent_failure


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "policy.db")


@pytest.fixture
def protocol(conn):
    return ProtocolStore(conn).set_current(
        metric_name="accuracy",
        direction="higher_is_better",
        fold_scheme="StratifiedGroupKFold",
        n_folds=5,
        seed=42,
    )


@pytest.fixture
def nodes(conn, protocol):
    store = NodeStore(conn)
    run_id = conn.execute(
        "INSERT INTO runs (protocol_id, max_nodes, max_wall_clock_s, max_usd, "
        "max_consecutive_failures, max_consecutive_infra_errors) VALUES (?,20,3600,5.0,5,3)",
        (protocol.id,),
    ).lastrowid

    def add(**kw):
        base = dict(
            run_id=run_id, protocol_id=protocol.id, parent_id=None, action="draft",
            depth=0, code="print(1)", status="ok", cv_score=0.8,
            fold_scores=[0.8] * 5,
        )
        return store.insert(**{**base, **kw})

    return add


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------


def test_best_node_respects_higher_is_better(conn, protocol, nodes):
    nodes(cv_score=0.70)
    win = nodes(cv_score=0.85)
    nodes(cv_score=0.80)
    assert best_node(conn, protocol).id == win.id


def test_best_node_respects_lower_is_better(conn, nodes):
    store = ProtocolStore(conn)
    # Seed nodes under the current protocol first, then retire it.
    nodes(cv_score=0.70)
    nodes(cv_score=0.85)
    p = store.current()
    conn.execute("UPDATE protocols SET direction = 'lower_is_better' WHERE id = ?", (p.id,))
    lower = store.current()
    assert best_node(conn, lower).cv_score == pytest.approx(0.70)


def test_failed_nodes_are_never_best(conn, protocol, nodes):
    nodes(cv_score=0.99, status="error")
    good = nodes(cv_score=0.50)
    assert best_node(conn, protocol).id == good.id


def test_ranking_excludes_retired_protocols(conn, protocol, nodes):
    """The staleness rule, as a WHERE clause rather than a paragraph of prose."""
    nodes(cv_score=0.95)  # scored under the original protocol
    new = ProtocolStore(conn).set_current(
        metric_name="accuracy", direction="higher_is_better",
        fold_scheme="GroupKFold", n_folds=5, seed=42,
    )
    assert best_node(conn, new) is None
    assert leaderboard(conn, new) == []
    # The old node still exists and still ranks under its own protocol.
    assert best_node(conn, protocol).cv_score == pytest.approx(0.95)


def test_empty_tree_has_no_best(conn, protocol):
    assert best_node(conn, protocol) is None
    assert leaderboard(conn, protocol) == []


def test_ties_break_deterministically_by_id(conn, protocol, nodes):
    first = nodes(cv_score=0.80)
    nodes(cv_score=0.80)
    assert best_node(conn, protocol).id == first.id


def test_leaderboard_reports_fold_spread(conn, protocol, nodes):
    nodes(cv_score=0.81, fold_scores=[0.75, 0.79, 0.81, 0.83, 0.87])
    row = leaderboard(conn, protocol)[0]
    assert row.fold_min == pytest.approx(0.75)
    assert row.fold_std > 0.04


def test_depth_cap_filters_best_node(conn, protocol, nodes):
    nodes(cv_score=0.90, depth=5)
    shallow = nodes(cv_score=0.70, depth=1)
    assert best_node(conn, protocol).cv_score == pytest.approx(0.90)
    assert best_node(conn, protocol, max_depth=5).id == shallow.id


def test_most_recent_failure_ignores_infra_and_truncated(conn, protocol, nodes):
    """Neither is a defect in the generated code, so neither is debuggable."""
    real = nodes(status="error", cv_score=None, fold_scores=None, error_text="boom")
    nodes(status="infra_error", cv_score=None, fold_scores=None, error_text="daemon down")
    nodes(status="truncated", cv_score=None, fold_scores=None, error_text="ceiling")
    assert most_recent_failure(conn, protocol).id == real.id


def test_most_recent_failure_prefers_the_newest(conn, protocol, nodes):
    nodes(status="error", cv_score=None, fold_scores=None, error_text="old")
    newest = nodes(status="timeout", cv_score=None, fold_scores=None, error_text="new")
    assert most_recent_failure(conn, protocol).id == newest.id


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


def test_first_steps_are_always_drafts(conn, protocol, nodes):
    nodes(cv_score=0.9)  # even with a great node available
    policy = SearchPolicy(protocol, seed_drafts=3)
    for step in range(3):
        assert policy.decide(conn, step=step).action == "draft"


def test_improve_falls_back_to_draft_when_nothing_has_scored(conn, protocol):
    """Fallback (a). A tree of pure failures must still make progress."""
    policy = SearchPolicy(protocol, p_improve=1.0, p_draft=0.0, p_debug=0.0, seed_drafts=0)
    d = policy.decide(conn, step=10)
    assert d.action == "draft"
    assert "nothing has scored" in d.reason


def test_debug_falls_back_to_draft_when_there_is_no_failure(conn, protocol, nodes):
    """Fallback (a), the other half."""
    nodes(cv_score=0.8)
    policy = SearchPolicy(protocol, p_improve=0.0, p_draft=0.0, p_debug=1.0, seed_drafts=0)
    d = policy.decide(conn, step=10)
    assert d.action == "draft"
    assert "no repairable failure" in d.reason


def test_improve_targets_a_node_below_the_depth_cap(conn, protocol, nodes):
    """Fallback (b): selecting the global best would pick a node we cannot extend."""
    nodes(cv_score=0.95, depth=5)
    shallow = nodes(cv_score=0.60, depth=2)
    policy = SearchPolicy(
        protocol, p_improve=1.0, p_draft=0.0, p_debug=0.0, seed_drafts=0, max_depth=5
    )
    d = policy.decide(conn, step=10)
    assert d.action == "improve"
    assert d.parent.id == shallow.id


def test_improve_falls_back_to_draft_at_the_depth_cap(conn, protocol, nodes):
    nodes(cv_score=0.95, depth=5)
    policy = SearchPolicy(
        protocol, p_improve=1.0, p_draft=0.0, p_debug=0.0, seed_drafts=0, max_depth=5
    )
    d = policy.decide(conn, step=10)
    assert d.action == "draft"
    assert "depth cap" in d.reason


def test_the_two_fallbacks_are_distinguishable_in_the_log(conn, protocol, nodes):
    """Same action, very different diagnoses — the reason string has to say which."""
    policy = SearchPolicy(protocol, p_improve=1.0, p_draft=0.0, p_debug=0.0, seed_drafts=0)
    empty = policy.decide(conn, step=10).reason
    nodes(cv_score=0.9, depth=5)
    capped = policy.decide(conn, step=10).reason
    assert empty != capped


def test_decisions_are_reproducible_for_a_given_seed(conn, protocol, nodes):
    nodes(cv_score=0.8)
    nodes(status="error", cv_score=None, fold_scores=None, error_text="e")
    a = [SearchPolicy(protocol, seed=7).decide(conn, step=s).action for s in range(20)]
    b = [SearchPolicy(protocol, seed=7).decide(conn, step=s).action for s in range(20)]
    assert a == b


def test_a_different_seed_gives_a_different_search(conn, protocol, nodes):
    nodes(cv_score=0.8)
    nodes(status="error", cv_score=None, fold_scores=None, error_text="e")
    a = [SearchPolicy(protocol, seed=1).decide(conn, step=s).action for s in range(30)]
    b = [SearchPolicy(protocol, seed=99).decide(conn, step=s).action for s in range(30)]
    assert a != b


def test_reseeding_by_step_makes_resume_match_an_uninterrupted_run(conn, protocol, nodes):
    """A resumed run must not diverge just because it restarted at step 12."""
    nodes(cv_score=0.8)
    nodes(status="error", cv_score=None, fold_scores=None, error_text="e")
    uninterrupted = SearchPolicy(protocol, seed=5)
    full = [uninterrupted.decide(conn, step=s).action for s in range(20)]

    resumed = SearchPolicy(protocol, seed=5)
    tail = [resumed.decide(conn, step=s).action for s in range(12, 20)]
    assert tail == full[12:]


def test_probabilities_must_sum_to_one():
    from kagglecracker.db.store import Protocol

    p = Protocol(1, "accuracy", "higher_is_better", "KFold", {}, 5, 42, "")
    with pytest.raises(ValueError, match="sum to 1.0"):
        SearchPolicy(p, p_improve=0.5, p_draft=0.5, p_debug=0.5)


def test_action_mix_is_roughly_as_configured(conn, protocol, nodes):
    """Not a distributional proof — just a guard against wiring the branches up wrong."""
    nodes(cv_score=0.8)
    nodes(status="error", cv_score=None, fold_scores=None, error_text="e")
    policy = SearchPolicy(protocol, seed=3, seed_drafts=0)
    actions = [policy.decide(conn, step=s).action for s in range(400)]
    assert 0.55 < actions.count("improve") / len(actions) < 0.85
    assert actions.count("debug") > 20
    assert actions.count("draft") > 20
