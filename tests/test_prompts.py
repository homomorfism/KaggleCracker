"""Prompt assembly.

The point of these is drift: the output contract is a natural-language
requirement that no compiler checks, and it has to say the same thing in all
three actions. Asserting it per-action is the only thing standing between a
wording fix and two silently stale prompts.
"""

from __future__ import annotations

import pytest

from kagglecracker.db.store import Node, Protocol
from kagglecracker.engine.prompts import PromptBuilder, sandbox_package_versions

PROTOCOL = Protocol(
    id=1,
    metric_name="accuracy",
    direction="higher_is_better",
    fold_scheme="StratifiedGroupKFold",
    fold_params={"group_key": "PassengerId prefix"},
    n_folds=5,
    seed=42,
    rationale="Kaggle split train/test by party, so folds must be group-disjoint.",
)


def make_node(**kw) -> Node:
    base = dict(
        id=1, run_id=1, protocol_id=1, parent_id=None, action="draft", depth=0,
        code="print(1)", status="ok", error_text=None, cv_score=0.81,
        fold_scores=[0.80, 0.82, 0.81, 0.79, 0.83], stdout_path=None, prompt=None,
        prompt_chars=100, tokens_in=10, tokens_out=20, usd=0.001,
        created_at="2026-07-31",
    )
    return Node(**{**base, **kw})


@pytest.fixture
def builder(settings) -> PromptBuilder:
    settings.competition = "spaceship-titanic"
    return PromptBuilder(settings, PROTOCOL)


# --------------------------------------------------------------------------
# the shared blocks appear in every action — this is the anti-drift test
# --------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["draft", "improve", "debug"])
def test_every_action_carries_the_same_contract(builder, action):
    parent = make_node(status="error", error_text="boom") if action != "draft" else None
    text = builder.render(action, nodes=[], parent=parent)

    assert "metrics.json" in text
    assert "submission.csv" in text
    assert "index=False" in text
    assert "NO network access" in text
    assert "StratifiedGroupKFold" in text
    assert "/data" in text and "/workspace" in text


@pytest.mark.parametrize("action", ["draft", "improve", "debug"])
def test_every_action_states_the_time_budget(builder, action, settings):
    parent = make_node(status="error", error_text="boom") if action != "draft" else None
    text = builder.render(action, nodes=[], parent=parent)
    assert f"{settings.node_timeout_s}s" in text


def test_package_versions_come_from_the_lock_the_image_installs():
    """Told, not guessed — the sandbox has no network, so a model assuming an
    older API writes code nothing can fix."""
    versions = sandbox_package_versions()
    assert "lightgbm" in versions
    assert "pandas" in versions
    # A bare name with no version would mean the lock was not parsed.
    assert any(ch.isdigit() for ch in versions)


# --------------------------------------------------------------------------
# action blocks differ where they should
# --------------------------------------------------------------------------


def test_draft_asks_for_a_different_approach(builder):
    assert "DRAFT" in builder.render("draft", nodes=[])


def test_improve_includes_parent_code_and_score(builder):
    parent = make_node(code="import lightgbm", cv_score=0.815)
    text = builder.render("improve", nodes=[parent], parent=parent)
    assert "import lightgbm" in text
    assert "0.81500" in text
    assert "hypothesis" in text


def test_improve_without_parent_is_a_programming_error(builder):
    with pytest.raises(ValueError, match="improve requires a parent"):
        builder.render("improve", nodes=[])


def test_debug_of_a_crash_quotes_the_traceback(builder):
    parent = make_node(status="error", error_text="NameError: name 'lgb' is not defined")
    text = builder.render("debug", nodes=[parent], parent=parent)
    assert "NameError" in text
    assert "raised an exception" in text


def test_debug_of_a_timeout_says_there_is_no_bug_to_find(builder):
    """A killed process leaves no traceback. If this prompt read like the crash
    prompt, the model would hunt for a defect that does not exist."""
    parent = make_node(status="timeout", error_text="killed at the 900s limit")
    text = builder.render("debug", nodes=[parent], parent=parent)
    assert "KILLED" in text
    assert "no traceback" in text
    assert "Do not hunt for a bug" in text


def test_debug_of_a_contract_violation_says_the_model_may_be_fine(builder):
    parent = make_node(status="contract_violation", error_text="submission.csv columns differ")
    text = builder.render("debug", nodes=[parent], parent=parent)
    assert "output contract" in text
    assert "worth keeping" in text


def test_debug_prompts_differ_by_status(builder):
    """The whole reason RunStatus is an enum rather than an error string."""
    crash = builder.render(
        "debug", nodes=[], parent=make_node(status="error", error_text="x")
    )
    timeout = builder.render(
        "debug", nodes=[], parent=make_node(status="timeout", error_text="x")
    )
    assert crash != timeout


# --------------------------------------------------------------------------
# tree summary — the agent's only memory
# --------------------------------------------------------------------------


def test_empty_tree_says_so(builder):
    assert "Nothing has been tried yet" in builder.tree_summary([])


def test_tree_summary_ranks_by_protocol_direction(builder):
    nodes = [
        make_node(id=1, cv_score=0.70),
        make_node(id=2, cv_score=0.85),
        make_node(id=3, cv_score=0.80),
    ]
    summary = builder.tree_summary(nodes)
    assert summary.index("node 2") < summary.index("node 3") < summary.index("node 1")


def test_tree_summary_groups_repeated_failures(builder):
    """Without this, nodes 5, 9 and 11 die identically and node 17 learns nothing."""
    nodes = [
        make_node(id=i, status="error", cv_score=None, fold_scores=[],
                  error_text="MemoryError: out of memory")
        for i in range(1, 4)
    ]
    summary = builder.tree_summary(nodes)
    assert "MemoryError" in summary
    assert "x3" in summary
    assert "Do not reintroduce" in summary


def test_tree_summary_reports_when_nothing_has_scored(builder):
    nodes = [make_node(id=1, status="error", cv_score=None, fold_scores=[], error_text="e")]
    assert "No attempt has produced a valid score" in builder.tree_summary(nodes)


def test_tree_summary_shows_fold_spread(builder):
    nodes = [make_node(id=1, cv_score=0.81, fold_scores=[0.75, 0.87])]
    summary = builder.tree_summary(nodes)
    assert "0.7500" in summary and "0.8700" in summary


# --------------------------------------------------------------------------
# context budget
# --------------------------------------------------------------------------


def test_no_context_docs_yields_an_empty_block(builder):
    assert builder.context_docs() == ""


def test_context_docs_are_included_when_present(builder, settings):
    settings.inject_eda = True
    (settings.context_dir).mkdir(parents=True, exist_ok=True)
    (settings.context_dir / "eda_findings.md").write_text("CryoSleep is the strongest signal.")
    assert "CryoSleep is the strongest signal" in builder.context_docs()


def test_context_is_capped_by_the_char_budget(builder, settings):
    """Unbounded context is the real cost curve: every node pays for every doc."""
    settings.inject_eda = True
    settings.context_char_budget = 2000
    (settings.context_dir / "discussions").mkdir(parents=True, exist_ok=True)
    (settings.context_dir / "eda_findings.md").write_text("E" * 5000)
    out = builder.context_docs()
    assert len(out) < 3000
    assert "truncated to fit context budget" in out


def test_newest_discussions_survive_the_budget(builder, settings):
    settings.inject_discussions = True
    settings.context_char_budget = 3000
    d = settings.context_dir / "discussions"
    d.mkdir(parents=True, exist_ok=True)
    (d / "2026-01-01.md").write_text("OLD " * 300)
    (d / "2026-07-31.md").write_text("NEW " * 300)
    out = builder.context_docs()
    assert "2026-07-31" in out
