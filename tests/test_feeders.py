"""Context feeders.

The line these tests defend: the EDA script computes statistics and knows
nothing about the competition, while the agent supplies every interpretation.
If competition knowledge ever leaks into the script, the system stops
discovering the data and starts being told about it — and the score stops
meaning anything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kagglecracker.executors.fake import FakeExecutor, FakeOutcome
from kagglecracker.feeders.discussions import list_topics, run_discussions
from kagglecracker.feeders.eda import EDA_SCRIPT, run_eda
from kagglecracker.runtime.fake import FakeResponse, FakeRuntime

# --------------------------------------------------------------------------
# the script must stay generic
# --------------------------------------------------------------------------

COMPETITION_SPECIFIC = [
    "PassengerId", "Transported", "CryoSleep", "HomePlanet", "Cabin",
    "Destination", "VIP", "RoomService", "spaceship",
]


@pytest.mark.parametrize("token", COMPETITION_SPECIFIC)
def test_eda_script_names_no_competition_specific_column(token):
    """Hardcoding a column name here would be me doing the agent's work."""
    assert token.lower() not in EDA_SCRIPT.lower()


def test_eda_script_prescribes_no_treatment():
    """It reports statistics. What to *do* about them is the agent's call."""
    for token in ("fillna", "astype(\"category\")", "LGBMClassifier", "StratifiedGroupKFold"):
        assert token not in EDA_SCRIPT


def test_eda_script_derives_the_target_rather_than_being_told():
    assert "c not in test.columns" in EDA_SCRIPT


def test_eda_script_is_valid_python():
    import ast

    ast.parse(EDA_SCRIPT)


# --------------------------------------------------------------------------
# run_eda
# --------------------------------------------------------------------------


def test_eda_writes_findings_from_the_agents_summary(settings, tmp_path):
    executor = FakeExecutor(
        root=settings.runs_dir, default=FakeOutcome(stdout="## Shapes\ntrain (8, 3)")
    )
    runtime = FakeRuntime(default=FakeResponse(text="# EDA Findings\n\n- 8 rows"))
    result = run_eda(settings=settings, executor=executor, runtime=runtime)

    assert result.ok
    assert result.findings_path.read_text().startswith("# EDA Findings")
    assert "## Shapes" in result.raw_output


def test_eda_passes_the_raw_statistics_to_the_agent(settings):
    executor = FakeExecutor(root=settings.runs_dir, default=FakeOutcome(stdout="NULL_COUNT 217"))
    runtime = FakeRuntime()
    run_eda(settings=settings, executor=executor, runtime=runtime)
    _, prompt = runtime.calls[0]
    assert "NULL_COUNT 217" in prompt


def test_eda_reports_a_failed_script_rather_than_writing_junk(settings):
    executor = FakeExecutor(root=settings.runs_dir, default=FakeOutcome.crash("KeyError"))
    result = run_eda(settings=settings, executor=executor, runtime=FakeRuntime())
    assert not result.ok
    assert "EDA script failed" in result.error
    assert not (settings.context_dir / "eda_findings.md").exists()


def test_eda_reports_an_empty_summary_rather_than_writing_an_empty_file(settings):
    executor = FakeExecutor(root=settings.runs_dir, default=FakeOutcome(stdout="stats"))
    runtime = FakeRuntime(default=FakeResponse(text="   "))
    result = run_eda(settings=settings, executor=executor, runtime=runtime)
    assert not result.ok
    assert "returned nothing" in result.error


def test_eda_runs_in_the_sandbox_with_no_network(settings):
    """It reads the same read-only /data mount as every node."""
    executor = FakeExecutor(root=settings.runs_dir, default=FakeOutcome(stdout="x"))
    run_eda(settings=settings, executor=executor, runtime=FakeRuntime())
    node_id, code, _ = executor.calls[0]
    assert node_id == "eda"
    assert "/data" in code


# --------------------------------------------------------------------------
# discussions
# --------------------------------------------------------------------------


def test_topic_list_survives_a_title_containing_a_comma(monkeypatch):
    """csv.DictReader files overflow fields under the key None, so a row's keys
    are not all strings — the exact crash this hit on the first live run."""
    csv_out = (
        "id,title,authorName,commentCount\n"
        "309323,Welcome to the ship,,277\n"
        "429952,Tips, tricks and more,,97\n"
    )
    monkeypatch.setattr(
        "kagglecracker.feeders.discussions._run", lambda *a, **k: (0, csv_out, "")
    )
    topics = list_topics("comp")
    assert [t[0] for t in topics] == ["309323", "429952"]


def test_topic_list_raises_when_the_cli_fails(monkeypatch):
    monkeypatch.setattr(
        "kagglecracker.feeders.discussions._run", lambda *a, **k: (1, "", "401 unauthorized")
    )
    with pytest.raises(RuntimeError, match="listing topics failed"):
        list_topics("comp")


def test_discussions_writes_a_dated_summary(settings, monkeypatch):
    monkeypatch.setattr(
        "kagglecracker.feeders.discussions.list_topics",
        lambda comp, limit=15: [("1", "Winning approach")],
    )
    monkeypatch.setattr(
        "kagglecracker.feeders.discussions.fetch_topic",
        lambda tid, max_chars=6000: "LightGBM with group features scored 0.80",
    )
    runtime = FakeRuntime(default=FakeResponse(text="## Techniques\n- LightGBM 0.80"))
    result = run_discussions(settings=settings, runtime=runtime)

    assert result.ok
    assert result.path.parent.name == "discussions"
    assert result.path.suffix == ".md"
    assert "LightGBM" in result.path.read_text()


def test_discussions_reports_when_there_is_nothing_to_read(settings, monkeypatch):
    monkeypatch.setattr(
        "kagglecracker.feeders.discussions.list_topics", lambda comp, limit=15: []
    )
    result = run_discussions(settings=settings, runtime=FakeRuntime())
    assert not result.ok
    assert "no discussion topics" in result.error


def test_discussions_respects_the_total_character_budget(settings, monkeypatch):
    """Forum threads are long and every node pays for them."""
    monkeypatch.setattr(
        "kagglecracker.feeders.discussions.list_topics",
        lambda comp, limit=15: [(str(i), f"t{i}") for i in range(20)],
    )
    monkeypatch.setattr(
        "kagglecracker.feeders.discussions.fetch_topic",
        lambda tid, max_chars=6000: "x" * 5000,
    )
    runtime = FakeRuntime()
    run_discussions(settings=settings, runtime=runtime, max_total_chars=12_000)
    _, prompt = runtime.calls[0]
    assert len(prompt) < 20_000


# --------------------------------------------------------------------------
# the context reaches the prompt
# --------------------------------------------------------------------------


def test_feeder_output_lands_in_the_node_prompt(settings):
    settings.inject_eda = True
    settings.inject_discussions = True
    from kagglecracker.db.store import Protocol
    from kagglecracker.engine.prompts import PromptBuilder

    settings.context_dir.mkdir(parents=True, exist_ok=True)
    (settings.context_dir / "eda_findings.md").write_text("- 24% of rows have a null")
    (settings.context_dir / "discussions").mkdir(exist_ok=True)
    (settings.context_dir / "discussions" / "2026-08-01.md").write_text("- Cabin splits help")

    protocol = Protocol(1, "accuracy", "higher_is_better", "KFold", {}, 5, 42, "because")
    text = PromptBuilder(settings, protocol).render("draft", nodes=[])
    assert "24% of rows have a null" in text
    assert "Cabin splits help" in text


def test_pandas_is_not_a_host_dependency_of_the_feeders():
    """The EDA script runs in the sandbox; the host only orchestrates."""
    import kagglecracker.feeders.eda as eda

    assert not hasattr(eda, "pd")
    assert pd is not None  # the test file may use pandas; the module must not
