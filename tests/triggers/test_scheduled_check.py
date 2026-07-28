"""Tests for the schedule trigger.

Every test asserts on the consequence on disk, not just the returned record: a
post means a line in channel_out.jsonl and an advanced last_seen.json; silence
means NO line in channel_out.jsonl and a journal line explaining itself. The
loop tests pin max_ticks with a fake sleep, so nothing here spends wall-clock
time.
"""

import ast
import json

import pytest

from triggers import scheduled_check


def _mute(*a, **k):
    return None


def _no_human(*a, **k):
    # Patched over builtins.input: nothing in a scheduled run may consult a
    # person, so an attempt raises here instead of blocking on stdin.
    raise AssertionError("the trigger asked a human for input")


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Redirect both the workspace (leaderboard) and runs/ (trigger outputs) at
    tmp dirs. Returns (workspace, runs/hw3)."""
    ws = tmp_path / "workspace"
    runs = tmp_path / "runs"
    monkeypatch.setenv("KC_WORKSPACE", str(ws))
    monkeypatch.setenv("KC_RUNS", str(runs))
    return ws, runs / "hw3"


def _seed_leaderboard(ws, scores):
    """One leaderboard row per score, in the shape record_experiment_result writes."""
    exp = ws / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps({"experiment_id": "e%d" % i, "cv_score": s, "fold_scores": [], "notes": ""})
        for i, s in enumerate(scores)
    ]
    (exp / "leaderboard.jsonl").write_text("".join(r + "\n" for r in rows))


def _lines(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _seed_last_seen(hw3, best):
    """Pre-set the high-water mark, as a previous run would have left it."""
    hw3.mkdir(parents=True, exist_ok=True)
    (hw3 / "last_seen.json").write_text(json.dumps({"best": best}) + "\n")


# --- the posting branch ------------------------------------------------------


def test_posts_on_the_first_recorded_score(dirs):
    ws, hw3 = dirs
    _seed_leaderboard(ws, [0.80])

    rec = scheduled_check.check_once(1, output_fn=_mute)

    assert rec["decision"] == "posted"
    posts = _lines(hw3 / "channel_out.jsonl")
    assert len(posts) == 1
    assert "0.8" in posts[0]["text"]                       # the message names the new best
    assert json.loads((hw3 / "last_seen.json").read_text())["best"] == 0.80
    assert _lines(hw3 / "scheduled_journal.jsonl") == []   # a post is not a silence


def test_posts_only_when_the_best_actually_improves(dirs):
    ws, hw3 = dirs
    _seed_leaderboard(ws, [0.80])
    scheduled_check.check_once(1, output_fn=_mute)

    _seed_leaderboard(ws, [0.80, 0.85])
    rec = scheduled_check.check_once(2, output_fn=_mute)

    assert rec["decision"] == "posted"
    posts = _lines(hw3 / "channel_out.jsonl")
    assert len(posts) == 2
    assert "0.85" in posts[1]["text"] and "0.8" in posts[1]["text"]  # new and previous
    assert json.loads((hw3 / "last_seen.json").read_text())["best"] == 0.85


# --- the silence branch ------------------------------------------------------


def test_silent_when_the_best_has_not_moved(dirs):
    ws, hw3 = dirs
    _seed_leaderboard(ws, [0.80])
    scheduled_check.check_once(1, output_fn=_mute)   # posts, sets the mark

    rec = scheduled_check.check_once(2, output_fn=_mute)

    assert rec["decision"] == "silent"
    assert len(_lines(hw3 / "channel_out.jsonl")) == 1   # nothing new was said
    journal = _lines(hw3 / "scheduled_journal.jsonl")
    assert journal == [{"tick": 2, "decision": "silent",
                        "reason": "no CV improvement over best 0.8", "best_seen": 0.80}]
    # And the mark did not move on a silent tick.
    assert json.loads((hw3 / "last_seen.json").read_text())["best"] == 0.80


def test_silent_when_new_scores_are_worse(dirs):
    ws, hw3 = dirs
    _seed_leaderboard(ws, [0.90])
    scheduled_check.check_once(1, output_fn=_mute)

    _seed_leaderboard(ws, [0.90, 0.85, 0.70])   # more work, no progress
    rec = scheduled_check.check_once(2, output_fn=_mute)

    assert rec["decision"] == "silent"
    assert len(_lines(hw3 / "channel_out.jsonl")) == 1
    assert _lines(hw3 / "scheduled_journal.jsonl")[0]["best_seen"] == 0.90


def test_silent_with_an_empty_leaderboard(dirs):
    _ws, hw3 = dirs

    rec = scheduled_check.check_once(1, output_fn=_mute)

    assert rec["decision"] == "silent"
    assert rec["reason"] == "no CV scores recorded yet"   # distinct from "no improvement"
    assert not (hw3 / "channel_out.jsonl").exists()      # never posted at all
    assert not (hw3 / "last_seen.json").exists()         # no mark invented from nothing
    assert _lines(hw3 / "scheduled_journal.jsonl")[0]["best_seen"] is None


def test_non_numeric_rows_never_become_a_score(dirs):
    ws, hw3 = dirs
    exp = ws / "experiments"
    exp.mkdir(parents=True)
    (exp / "leaderboard.jsonl").write_text(
        "".join(json.dumps({"experiment_id": "e", "cv_score": s}) + "\n"
                for s in (True, False, None, "0.99"))
    )

    rec = scheduled_check.check_once(1, output_fn=_mute)

    # True must not be read as a 1.0 best score, and no string is parsed into one.
    assert rec["decision"] == "silent"
    assert rec["reason"] == "no CV scores recorded yet"
    assert not (hw3 / "channel_out.jsonl").exists()


def test_respects_lower_is_better_direction(dirs, monkeypatch):
    ws, hw3 = dirs
    monkeypatch.setattr("tools.exec.experiments.HIGHER_IS_BETTER", False)

    _seed_leaderboard(ws, [0.30])
    scheduled_check.check_once(1, output_fn=_mute)
    assert json.loads((hw3 / "last_seen.json").read_text())["best"] == 0.30

    # Lower-is-better: a smaller score IS the improvement.
    _seed_leaderboard(ws, [0.30, 0.20])
    assert scheduled_check.check_once(2, output_fn=_mute)["decision"] == "posted"
    # ...and a bigger one is not.
    _seed_leaderboard(ws, [0.30, 0.20, 0.25])
    assert scheduled_check.check_once(3, output_fn=_mute)["decision"] == "silent"
    assert len(_lines(hw3 / "channel_out.jsonl")) == 2


def test_an_equal_score_is_not_news(dirs):
    ws, hw3 = dirs
    _seed_leaderboard(ws, [0.80])
    scheduled_check.check_once(1, output_fn=_mute)

    _seed_leaderboard(ws, [0.80, 0.80])   # a tie, not an improvement
    assert scheduled_check.check_once(2, output_fn=_mute)["decision"] == "silent"
    assert len(_lines(hw3 / "channel_out.jsonl")) == 1


# --- the interval loop -------------------------------------------------------


def test_loop_stops_at_max_ticks_and_sleeps_between(dirs):
    slept = []
    seen = []

    def fake_tick(n, output_fn=None):
        seen.append(n)

    ticks = scheduled_check.run_on_schedule(
        900, max_ticks=3, tick=fake_tick, sleep=slept.append, output_fn=_mute)

    assert ticks == 3
    assert seen == [1, 2, 3]        # ticks are numbered for the journal
    assert slept == [900, 900]      # between ticks only: three ticks, two sleeps
    # No wall-clock time was spent: the fake sleep only recorded the interval.


def test_first_tick_happens_before_any_sleep(dirs):
    order = []

    def fake_tick(n, output_fn=None):
        order.append("tick")

    scheduled_check.run_on_schedule(
        900, max_ticks=1, tick=fake_tick,
        sleep=lambda s: order.append("sleep"), output_fn=_mute)

    assert order == ["tick"]   # a fresh start looks immediately and never sleeps first


def test_loop_journals_one_silent_line_per_quiet_tick(dirs):
    ws, hw3 = dirs
    _seed_leaderboard(ws, [0.80])

    # Tick 1 posts the first score; ticks 2 and 3 have nothing new to say.
    scheduled_check.run_on_schedule(60, max_ticks=3, sleep=_mute, output_fn=_mute)

    assert len(_lines(hw3 / "channel_out.jsonl")) == 1
    journal = _lines(hw3 / "scheduled_journal.jsonl")
    assert [r["tick"] for r in journal] == [2, 3]
    assert all(r["decision"] == "silent" for r in journal)
    assert all("no CV improvement over best 0.8" == r["reason"] for r in journal)


# --- started by the clock, not by a message ---------------------------------


def test_one_tick_posts_a_new_best_with_no_human_involved(dirs, monkeypatch):
    """The trigger fires on its own clock and speaks without being asked.

    Nothing hands it a message, a prompt or a transcript: run_on_schedule takes
    an interval and a tick count, and the only reason anything happens is that a
    tick came round. builtins.input is booby-trapped, so a run that tried to
    consult a person would fail here rather than quietly block.
    """
    ws, hw3 = dirs
    _seed_leaderboard(ws, [0.80, 0.91])   # a new best is already sitting there
    monkeypatch.setattr("builtins.input", _no_human)

    ticks = scheduled_check.run_on_schedule(
        900, max_ticks=1, sleep=_no_human, output_fn=_mute)

    assert ticks == 1                     # exactly one tick, and it never slept
    posts = _lines(hw3 / "channel_out.jsonl")
    assert len(posts) == 1                # a post was emitted, unprompted
    assert "0.91" in posts[0]["text"]     # naming the new best CV
    assert _lines(hw3 / "scheduled_journal.jsonl") == []   # speaking is not silence


def test_one_tick_with_no_improvement_records_its_silence(dirs, monkeypatch):
    """Silence is a decision the trigger writes down, not an absence of one.

    The mark is already at 0.80 and the leaderboard has nothing better, so this
    tick has nothing to say. The consequence is two-sided: no post exists at
    all, and the journal carries a line explaining why it stayed quiet.
    """
    ws, hw3 = dirs
    _seed_last_seen(hw3, 0.80)            # 0.80 was announced by an earlier run
    _seed_leaderboard(ws, [0.80, 0.79])   # more work since, but no improvement
    monkeypatch.setattr("builtins.input", _no_human)

    ticks = scheduled_check.run_on_schedule(
        900, max_ticks=1, sleep=_no_human, output_fn=_mute)

    assert ticks == 1
    # NO post: not an empty file, no file at all — nothing was ever said.
    assert not (hw3 / "channel_out.jsonl").exists()

    journal = _lines(hw3 / "scheduled_journal.jsonl")
    assert len(journal) == 1
    assert journal[0]["tick"] == 1
    assert journal[0]["decision"] == "silent"
    assert journal[0]["best_seen"] == 0.80
    reason = journal[0]["reason"]
    assert isinstance(reason, str) and reason.strip()      # a real reason string
    assert reason == "no CV improvement over best 0.8"

    # And the silent tick left the mark alone, so it is still not news tomorrow.
    assert json.loads((hw3 / "last_seen.json").read_text())["best"] == 0.80


def test_trigger_never_reaches_the_submit_machinery():
    # This trigger observes and posts. If it ever grows an import of the registry,
    # the gate, or the agent loop, a scheduled wake-up could reach the submit tool
    # and spend one of the day's capped live slots with no human involved. Read
    # the real imports rather than grepping the text, so the docstring stays free
    # to explain the rule by name.
    tree = ast.parse(open(scheduled_check.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
            imported.update("%s.%s" % (node.module, a.name) for a in node.names)

    assert not any(m.startswith("core") for m in imported)    # no gate, loop or registry
    assert not any("submit" in m for m in imported)
    assert not any(m.startswith("agents") for m in imported)
