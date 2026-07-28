from ui.journal import JournalWriter, read_events, run_status


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "journal.jsonl"
    writer = JournalWriter(path)
    writer.write("run_started", prompt="TASK")
    writer.write("reason", text="thinking")
    writer.close()

    events = read_events(path)
    assert [e["type"] for e in events] == ["run_started", "reason"]
    assert [e["seq"] for e in events] == [1, 2]
    assert events[1]["text"] == "thinking"


def test_since_cursor_returns_only_new_events(tmp_path):
    path = tmp_path / "journal.jsonl"
    writer = JournalWriter(path)
    writer.write("run_started")
    writer.write("reason", text="a")
    writer.close()

    assert [e["type"] for e in read_events(path, since=1)] == ["reason"]
    assert read_events(path, since=2) == []


def test_missing_journal_reads_empty_and_running(tmp_path):
    path = tmp_path / "nope.jsonl"
    assert read_events(path) == []
    assert run_status(path) == "running"


def test_torn_last_line_is_held_back_not_crashed(tmp_path):
    path = tmp_path / "journal.jsonl"
    writer = JournalWriter(path)
    writer.write("run_started")
    writer.close()
    # Simulate a poller catching the writer mid-append.
    with path.open("a") as f:
        f.write('{"seq": 2, "ty')

    events = read_events(path)
    assert [e["seq"] for e in events] == [1]


def test_run_status_follows_the_last_event(tmp_path):
    path = tmp_path / "journal.jsonl"
    writer = JournalWriter(path)
    writer.write("run_started")
    writer.write("reason", text="a")
    assert run_status(path) == "running"
    writer.write("run_finished", text="done")
    writer.close()
    assert run_status(path) == "finished"

    failed = tmp_path / "failed.jsonl"
    writer = JournalWriter(failed)
    writer.write("run_started")
    writer.write("run_failed", reason="boom")
    writer.close()
    assert run_status(failed) == "failed"
