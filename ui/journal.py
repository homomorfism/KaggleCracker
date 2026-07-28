"""The append-only run journal: the contract between the agent loop and the UI.

The loop side writes one JSON object per line as events happen; the UI side
polls the file and asks for "everything after seq N". A file rather than a
socket on purpose: a finished run is automatically a replayable recording, and
the UI needs no live process to render one — the journal IS the run.
"""

import json
import time


class JournalWriter:
    """Writes events as JSON lines, flushing each so a poller sees it
    immediately. seq is dense and per-file, so the reader can resume from any
    point with a single integer cursor."""

    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w")
        self._seq = 0

    def write(self, event_type, **fields):
        self._seq += 1
        event = {"seq": self._seq, "ts": round(time.time(), 3), "type": event_type}
        event.update(fields)
        self._f.write(json.dumps(event) + "\n")
        self._f.flush()
        return event

    def close(self):
        self._f.close()


def read_events(path, since=0):
    """Events with seq > since, in order.

    A torn final line (the writer caught mid-append) fails to parse; reading
    stops there and the poller picks the completed line up next time. That is
    the one JSON error deliberately not raised: it is a normal consequence of
    reading a file someone else is writing, not corruption.
    """
    events = []
    try:
        f = path.open()
    except FileNotFoundError:
        return events
    with f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                break
            if event["seq"] > since:
                events.append(event)
    return events


def run_status(path):
    """'running', 'finished' or 'failed', judged by the journal's last event.

    A missing or still-empty journal reads as 'running': the subprocess exists
    before its first event lands, and the UI treating that gap as running is
    correct — it just keeps polling.
    """
    events = read_events(path)
    if not events:
        return "running"
    last = events[-1]["type"]
    if last == "run_finished":
        return "finished"
    if last == "run_failed":
        return "failed"
    return "running"
