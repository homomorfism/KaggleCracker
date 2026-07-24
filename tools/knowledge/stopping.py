"""A stopping condition for run_agent: stop once mining has stalled.

stall_detector builds a should_stop(state, messages) callable that run_agent
polls at the top of each step. Like the exec slice's plateau detector, it
ignores the loop state and the transcript and reads the durable record instead
— workspace/knowledge/search_log.jsonl, written by search_kaggle_discussions —
because "are we going in circles" is a question about what was actually
searched and found, not about what the model just said.

Two independent stall signals, either one fires:

  * the same query was issued twice in a row — the model is repeating itself;
  * the last two searches surfaced no URL that any earlier search had not
    already found — the corpus is mined out for now.
"""

import json

from tools.knowledge.paths import search_log_path


def stall_detector():
    """Return a should_stop callable for run_agent. Returns a message when the
    mining has stalled, None while it is still finding new ground."""

    def should_stop(state, messages):
        # Both parameters are unused on purpose — they exist to match how
        # run_agent invokes its stopping condition; the signal lives in the log.
        entries = _log_entries()
        if len(entries) < 2:
            return None

        prev, last = entries[-2], entries[-1]
        if (
            prev["query"].strip().lower() == last["query"].strip().lower()
            and prev.get("slug") == last.get("slug")
        ):
            return "stall: the same query was issued twice (%r); summarise what is known" % last["query"]

        # Replay the log and note what each search added that no earlier search
        # had found. Two consecutive empty contributions mean the well is dry.
        seen = set()
        fresh = []
        for entry in entries:
            urls = set(entry.get("urls", ()))
            fresh.append(urls - seen)
            seen |= urls
        if not fresh[-1] and not fresh[-2]:
            return "stall: the last two searches surfaced no new sources; summarise what is known"

        return None

    return should_stop


def _log_entries():
    """Every logged search, in order. A missing log means nothing has been
    searched yet -> empty list. The lines are trusted to be valid JSON because
    they are written by search_kaggle_discussions — a corrupt line is a real
    problem to surface, not to silently swallow."""
    path = search_log_path()
    if not path.exists():
        return []
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
