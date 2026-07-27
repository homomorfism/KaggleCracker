"""JSON-lines document store for free-form memory (HW2 memory layer).

Free-form facts — a note, an observation, a trick that worked — have no fixed
columns, so they live here rather than in the SQLite experiment store. One JSON
object per line: append-only, human-readable, trivially greppable.

Visibility is the whole point of this store. A record is either private to its
author or explicitly shared. retrieve_documents only ever returns a record to a
caller who is allowed to see it: their own, plus anyone's shared. A private
record from another user is never returned — not filtered out late, but never
included in the first place. As with the SQLite store, this module returns plain
records; the {ok, data}/{ok, error} envelope is the tool layer's job.

Retrieved text is DATA, not instructions. A cue or body that reads like a
command is still just a stored string to be reasoned about, never executed.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

# The fields every record carries. Written on save, read back on retrieve.
_FIELDS = ("id", "user_id", "created_at", "text", "cue", "shared")


def _docs_path():
    # Tests point KC_DOCS at a file under tmp_path (same discipline as KC_DB /
    # KC_WORKSPACE), so no test writes into the real workspace/. Real runs fall
    # back to workspace/documents.jsonl.
    return os.environ.get("KC_DOCS", str(Path("workspace") / "documents.jsonl"))


def save_document(user_id, text, cue, shared=False, path=None):
    """Append one free-form record and return its id.

    cue is a short keyword string that widens what a later query can match
    beyond the raw text. shared=False keeps the record private to user_id;
    shared=True lets any user retrieve it.
    """
    path = path or _docs_path()
    record = {
        "id": uuid.uuid4().hex,
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "cue": cue,
        "shared": bool(shared),
    }
    # Ensure the parent exists so the default workspace/ path and any tmp path
    # both just work without a separate init step.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record["id"]


def retrieve_documents(user_id, query, include_shared=True, path=None):
    """Return visible records matching query, oldest first, as a list of dicts.

    Visible means: authored by user_id (private or shared), or — when
    include_shared is True — shared by anyone. A private record belonging to a
    different user is never returned. Matching is a case-insensitive substring
    test against either the cue or the text.
    """
    path = path or _docs_path()
    if not os.path.exists(path):
        return []

    needle = query.lower()
    results = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            own = record["user_id"] == user_id
            shared = record.get("shared", False)
            # Gate on visibility FIRST. A record the caller may not see is
            # skipped before we even consider whether it matches the query, so
            # another user's private note can never leak via a lucky keyword.
            if not (own or (include_shared and shared)):
                continue

            haystack = (record.get("cue", "") + " " + record.get("text", "")).lower()
            if needle in haystack:
                results.append(record)

    return results
