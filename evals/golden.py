"""Load the eval set and resolve content anchors to live chunk ids.

Cases store ANCHORS — (path, contains-substring) pairs — not chunk ids,
because ordinal ids move whenever a document is re-chunked. Resolution
happens at load time against the current index; an anchor that matches
nothing is a loud error (a broken anchor is a harness bug, and scoring
against a silently-empty golden set would flatter every retriever).

Deliberately-unanswerable cases have `anchors: []` and resolve to an empty
golden set — correct, not missing data.
"""

import json
from pathlib import Path

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"


def load_cases(path=CASES_PATH):
    cases = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def resolve_golden(case, index):
    """Anchor list -> set of chunk ids in the given index."""
    golden = set()
    for anchor in case["anchors"]:
        needle = anchor["contains"].lower()
        matched = [
            c["id"]
            for c in index.chunks
            if c["path"] == anchor["path"] and needle in c["text"].lower()
        ]
        if not matched:
            raise ValueError(
                "case %r: anchor %r matched no chunk — fix the anchor"
                % (case["id"], anchor)
            )
        golden.update(matched)
    return golden
