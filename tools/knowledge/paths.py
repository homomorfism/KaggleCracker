"""Workspace paths for the knowledge slice.

This mirrors tools/exec/experiments._workspace_root on purpose instead of
importing it: the plan requires each slice to be independently submittable, so
the knowledge slice must not fail to import because the exec slice is absent or
broken. The env-var contract (KC_WORKSPACE, redirected to a tmp dir by tests)
is the shared agreement; the one-line helper is duplicated, the contract is not.
"""

import os
from pathlib import Path


def _workspace_root():
    return Path(os.environ.get("KC_WORKSPACE", "workspace"))


def knowledge_dir():
    return _workspace_root() / "knowledge"


def search_log_path():
    # A durable record of every search the agent has run. The stall detector
    # reads this file rather than parsing the transcript, for the same reason
    # the exec slice's plateau detector reads leaderboard.jsonl: "are we going
    # in circles" is a question about what was actually searched and found.
    return knowledge_dir() / "search_log.jsonl"
