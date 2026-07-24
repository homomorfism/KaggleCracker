"""Workspace paths for the recon slice.

This mirrors tools/knowledge/paths.py on purpose instead of importing it (or
tools/exec/experiments._workspace_root): each slice must be independently
submittable, so recon must not fail to import because a sibling slice is
absent or broken. The env-var contract (KC_WORKSPACE, redirected to a tmp dir
by tests) is the shared agreement; the one-line helper is duplicated, the
contract is not.
"""

import os
from pathlib import Path


def _workspace_root():
    return Path(os.environ.get("KC_WORKSPACE", "workspace"))


def data_dir():
    return _workspace_root() / "data"


def plans_dir():
    return _workspace_root() / "plans"


def plan_path(dataset):
    # One plan per dataset, named after it: plans/train.csv.plan.md. The
    # stopping condition and both plan tools all resolve the path through this
    # one function, so "does the plan exist" means the same thing everywhere.
    return plans_dir() / (dataset + ".plan.md")
