"""Where UI-created projects live.

Each project directory is shaped exactly like the repo's workspace/ (data/,
plans/, ...) so a run can point KC_WORKSPACE at it and every existing tool
works unchanged. KC_UI_PROJECTS relocates the root for tests, mirroring the
KC_WORKSPACE contract the tool slices already follow.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def projects_root():
    # Read at call time, not import time, so a test can redirect the root
    # after this module is already imported.
    override = os.environ.get("KC_UI_PROJECTS")
    if override:
        return Path(override)
    return REPO_ROOT / "workspace" / "projects"
