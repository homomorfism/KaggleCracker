"""Where HW3's artifacts live: runs/hw3/, redirectable with KC_RUNS.

Two separate things write here — the schedule trigger's journal and channel
output, and the admin path's journal and state files — so the directory is
decided in ONE place. The env var mirrors the KC_WORKSPACE contract the tool
slices already follow: tests point KC_RUNS at a tmp dir and never append to the
real runs/hw3/.
"""

import os
from pathlib import Path


def hw3_dir():
    return Path(os.environ.get("KC_RUNS", "runs")) / "hw3"
