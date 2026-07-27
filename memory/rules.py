"""The always-on operating rules (HW2 memory layer).

rules.md is the markdown instruction layer: text pushed into the system prompt
on every run, unlike the SQLite and JSON stores which are pulled on demand.
load_rules just returns that text — resolved next to this module, not relative
to the current working directory, so it reads the same whatever the run's cwd.
"""

from pathlib import Path

# Beside this file on purpose: the rules ship with the code, so a run started
# from any directory still injects the same text.
_RULES_PATH = Path(__file__).with_name("rules.md")


def load_rules():
    """Return the operating-rules markdown as a string, to inject every run."""
    return _RULES_PATH.read_text(encoding="utf-8")
