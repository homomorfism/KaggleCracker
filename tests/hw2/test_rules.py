"""Test for the always-on operating rules (memory/rules.py)."""

from memory import rules


def test_load_rules_returns_the_human_approval_rule():
    text = rules.load_rules()
    assert text.strip()                    # non-empty
    assert "human approval" in text        # the irreversible-action rule is present
