"""The push side: rules and standing facts attached to every run unasked.

Small on purpose — push_context is wiring, not a tool — but two properties
are worth pinning: the seeded message really contains the rules AND the
facts, and the rules are re-read from the editable markdown file on every
call, so an admin's hand edit lands in the very next run with no restart.
"""

from tools.recon import context


def test_push_context_carries_the_rules_and_the_standing_facts():
    pushed = context.push_context()
    assert pushed.startswith("[SYSTEM]")
    # One recognizable rule, verbatim from memory/recon_rules.md:
    assert "Never recommend dropping a column because of drift alone" in pushed
    # And the day-0 facts a run must never re-ask:
    assert "health_condition" in pushed
    assert "balanced accuracy" in pushed


def test_a_hand_edit_to_the_rules_file_lands_on_the_next_call(tmp_path, monkeypatch):
    edited = tmp_path / "recon_rules.md"
    edited.write_text("# Recon operating rules\n1. HAND-EDITED-SENTINEL rule.\n")
    monkeypatch.setattr(context, "RULES_PATH", edited)
    assert "HAND-EDITED-SENTINEL" in context.push_context()
