"""A pushed operating rule changes the agent's output — the user never says it.

The same request runs twice against the same reactive model. The ONLY difference
is the system prompt: one run has the operating rules from memory/rules.md pushed
into it (that is what "push" context means — attached to every run), the other
does not. The user's message is identical in both and never mentions submissions,
approval, or CV scores.

  * WITHOUT the rule: the agent cheerfully offers to submit to the live
    competition straight away — an irreversible action, proposed with no CV
    score and no approval.
  * WITH the rule: the agent reads the pushed rules, QUOTES the governing lines
    (they are DATA it must honor, not prose to skim), states the CV score first,
    and refuses to submit without an explicit yes/y — exactly what the rules say.

The model is one deterministic, network-free function. It does not hardcode two
answers; it branches on whether the governing rule text is present in the pushed
system prompt, and quotes whatever governing lines it finds there.

Run from the repo root:  python examples/rule_changes_behavior.py
Transcript lands in runs/hw2/rule_changes.txt.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running as `python examples/...` from root

from core.loop import run_agent
from core.registry import Registry
from memory.rules import load_rules
from tests.fakemodel import Reply
from tools.memory import memtools

USER_REQUEST = "I've got a trained model ready. Should I submit it to Kaggle now?"
BEST_CV = 0.883  # a number the agent already has from an earlier experiment


def _mute(*a, **k):
    return None


def _governing_rules(messages):
    """Pull the rule lines from the pushed system prompt that bear on submitting.

    Returns the matching lines verbatim so the model can quote them. An empty
    list means no such rule was pushed into this run."""
    sys_text = "\n".join(
        m for m in messages if isinstance(m, str) and m.startswith("[SYSTEM]")
    )
    hits = []
    for line in sys_text.splitlines():
        low = line.lower()
        if "human approval" in low or "cv score" in low or "never submit" in low:
            line = line.strip().lstrip("0123456789. ").strip()
            # rules.md wraps each rule over several physical lines; keep just the
            # first sentence so a quoted line does not dangle a continuation word.
            if ". " in line:
                line = line.split(". ", 1)[0] + "."
            hits.append(line)
    return hits


def reactive_model(messages, tools):
    """One decision, branching only on whether governing rules were pushed in."""
    rules = _governing_rules(messages)
    if not rules:
        # No standing rules in this run: nothing tempers the request.
        return Reply(text=(
            "Sure — I'll go ahead and submit the current model to the live "
            "Kaggle competition now."
        ))
    quoted = "\n".join("    > %s" % r for r in rules)
    return Reply(text=(
        "Operating rules were attached to this run (the user did not restate "
        "them). The governing ones say:\n%s\n"
        "So, before any submission: the current best CV score is %.3f "
        "(balanced accuracy). I will NOT submit to the live competition without "
        "your explicit 'yes'/'y'. Do you approve submitting?" % (quoted, BEST_CV)
    ))


def run_once(system_prompt):
    """Run the same request under one system prompt. Returns (messages, final)."""
    reg = Registry()
    memtools.register(reg)  # tools exist; the point here is the pushed prompt
    messages = ["[SYSTEM] %s" % system_prompt, "[USER] %s" % USER_REQUEST]
    final = run_agent(messages, reactive_model, reg, max_steps=4,
                      input_fn=_mute, output_fn=_mute)
    return messages, final


def render(sections):
    lines = ["RULE CHANGES BEHAVIOR — a pushed rule the user never mentioned",
             "=" * 61,
             "Same request, same model. Only the pushed system prompt differs.",
             ""]
    for title, note, messages, final in sections:
        lines += [title, "-" * len(title), note, ""]
        for m in messages:
            if isinstance(m, str):
                # Keep the (possibly multi-line) system prompt readable.
                lines.append(m if not m.startswith("[SYSTEM]")
                             else m.replace("\n", "\n           "))
            elif m.tool_calls:
                calls = ", ".join("%s(%s)" % (c.name, c.args) for c in m.tool_calls)
                lines.append("[MODEL] act  -> %s" % calls)
            else:
                lines.append("[MODEL] say  -> %s" % m.text)
        lines += ["", "[FINAL] %s" % final, ""]
    return "\n".join(lines).rstrip() + "\n"


def main():
    m_without, f_without = run_once("You are the exec agent for a Kaggle project.")
    # WITH: the real rules.md, pushed in exactly as a run always would.
    m_with, f_with = run_once(
        "You are the exec agent for a Kaggle project.\n"
        "The following operating rules are always in effect:\n" + load_rules()
    )

    out = ROOT / "runs" / "hw2" / "rule_changes.txt"
    out.write_text(render([
        ("WITHOUT the rule pushed in",
         "No operating rules in the system prompt. The agent offers to submit "
         "immediately — irreversible, no score, no approval.",
         m_without, f_without),
        ("WITH memory/rules.md pushed in",
         "The same agent, now given the rules, quotes the governing lines, "
         "states the CV score, and refuses to submit without a yes/y.",
         m_with, f_with),
    ]))
    print("wrote %s" % out)
    print("  without rule -> %s" % f_without)
    print("  with rule    -> %s" % f_with)


if __name__ == "__main__":
    main()
