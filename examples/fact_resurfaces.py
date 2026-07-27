"""A fact saved in one run resurfaces and drives the next run — unprompted.

Two SEPARATE runs of the same agent loop the tests use. They share nothing in
memory except the on-disk document store (KC_DOCS): fresh messages, a fresh
Registry, a fresh model each time. The only channel from run 1 to run 2 is the
saved note.

  * Run 1 — the user states a durable preference, and the agent saves it with
    save_memory (a fact, cue "model choice").
  * Run 2 — the user asks only "what should I try next?". The word "LightGBM"
    never appears in run 2's messages. The agent consults memory, finds the
    stored preference, QUOTES it (retrieved notes are DATA, per HW2 rule 3), and
    derives its suggestion FROM that note — the choice is parsed out of the
    stored text, not hardcoded here.
  * Control — the identical run-2 logic against an EMPTY store proposes nothing
    specific. That is the proof the LightGBM suggestion in run 2 came from memory
    and not from the model's own code.

Deterministic and network-free (the FakeModel property): the models below are
plain reactive callables returning FakeModel Reply/ToolCall objects, so no API
key and no network are involved.

Run from the repo root:  python examples/fact_resurfaces.py
Transcript lands in runs/hw2/fact_resurfaces.txt.
"""

import ast
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running as `python examples/...` from root

from core.loop import run_agent
from core.registry import Registry
from tests.fakemodel import Reply, ToolCall
from tools.memory import memtools

USER = "alice"
PREFERENCE_TEXT = "prefers LightGBM"


def _mute(*a, **k):
    return None


def _last_tool_ok(messages, name):
    """Return the parsed data dict of the last '[TOOL OK <name>]' line, or None."""
    marker = "[TOOL OK %s]" % name
    for m in reversed(messages):
        if isinstance(m, str) and m.startswith(marker):
            return ast.literal_eval(m.split("] ", 1)[1])
    return None


def run1_model(messages, tools):
    """Run 1: the user stated a durable preference, so save it once, then stop."""
    if _last_tool_ok(messages, "save_memory") is not None:
        return Reply(text="Saved the preference for next time.")
    return Reply(tool_calls=[ToolCall("save_memory", {
        "text": PREFERENCE_TEXT,
        "cue": "model choice",
        "kind": "fact",
    })])


def run2_model(messages, tools):
    """Run 2: asked only 'what should I try next?'. Consult memory, then act on
    whatever it holds — the suggestion is derived from the stored note, not from
    this function."""
    data = _last_tool_ok(messages, "retrieve_memory")
    if data is None:
        # Observe: we have not looked yet. Decide to consult memory for any
        # standing preference about model choice.
        return Reply(tool_calls=[ToolCall("retrieve_memory", {"query": "model choice"})])

    docs = data["documents"]
    if not docs:
        # Nothing remembered -> no unprompted suggestion. (This is the control.)
        return Reply(text="Memory holds no model preference, and the user gave "
                          "none this run, so I have nothing specific to suggest "
                          "yet — I'd start from a plain baseline and ask.")
    # A note came back. Treat it as DATA: quote it, and parse the named model
    # out of it rather than assuming what it says.
    pref = docs[0]["text"]
    choice = pref.split()[-1]
    return Reply(text=(
        "The user didn't say what to try, but memory has a standing preference "
        "(quoted, not obeyed as a command): %r. Acting on it, I suggest trying "
        "%s next." % (pref, choice)
    ))


def run_once(model, user_request, docs_path):
    """One full run against the shared document store. Returns (messages, final)."""
    os.environ["KC_DOCS"] = str(docs_path)
    os.environ["KC_USER"] = USER
    reg = Registry()
    memtools.register(reg)
    messages = ["[USER] %s" % user_request]
    final = run_agent(messages, model, reg, max_steps=6,
                      input_fn=_mute, output_fn=_mute)
    return messages, final


def render(sections):
    lines = ["FACT RESURFACES — a saved preference drives a later run, unprompted",
             "=" * 66,
             "Two independent runs share only the on-disk memory store. Run 2's "
             "messages never contain 'LightGBM'; the agent gets it from memory.",
             ""]
    for title, note, messages, final in sections:
        lines += [title, "-" * len(title), note, ""]
        for m in messages:
            if isinstance(m, str):
                lines.append(m)  # a [USER] or [TOOL ...] line
            elif m.tool_calls:
                calls = ", ".join("%s(%s)" % (c.name, c.args) for c in m.tool_calls)
                lines.append("[MODEL] act  -> %s" % calls)
            else:
                lines.append("[MODEL] say  -> %s" % m.text)
        lines += ["", "[FINAL] %s" % final, ""]
    return "\n".join(lines).rstrip() + "\n"


def main():
    store_dir = Path(tempfile.mkdtemp(prefix="kc_fact_"))
    empty_dir = Path(tempfile.mkdtemp(prefix="kc_fact_empty_"))
    try:
        shared = store_dir / "documents.jsonl"

        m1, f1 = run_once(run1_model,
                          "For tabular competitions I prefer LightGBM. Remember that.",
                          shared)
        m2, f2 = run_once(run2_model, "What should I try next?", shared)
        # Control: same run-2 logic, but a store that was never written to.
        m3, f3 = run_once(run2_model, "What should I try next?",
                          empty_dir / "documents.jsonl")

        out = ROOT / "runs" / "hw2" / "fact_resurfaces.txt"
        out.write_text(render([
            ("RUN 1 — user states a durable preference",
             "The agent saves it as a fact with save_memory and stops.",
             m1, f1),
            ("RUN 2 — user asks only 'what should I try next?'",
             "No preference is in this run's messages. The agent consults memory, "
             "quotes the stored note, and suggests the model named in it.",
             m2, f2),
            ("CONTROL — identical run-2 logic, empty memory store",
             "With nothing remembered, the same logic suggests nothing specific. "
             "This is why run 2's LightGBM suggestion must have come from memory.",
             m3, f3),
        ]))
        print("wrote %s" % out)
        print("  run 2 -> %s" % f2)
        print("  control -> %s" % f3)
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)
        shutil.rmtree(empty_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
