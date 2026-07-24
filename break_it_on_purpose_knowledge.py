"""Regenerate the knowledge slice's two comparison transcripts (requirement 4).

The same scenario runs twice against an HTTP layer that always answers 429:

  * WITH the error branch: the real search_kaggle_discussions returns a
    rate_limited envelope, the loop takes its error branch, and the agent says
    the corpus state is UNKNOWN and it must retry later.
  * WITHOUT the error branch: a naive search catches the 429, hands back an
    empty list as if it were data, and the agent concludes with total
    confidence that nobody has discussed the competition — the exact failure
    the plan warns about.

The model is the SAME reactive function in both runs: it reacts to what the
tool told it, and only that differs. Transcripts land in runs/.

Run from the repo root:  python break_it_on_purpose_knowledge.py
"""

import ast
import os
import shutil
import tempfile
import urllib.error
from pathlib import Path

from core.contracts import ok
from core.errors import StepLimitReached
from core.loop import run_agent
from core.registry import Registry, ToolSpec
from tests.fakemodel import Reply, ToolCall
from tools.knowledge import discussions
from tools.knowledge.discussions import SEARCH_DISCUSSIONS, _search_precheck

ROOT = Path(__file__).resolve().parent
SEARCH_ARGS = {
    "competition_slug": "playground-series-s6e7",
    "query": "winning techniques",
    "sort": "votes",
    "max_results": 10,
}


def _mute(*a, **k):
    return None


def _always_429(url):
    """The simulated Kaggle API for both runs: every request is rate-limited."""
    raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)


def reactive_model(messages, tools):
    """The 'model'. It observes the last tool result and reacts to it. This
    exact logic runs in both scenarios; only what the search tool reported back
    differs — that is the whole point."""
    last = messages[-1] if messages else ""

    # Observe: nothing has run yet -> act by searching the corpus.
    if not (isinstance(last, str) and last.startswith("[TOOL ")):
        return Reply(tool_calls=[ToolCall("search_kaggle_discussions", dict(SEARCH_ARGS))])

    # Reason + act, branching purely on what the tool told us:
    if last.startswith("[TOOL OK search_kaggle_discussions]"):
        data = ast.literal_eval(last.split("] ", 1)[1])
        if data["count"] == 0:
            # The tool said the search SUCCEEDED and found nothing. Trusting
            # that is exactly right — which is why a 429 disguised as an empty
            # list is so dangerous.
            return Reply(text=(
                "No prior public work exists for this competition; starting "
                "from scratch and skipping further mining."
            ))
        return Reply(text="Found %d sources; proceeding to mine them." % data["count"])

    if last.startswith("[TOOL ERROR search_kaggle_discussions]"):
        # The search reported failure as its own branch -> the corpus state is
        # unknown, so do NOT conclude anything about prior work.
        return Reply(text=(
            "Search was rate-limited; the corpus state is UNKNOWN. Retrying "
            "later — NOT concluding that no prior work exists."
        ))

    # Any other surface -> stop rather than guess.
    return Reply(text="Unexpected tool result; stopping: %s" % last)


def naive_search(args):
    """The DANGEROUS alternative, with no error branch. It swallows the rate
    limit and returns the same shape a genuinely empty answer has, so the loop
    cannot tell 'nothing published' from 'the API refused to answer'."""
    try:
        payload = discussions._http_get_json("https://example.invalid/naive")
    except Exception:
        payload = []  # the collapse: an error becomes an empty list
    return ok(
        results=payload,
        count=len(payload),
        note="no public work matched this query yet — valid result, not a failure",
    )


# Same name, same parameters, same precheck as the real tool — only the body
# (and its missing error branch) differs, so the two runs are a fair comparison.
NAIVE_SEARCH = ToolSpec(
    name="search_kaggle_discussions",
    description=SEARCH_DISCUSSIONS.description,
    parameters=SEARCH_DISCUSSIONS.parameters,
    fn=naive_search,
    precheck=_search_precheck,
)


def run_scenario(search_spec):
    """Run the reactive model to completion against one search spec in a
    throwaway workspace, with the API always answering 429."""
    ws = Path(tempfile.mkdtemp(prefix="kc_break_"))
    real_fetch = discussions._http_get_json
    try:
        os.environ["KC_WORKSPACE"] = str(ws)
        discussions._http_get_json = _always_429

        reg = Registry()
        reg.register(search_spec)

        messages = []
        try:
            final = run_agent(messages, reactive_model, reg, max_steps=8,
                              input_fn=_mute, output_fn=_mute)
        except StepLimitReached as e:
            messages = e.messages
            final = "<step limit reached>"
        return messages, final
    finally:
        discussions._http_get_json = real_fetch
        shutil.rmtree(ws, ignore_errors=True)


def render(title, note, messages, final):
    lines = [title, "=" * len(title), note, ""]
    for m in messages:
        if isinstance(m, str):
            lines.append(m)  # a tool result line, already formatted by tool_message
        elif m.tool_calls:
            calls = ", ".join("%s(%s)" % (c.name, c.args) for c in m.tool_calls)
            lines.append("[MODEL] act  -> %s" % calls)
        else:
            lines.append("[MODEL] stop -> %s" % m.text)
    lines += ["", "[FINAL] %s" % final]
    return "\n".join(lines) + "\n"


def main():
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)

    msgs_with, final_with = run_scenario(SEARCH_DISCUSSIONS)
    (runs / "knowledge_with_branch.txt").write_text(render(
        "WITH the error branch (real search_kaggle_discussions)",
        "The API answers 429. The tool returns a rate_limited envelope, the loop "
        "takes its error branch, and the agent treats the corpus as unknown "
        "instead of empty.",
        msgs_with, final_with,
    ))

    msgs_without, final_without = run_scenario(NAIVE_SEARCH)
    (runs / "knowledge_without_branch.txt").write_text(render(
        "WITHOUT the error branch (naive search_kaggle_discussions)",
        "The API answers 429. The naive tool swallows it into an empty list, and "
        "the agent concludes with total confidence that no prior public work "
        "exists — a conclusion the API never supported.",
        msgs_without, final_without,
    ))

    print("wrote runs/knowledge_with_branch.txt and runs/knowledge_without_branch.txt")
    print("  with branch    -> %s" % final_with)
    print("  without branch -> %s" % final_without)


if __name__ == "__main__":
    main()
