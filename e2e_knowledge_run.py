"""One end-to-end run of the knowledge slice against the REAL model (§6.2 #6).

The same run_agent loop the tests drive with FakeModel is driven here by a live
Anthropic model over the live Kaggle API. The adapter below is the only new
piece: it translates between our envelope/transcript world and the Messages
API, using nothing but the standard library (per the no-dependencies rule).

The gate stays real: if the model asks for fetch_external_dataset, the approval
prompt goes to stdin, and a closed stdin is a denial — absence of a human is
never consent.

Run from the repo root:  python e2e_knowledge_run.py
Transcript lands in runs/knowledge_e2e_run.txt.
"""

import datetime
import json
import os
import urllib.request
from pathlib import Path

from core.errors import StepLimitReached
from core.loop import run_agent
from core.registry import Registry
from tests.fakemodel import Reply, ToolCall
from tools.knowledge import discussions, external, notes
from tools.knowledge.paths import search_log_path
from tools.knowledge.stopping import stall_detector

ROOT = Path(__file__).resolve().parent
MODEL = "claude-sonnet-5"
API_URL = "https://api.anthropic.com/v1/messages"

TASK = """\
You are the knowledge-mining agent for the Kaggle Playground competition
'playground-series-s6e7' (Predicting Student Health Risk, tabular, metric is
Balanced Accuracy Score). Your job in this run:

1. Search the public Kaggle work for this competition with a few DISTINCT
   queries (techniques, ensembling, feature engineering, original dataset...).
   Run ONE search at a time.
2. Immediately after each search, BEFORE searching again, save each concrete,
   evidenced technique it surfaced as a note, with the URL that backs it, an
   honest confidence level and an effort estimate.
3. When searches stop surfacing anything new, STOP and summarise what you
   learned and what the exec slice should try first.

Do not fetch external datasets in this run unless a search gives you a specific,
named original dataset with a strong reason — fetching requires human approval.
Never repeat a query you already ran."""


# --- our ToolSpec schema -> JSON Schema for the Messages API -----------------

_JSON_TYPE = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def _to_json_schema(parameters):
    props, required = {}, []
    for name, spec in parameters.items():
        if spec.get("type") == "list":
            p = {"type": "array"}
            if "item_enum" in spec:
                p["items"] = {"enum": list(spec["item_enum"])}
        else:
            p = {"type": _JSON_TYPE[spec["type"]]}
            if "enum" in spec:
                p["enum"] = list(spec["enum"])
            if "min" in spec:
                p["minimum"] = spec["min"]
            if "max" in spec:
                p["maximum"] = spec["max"]
        if "default" in spec:
            p["default"] = spec["default"]
        props[name] = p
        if spec.get("required"):
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


class AnthropicModel:
    """Callable with the model(messages, tools) contract run_agent expects.

    Keeps its own native Messages-API conversation. Our transcript strings
    (appended by run_agent after each dispatch) are paired, in order, with the
    tool_use blocks of the previous reply and fed back as tool_result blocks —
    so the model sees exactly what the loop saw, envelope formatting included.
    """

    def __init__(self, api_key):
        self._api_key = api_key
        self._native = []
        self._pending_ids = []

    def __call__(self, messages, tools):
        if not self._native:
            self._native.append({"role": "user", "content": TASK})
        elif self._pending_ids:
            results = [m for m in messages if isinstance(m, str)][-len(self._pending_ids):]
            self._native.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": text}
                for tid, text in zip(self._pending_ids, results)
            ]})

        body = {
            "model": MODEL,
            "max_tokens": 1500,
            "messages": self._native,
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": _to_json_schema(t["parameters"]),
                }
                for t in tools
            ],
        }
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode(),
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            reply = json.loads(resp.read().decode())

        self._native.append({"role": "assistant", "content": reply["content"]})
        text = " ".join(b["text"] for b in reply["content"] if b["type"] == "text")
        calls = [b for b in reply["content"] if b["type"] == "tool_use"]
        self._pending_ids = [b["id"] for b in calls]
        return Reply(text=text.strip(),
                     tool_calls=[ToolCall(b["name"], b["input"]) for b in calls])


def render(messages, final, stopped_by_stall):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    title = "END-TO-END RUN — real model (%s), live Kaggle API, %s" % (MODEL, stamp)
    lines = [title, "=" * len(title),
             "The same run_agent/dispatch/gate/stall machinery the tests drive with "
             "FakeModel, driven by the live model. Nothing mocked.", ""]
    for m in messages:
        if isinstance(m, str):
            lines.append(m)
        elif m.tool_calls:
            if m.text:
                lines.append("[MODEL] say  -> %s" % m.text)
            for c in m.tool_calls:
                lines.append("[MODEL] act  -> %s(%s)" % (c.name, json.dumps(c.args)))
        else:
            lines.append("[MODEL] stop -> %s" % m.text)
    if stopped_by_stall:
        lines.append("[LOOP] stall detector fired — run stopped by the stopping condition")
    lines += ["", "[FINAL] %s" % final]
    return "\n".join(lines) + "\n"


def main():
    # A stale search log would let an old run trip the stall detector; rotate it
    # aside so this run's stopping condition judges only this run.
    log = search_log_path()
    if log.exists():
        log.rename(log.with_suffix(".jsonl.bak"))

    registry = Registry()
    discussions.register(registry)
    notes.register(registry)
    external.register(registry)

    model = AnthropicModel(os.environ["ANTHROPIC_API_KEY"])
    messages = []
    stopped_by_stall = False
    try:
        final = run_agent(messages, model, registry, max_steps=24,
                          should_stop=stall_detector())
        if final is None:
            stopped_by_stall = True
            final = "<stopped by the stall detector>"
    except StepLimitReached as e:
        messages, final = e.messages, "<step limit reached>"

    out = ROOT / "runs" / "knowledge_e2e_run.txt"
    out.write_text(render(messages, final, stopped_by_stall))
    print("wrote %s" % out)
    print("[FINAL] %s" % final)


if __name__ == "__main__":
    main()
