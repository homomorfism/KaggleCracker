"""Live Anthropic driver for UI runs.

The same ``model(messages, tools)`` contract FakeModel satisfies, backed by
the Messages API over stdlib urllib (the no-dependency rule). This generalises
the adapter proven in e2e_knowledge_run.py: the task prompt is a constructor
argument instead of a module constant, and API failures raise with the API's
own error text so the runner can journal an honest run_failed.
"""

import json
import urllib.error
import urllib.request

from tests.fakemodel import Reply, ToolCall

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5"

_JSON_TYPE = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def to_json_schema(parameters):
    """Our ToolSpec parameter dicts -> JSON Schema for the Messages API."""
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
    """Keeps its own native Messages-API conversation. Our transcript strings
    (appended by run_agent after each dispatch) are paired, in order, with the
    tool_use blocks of the previous reply and fed back as tool_result blocks —
    the model sees exactly what the loop saw, envelope formatting included."""

    def __init__(self, api_key, task, model=DEFAULT_MODEL, max_tokens=3000):
        self._api_key = api_key
        self._task = task
        self._model = model
        self._max_tokens = max_tokens
        self._native = []
        self._pending_ids = []

    def __call__(self, messages, tools):
        if not self._native:
            self._native.append({"role": "user", "content": self._task})
        elif self._pending_ids:
            results = [m for m in messages if isinstance(m, str)][-len(self._pending_ids):]
            self._native.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": text}
                for tid, text in zip(self._pending_ids, results)
            ]})

        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": self._native,
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": to_json_schema(t["parameters"]),
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
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                reply = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            # Surface the API's own words: the runner journals this as
            # run_failed, and "overloaded_error" beats a bare 529.
            detail = e.read().decode(errors="replace")[:400]
            raise RuntimeError("Anthropic API %s: %s" % (e.code, detail))

        self._native.append({"role": "assistant", "content": reply["content"]})
        text = " ".join(b["text"] for b in reply["content"] if b["type"] == "text")
        calls = [b for b in reply["content"] if b["type"] == "tool_use"]
        self._pending_ids = [b["id"] for b in calls]
        return Reply(
            text=text.strip(),
            tool_calls=[ToolCall(b["name"], b["input"]) for b in calls],
        )
