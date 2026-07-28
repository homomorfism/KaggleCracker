"""One-shot LLM completion for scripted pipelines.

The agent loop keeps its stateful adapter in ui/live_model.py; this module is
for pipelines (the observer) that need a single answer to a single prompt.
Same transport, same honest failure mode: the API's own error text in the
raised exception.
"""

import json
import os
import urllib.error
import urllib.request

from ui.live_model import API_URL, DEFAULT_MODEL


def complete(system, user, max_tokens=2000, api_key=None, model=DEFAULT_MODEL):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            reply = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        raise RuntimeError("Anthropic API %s: %s" % (e.code, detail))
    return " ".join(
        b["text"] for b in reply.get("content", []) if b.get("type") == "text"
    ).strip()
