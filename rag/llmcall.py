"""One cached LLM transport for everything retrieval-eval related.

The reranker and the judges both go through here: temperature 0, and every
response cached on disk keyed by a hash of (model, system, user). Two
properties follow. Re-running the harness is free — the graded "do the full
set both ways" stops costing anything after the first pass — and it is as
reproducible as a judged pipeline can honestly be: cached numbers reproduce
exactly; a cold cache may not, and the report says so instead of pretending.

Stdlib urllib like ui/live_model.py; the cache lives in evals/cache/ (kept
out of git — the numbers in the report are the durable artifact).
"""

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-5"

CACHE_DIR = Path(__file__).resolve().parents[1] / "evals" / "cache"


def _cache_path(model, system, user):
    digest = hashlib.sha256(
        json.dumps([model, system, user], ensure_ascii=False).encode()
    ).hexdigest()
    return CACHE_DIR / ("%s.json" % digest)


def complete(system, user, max_tokens=1500, model=MODEL, use_cache=True):
    """Temperature-0 completion with a disk cache. Raises on missing key or
    API failure — a judged metric that silently skips calls is a lie."""
    path = _cache_path(model, system, user)
    if use_cache and path.is_file():
        return json.loads(path.read_text())["text"]

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — needed for rerank/judge calls "
            "(cached responses in evals/cache/ work without it)"
        )
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
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
    text = " ".join(
        b["text"] for b in reply.get("content", []) if b.get("type") == "text"
    ).strip()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"text": text}))
    os.replace(tmp, path)
    return text
