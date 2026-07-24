"""The knowledge slice's search tool: find what the community already knows.

The error branch here is the slice's whole point: "the API returned zero
results" is VALID DATA — early in a Playground month there may genuinely be
nothing yet — while "the API returned 429" is an ERROR. A naive implementation
collapses both into an empty list, and the model then concludes with total
confidence that nobody has discussed the competition. Here they are two
different envelopes and the loop visibly takes two different paths.

The live backend is Kaggle's documented v1 API (kernels/list), which surfaces
the public notebooks and write-ups for a competition — for a Playground
episode that is where the community's techniques actually live. Tests never
touch the network: they monkeypatch _http_get_json with canned payloads.
"""

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from core.contracts import err, ok
from core.registry import ToolSpec
from tools.knowledge.paths import knowledge_dir, search_log_path

_API_BASE = "https://www.kaggle.com/api/v1/kernels/list"

# The schema's friendly sort names mapped onto what the API actually accepts.
# The enum in the schema and the keys here must stay in step; the tool body
# indexes this dict with an already-validated value.
_SORT_PARAM = {
    "votes": "voteCount",
    "hot": "hotness",
    "recent": "dateCreated",
}

_SLUG_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-")


def _auth_header():
    """Basic-auth header from ~/.kaggle/kaggle.json, or None if absent.

    Missing credentials are not an error here: the request is still attempted
    and the API's own 401 comes back through the normal error branch, which is
    a truthful report the model can act on.
    """
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    if not token_path.exists():
        return None
    try:
        creds = json.loads(token_path.read_text())
        raw = "%s:%s" % (creds["username"], creds["key"])
    except (OSError, ValueError, KeyError):
        return None
    import base64

    return "Basic " + base64.b64encode(raw.encode()).decode()


def _http_get_json(url):
    """GET a URL and parse the JSON body. The single monkeypatch point for
    tests: canned payloads are returned from here, and the 429/timeout branches
    are exercised by making this raise. Lets urllib's exceptions propagate —
    mapping them to envelopes is the caller's job, kept in one visible place."""
    req = urllib.request.Request(url)
    auth = _auth_header()
    if auth is not None:
        req.add_header("Authorization", auth)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _search_precheck(args):
    """Machine-checkable guards before the body: a malformed slug or a blank
    query is rejected by string comparison, never sent over the network.
    Returns an error envelope or None."""
    slug = args["competition_slug"]
    if not slug or not set(slug) <= _SLUG_CHARS:
        return err(
            "bad_input",
            "competition_slug must be lowercase letters, digits and hyphens, got %r" % slug,
        )
    if not args["query"].strip():
        return err("bad_input", "query is empty")
    return None


def _log_search(slug, query, sort, urls):
    """Append this search to the durable search log the stall detector reads.
    Only successful searches are logged: a rate-limited attempt found nothing
    because it never ran, and counting it as 'no new sources' would let a flaky
    network trip the stall condition."""
    knowledge_dir().mkdir(parents=True, exist_ok=True)
    row = {"slug": slug, "query": query, "sort": sort, "urls": urls}
    with search_log_path().open("a") as f:
        f.write(json.dumps(row) + "\n")


def search_kaggle_discussions(args):
    slug = args["competition_slug"]
    query = args["query"]
    sort = args["sort"]
    max_results = args["max_results"]

    url = _API_BASE + "?" + urllib.parse.urlencode(
        {
            "competition": slug,
            "search": query,
            "sort_by": _SORT_PARAM[sort],
            "page_size": max_results,
        }
    )

    try:
        payload = _http_get_json(url)
    except urllib.error.HTTPError as e:
        # THE branch this slice is graded on. 429 is not "zero results" — it is
        # a rate limit, retryable after a pause, and the model must see it as a
        # failure, never as an empty (and falsely reassuring) answer.
        if e.code == 429:
            return err("rate_limited", "Kaggle API rate limit (HTTP 429); wait and retry", retryable=True)
        if e.code == 404:
            return err("not_found", "no such competition on Kaggle: %r" % slug)
        return err("exec_failed", "Kaggle API returned HTTP %d" % e.code)
    except (TimeoutError, socket.timeout):
        return err("timeout", "Kaggle API did not answer within 30s", retryable=True)
    except urllib.error.URLError as e:
        return err("exec_failed", "could not reach Kaggle: %s" % e.reason, retryable=True)
    except ValueError as e:  # covers json.JSONDecodeError
        return err("exec_failed", "Kaggle API returned unparseable JSON: %s" % e)

    if not isinstance(payload, list):
        return err("exec_failed", "unexpected response shape: expected a list")

    results = []
    for item in payload[:max_results]:
        if not isinstance(item, dict) or "ref" not in item:
            # A malformed item is skipped, not fatal: one odd record must not
            # discard an otherwise good page of results.
            continue
        results.append(
            {
                "title": item.get("title", ""),
                "url": "https://www.kaggle.com/code/" + str(item["ref"]).lstrip("/"),
                "author": item.get("author", ""),
                "votes": item.get("totalVotes", 0),
            }
        )

    _log_search(slug, query, sort, [r["url"] for r in results])

    # Zero results is a SUCCESS, deliberately. The note spells out for the model
    # what an empty list means, so "nothing published yet" is an observation it
    # can reason about — not an error and not proof that nothing ever will be.
    return ok(
        results=results,
        count=len(results),
        note=(
            "no public work matched this query yet — valid result, not a failure"
            if not results
            else "public notebooks/write-ups for this competition, best first"
        ),
    )


SEARCH_DISCUSSIONS = ToolSpec(
    name="search_kaggle_discussions",
    description=(
        "Search the public Kaggle knowledge for a competition — community "
        "notebooks, write-ups and their titles — and return titles, URLs, "
        "authors and vote counts. Call this to find techniques other people "
        "have already tried for this competition. An empty result list is a "
        "valid answer meaning nothing public matches yet; do NOT retry the "
        "same query hoping for a different answer — a repeated query trips "
        "the stall condition. Do NOT call it to download data or notebooks "
        "(that is fetch_external_dataset), and do NOT call it to record a "
        "finding you already have (that is save_technique_note)."
    ),
    parameters={
        "competition_slug": {"type": "str", "required": True},
        "query": {"type": "str", "required": True},
        # Constrained on purpose: the API accepts many sort orders, but only
        # these three answer a question the agent actually has.
        "sort": {"type": "str", "enum": ["hot", "recent", "votes"], "default": "votes"},
        # Capped at 50 in the schema so an oversized ask is a bad_input
        # rejection before any network traffic.
        "max_results": {"type": "int", "min": 1, "max": 50, "default": 10},
    },
    fn=search_kaggle_discussions,
    precheck=_search_precheck,
    # Reversible: a search reads public pages and appends to a local log.
    # Nothing external changes, so it is ungated by design.
)


def register(registry):
    registry.register(SEARCH_DISCUSSIONS)
    return registry
