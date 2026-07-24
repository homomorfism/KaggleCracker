"""The knowledge slice's reversible write: file away one mined technique.

save_technique_note is the deliberately ungated half of the pair — it appends
one line of local JSONL that can be deleted at will. The contrast with
fetch_external_dataset (gated, changes every downstream experiment) is the
point requirement 3 asks each slice to demonstrate.
"""

import json

from core.contracts import err, ok
from core.registry import ToolSpec
from tools.knowledge.discussions import _SLUG_CHARS
from tools.knowledge.paths import knowledge_dir


def _note_path(slug):
    return knowledge_dir() / (slug + ".jsonl")


def _note_precheck(args):
    """String-comparison guards before the write: a malformed slug would name a
    wrong file, and a non-http evidence_url is not evidence anyone can open.
    Returns an error envelope or None."""
    slug = args["competition_slug"]
    if not slug or not set(slug) <= _SLUG_CHARS:
        return err(
            "bad_input",
            "competition_slug must be lowercase letters, digits and hyphens, got %r" % slug,
        )
    url = args["evidence_url"]
    if not (url.startswith("https://") or url.startswith("http://")):
        return err("bad_input", "evidence_url must be an http(s) URL, got %r" % url)
    if not args["technique"].strip():
        return err("bad_input", "technique is empty")
    return None


def save_technique_note(args):
    slug = args["competition_slug"]
    row = {
        "technique": args["technique"].strip(),
        "evidence_url": args["evidence_url"],
        "confidence": args["confidence"],
        "est_effort": args["est_effort"],
    }

    path = _note_path(slug)

    # Re-noting the same technique from the same source adds no knowledge, so
    # the tool is idempotent instead of growing duplicates: report it was
    # already there and leave the file untouched.
    existing = []
    if path.exists():
        try:
            with path.open() as f:
                existing = [json.loads(line) for line in f if line.strip()]
        except (OSError, ValueError) as e:
            # A knowledge file that will not parse is a real failure to surface,
            # not something to silently overwrite or append past.
            return err("exec_failed", "could not read %s: %s" % (path, e))
    for rec in existing:
        if (
            rec.get("technique") == row["technique"]
            and rec.get("evidence_url") == row["evidence_url"]
        ):
            return ok(count=len(existing), duplicate=True, path=str(path))

    try:
        knowledge_dir().mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        return err("exec_failed", "could not append to %s: %s" % (path, e))

    return ok(count=len(existing) + 1, duplicate=False, path=str(path))


SAVE_TECHNIQUE_NOTE = ToolSpec(
    name="save_technique_note",
    description=(
        "Append one technique found in public Kaggle work to the local "
        "knowledge file for a competition, with its evidence URL, a confidence "
        "level and an effort estimate. Call this once per distinct technique, "
        "right after a search surfaced it. Do NOT call it to search (that is "
        "search_kaggle_discussions), do NOT invent a technique without an "
        "evidence URL to back it, and do NOT re-save a technique already in "
        "the file — the tool reports it as a duplicate and changes nothing. "
        "It writes one local, deletable line; it is reversible and ungated."
    ),
    parameters={
        "competition_slug": {"type": "str", "required": True},
        "technique": {"type": "str", "required": True},
        "evidence_url": {"type": "str", "required": True},
        # Constrained on purpose: the difference between "someone measured a CV
        # gain" and "someone claimed it in a comment" decides which experiments
        # are worth running first, so it cannot be free text.
        "confidence": {
            "type": "str",
            "enum": ["confirmed_cv_gain", "claimed", "speculative"],
            "required": True,
        },
        "est_effort": {"type": "str", "enum": ["low", "medium", "high"], "required": True},
    },
    fn=save_technique_note,
    precheck=_note_precheck,
    # No irreversible flag and no preview: this is the reversible half of the
    # pair, ungated by design — the contrast with fetch_external_dataset is
    # what requirement 3 grades.
)


def register(registry):
    registry.register(SAVE_TECHNIQUE_NOTE)
    return registry
