"""Knowledge observer: what has the community already figured out?

    python -m ui.run_observer <slug>

A scripted pipeline, deliberately not an agent loop: the sequence is fixed
(list notebooks -> pull the top new ones -> summarize each -> aggregate), and
the LLM is only needed for the summarize steps. It journals to
<project>/observer/journal.jsonl in the same event vocabulary as agent runs so
the UI's status logic works unchanged, and appends durable records to
<project>/knowledge/notebooks.jsonl — the dedupe store that makes a refresh
cheap: already-summarized notebooks are never pulled or paid for twice.
"""

import argparse
import json
import os
import shutil
import sys
import time

from ui import kaggle_client, projects
from ui.journal import JournalWriter
from ui.llm import complete

# Per refresh: how many not-yet-summarized notebooks to pull and pay for.
_BATCH = 8
# The prompt gets at most this much notebook source; kernels can be huge and
# the techniques worth extracting are visible well before this cutoff.
_MAX_SOURCE = 15_000

_SUMMARY_SYSTEM = """You summarize Kaggle competition notebooks for a team
building its own pipeline. Answer with ONE JSON object, no markdown fence, no
prose around it, shaped exactly like:
{"summary": "2-4 sentences on what this notebook does",
 "models": ["model names used"],
 "cv_claim": "claimed CV score or ''",
 "lb_claim": "claimed leaderboard score or ''",
 "techniques": [{"name": "short technique name",
                 "why_it_matters": "one sentence",
                 "code_snippet": "the exact lines from the source that implement it",
                 "snippet_explanation": "one sentence tying snippet to technique"}]}
code_snippet must be copied verbatim from the provided source. 2-6 techniques."""

_AGGREGATE_SYSTEM = """You aggregate notebook summaries for one Kaggle
competition into a briefing. Output markdown with exactly these sections:
# Community briefing
## Technique frequency  (bullet list: technique — how many notebooks used it)
## Score claims  (bullet list of CV/LB claims worth trusting)
## Recommended pipeline  (short ordered list built from the consensus)"""


def _knowledge_dir(slug):
    d = projects.project_dir(slug) / "knowledge"
    d.mkdir(exist_ok=True)
    return d


def _load_seen(store):
    seen = set()
    if store.is_file():
        for line in store.read_text().splitlines():
            try:
                seen.add(json.loads(line)["ref"])
            except (ValueError, KeyError):
                continue
    return seen


def _parse_summary(text):
    """The model was told 'one JSON object, nothing else' — but be tolerant of
    a fenced or prefixed answer, since a retry costs real money."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict) or "techniques" not in obj:
        raise ValueError("reply JSON is missing 'techniques'")
    return obj


def _summarize(kernel, source, llm):
    user = "Notebook: %s by %s (%d votes)\n\nSOURCE:\n%s" % (
        kernel["title"],
        kernel["author"],
        kernel["votes"],
        source[:_MAX_SOURCE],
    )
    text = llm(_SUMMARY_SYSTEM, user, max_tokens=2500)
    try:
        return _parse_summary(text)
    except ValueError:
        # One paid retry with the failure named; after that the notebook is
        # skipped rather than sinking the whole refresh.
        text = llm(
            _SUMMARY_SYSTEM,
            user + "\n\nYour previous answer was not a single valid JSON object. "
            "Reply with ONLY the JSON object.",
            max_tokens=2500,
        )
        return _parse_summary(text)


def run_observer(slug, llm=complete, client=kaggle_client):
    project_dir = projects.project_dir(slug)
    obs_dir = project_dir / "observer"
    obs_dir.mkdir(exist_ok=True)
    writer = JournalWriter(obs_dir / "journal.jsonl")
    store = _knowledge_dir(slug) / "notebooks.jsonl"

    writer.write("run_started", prompt="observe community notebooks for %s" % slug)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        writer.write("run_failed", reason="ANTHROPIC_API_KEY not set on the server")
        return 1

    try:
        kernels = client.list_kernels(slug, page_size=20)
    except kaggle_client.KaggleError as e:
        writer.write("run_failed", reason="%s: %s" % (e.kind, e.msg))
        return 1
    writer.write("kernels_listed", count=len(kernels))

    seen = _load_seen(store)
    fresh = [k for k in kernels if k["ref"] not in seen][:_BATCH]
    writer.write("batch_selected", count=len(fresh), skipped_known=len(seen))

    scratch = obs_dir / "scratch"
    summarized = 0
    for kernel in fresh:
        try:
            pulled = client.pull_kernel(kernel["ref"], scratch)
            writer.write("kernel_pulled", ref=kernel["ref"])
            summary = _summarize(kernel, pulled["source"], llm)
        except (kaggle_client.KaggleError, RuntimeError, ValueError) as e:
            # One bad notebook must not sink the refresh: journal and move on.
            writer.write(
                "kernel_failed", ref=kernel["ref"], msg=str(e)[:300]
            )
            continue
        record = dict(
            summary,
            ref=kernel["ref"],
            url=kernel["url"],
            title=kernel["title"],
            author=kernel["author"],
            votes=kernel["votes"],
            pulled_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        with store.open("a") as f:
            f.write(json.dumps(record) + "\n")
        summarized += 1
        writer.write(
            "kernel_summarized",
            ref=kernel["ref"],
            techniques=[t.get("name", "?") for t in summary.get("techniques", [])],
        )
    shutil.rmtree(scratch, ignore_errors=True)

    # Aggregate over EVERYTHING known, not just this batch, so the briefing
    # stays whole across refreshes.
    records = []
    if store.is_file():
        for line in store.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    if records:
        digest = "\n\n".join(
            "## %s (%d votes)\n%s\ntechniques: %s\nCV: %s LB: %s"
            % (
                r["title"],
                r.get("votes", 0),
                r.get("summary", ""),
                ", ".join(t.get("name", "?") for t in r.get("techniques", [])),
                r.get("cv_claim", ""),
                r.get("lb_claim", ""),
            )
            for r in records
        )
        try:
            briefing = llm(_AGGREGATE_SYSTEM, digest, max_tokens=2000)
            (_knowledge_dir(slug) / "summary.md").write_text(briefing)
            writer.write("summary_written", notebooks=len(records))
        except RuntimeError as e:
            writer.write("kernel_failed", ref="(aggregate)", msg=str(e)[:300])

    writer.write(
        "run_finished",
        text="observed %d new notebooks (%d known total)" % (summarized, len(records)),
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize community notebooks.")
    parser.add_argument("slug")
    args = parser.parse_args(argv)
    obs_dir = projects.project_dir(args.slug) / "observer"
    obs_dir.mkdir(exist_ok=True)
    lock = obs_dir / "lock"
    try:
        lock.mkdir()
    except FileExistsError:
        print("observer already running for %s" % args.slug)
        return 1
    try:
        return run_observer(args.slug)
    finally:
        shutil.rmtree(lock, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
