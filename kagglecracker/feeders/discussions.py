"""Discussion feeder: read the competition forum, summarise it into context.

    kaggle competitions topics <comp>   ──► topic list
    kaggle forums topics show <ref>     ──► thread bodies
                                                │
                                    AgentRuntime.generate
                                                │
                              context/discussions/<date>.md

This runs host-side, not in the sandbox: it is the one component allowed to
touch the network, and it only ever talks to the Kaggle API. Experiment
containers stay at `--network none`.

The design doc originally planned this as a manual copy-paste step, on the
belief that the Kaggle API had no discussions endpoint. It does as of
kaggle-cli 2.2.0, which is what makes this unattended.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kagglecracker.config import Settings
from kagglecracker.runtime.base import AgentRuntime, GenerateResult

DISCUSSIONS_DIRNAME = "discussions"

SUMMARY_PROMPT = """\
Below are discussion threads from a Kaggle competition's forum, fetched with the
official CLI.

Write a summary for an engineer about to write a solution. Forums are mostly
noise — greetings, leaderboard chatter, questions with no answers. Extract only
what would change how someone models this problem: reported techniques and the
scores they achieved, data quirks or leaks people found, validation advice,
approaches reported as dead ends.

Attribute scores to the technique that produced them where the thread says so.
Where a claim is unverified or contested, say so. Do not invent numbers. If the
threads contain nothing useful, say exactly that rather than padding.

This document is injected into every subsequent attempt, so length has a real
cost. Return markdown only, no preamble.

---

{threads}
"""


@dataclass(frozen=True, slots=True)
class DiscussionResult:
    ok: bool
    path: Path | None
    topics_found: int
    raw_chars: int
    generation: GenerateResult | None
    error: str = ""


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return 127, "", "kaggle CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def list_topics(competition: str, *, limit: int = 15) -> list[tuple[str, str]]:
    """Return (topic_id, title) pairs, newest first."""
    # `--page-size` is not supported for competition topics (the CLI warns and
    # ignores it), so the limit is applied here instead.
    code, out, err = _run(
        ["kaggle", "competitions", "topics", "list", competition, "--csv"]
    )
    if code != 0:
        raise RuntimeError(f"listing topics failed: {err or out}")

    import csv
    import io

    topics: list[tuple[str, str]] = []
    for r in csv.DictReader(io.StringIO(out)):
        # A row with more fields than the header puts the overflow under the
        # key None (csv.DictReader's restkey default), so keys are not all
        # strings. Titles containing commas make that routine here.
        keys = [k for k in r if isinstance(k, str)]
        # Column names have moved between CLI versions; match on shape rather
        # than pinning to one header spelling.
        tid = next((r[k] for k in keys if k.lower() in ("id", "topicid", "ref")), None)
        title = next((r[k] for k in keys if "title" in k.lower()), "") or ""
        if tid:
            topics.append((str(tid), title))
    return topics[:limit]


def fetch_topic(topic_id: str, *, max_chars: int = 6000) -> str:
    code, out, err = _run(["kaggle", "forums", "topics", "show", topic_id])
    if code != 0:
        return ""
    return out[:max_chars]


def run_discussions(
    *,
    settings: Settings,
    runtime: AgentRuntime,
    limit: int = 15,
    max_total_chars: int = 60_000,
) -> DiscussionResult:
    try:
        topics = list_topics(settings.competition, limit=limit)
    except RuntimeError as exc:
        return DiscussionResult(False, None, 0, 0, None, str(exc))

    if not topics:
        return DiscussionResult(
            False, None, 0, 0, None, "no discussion topics returned for this competition"
        )

    chunks: list[str] = []
    used = 0
    for tid, title in topics:
        body = fetch_topic(tid)
        if not body:
            continue
        block = f"### {title} (topic {tid})\n{body}"
        if used + len(block) > max_total_chars:
            break
        chunks.append(block)
        used += len(block)

    if not chunks:
        return DiscussionResult(
            False, None, len(topics), 0, None, "topics listed but none could be fetched"
        )

    raw = "\n\n".join(chunks)
    gen = runtime.generate(SUMMARY_PROMPT.format(threads=raw), action="draft")
    text = gen.raw_text.strip()
    if not text:
        return DiscussionResult(
            False, None, len(topics), len(raw), gen,
            f"summariser returned nothing ({gen.status})",
        )

    out_dir = settings.context_dir / DISCUSSIONS_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dt.date.today().isoformat()}.md"
    path.write_text(text + "\n")
    return DiscussionResult(True, path, len(topics), len(raw), gen)
