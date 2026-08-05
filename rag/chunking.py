"""Markdown-aware chunking for the project-docs corpus.

Strategy, and why it fits these documents: the corpus is reference material
(architecture notes, plans, rules) whose authors already organised it under
headings — so headings are the semantic boundaries, and a chunk is a heading's
section. Sections longer than TARGET_CHARS split further at paragraph breaks
with OVERLAP_CHARS carried over, so a sentence cut mid-thought at a split
still appears whole in one of the two chunks. Sections are never merged
across headings: a chunk answering "what does the gate do" must not drag in
half of an unrelated section just to fill a size quota.

Chunk ids are ordinal (path::NNN) and therefore MOVE whenever a document is
re-chunked — which is exactly why the eval set stores content anchors, not
ids, and resolves them at load time.
"""

import re

TARGET_CHARS = 1200
OVERLAP_CHARS = 200

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def _split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_long(section_text):
    """Split one section into <= TARGET_CHARS pieces at paragraph boundaries,
    overlapping by carrying the tail of the previous piece forward."""
    paragraphs = _split_paragraphs(section_text)
    pieces, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) + 2 > TARGET_CHARS:
            pieces.append(current)
            current = current[-OVERLAP_CHARS:] + "\n\n" if OVERLAP_CHARS else ""
        current = (current + "\n\n" + p).strip() if current else p
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(path, text):
    """One markdown document -> ordered chunk dicts.

    Each chunk: {id, path, heading, text}. `heading` is the full heading path
    ("KaggleCracker — LLM Architecture › 2. Agents › 2.4 Offline monitor") so
    a chunk stays self-describing after retrieval, and the heading is
    prepended to the indexed text because section bodies often say "it" and
    "this tool" while the heading names the referent.
    """
    lines = text.splitlines()
    trail = []  # (level, title) breadcrumb of enclosing headings
    sections = []  # (heading_path, body_lines)
    body = []

    def flush():
        if body and any(l.strip() for l in body):
            heading = " › ".join(t for _, t in trail) or path
            sections.append((heading, "\n".join(body).strip()))
        body.clear()

    in_fence = False
    for line in lines:
        # A '#' inside a fenced code block is a comment, not a heading —
        # PLAN.md's code samples would otherwise pollute the breadcrumbs.
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            body.append(line)
            continue
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while trail and trail[-1][0] >= level:
                trail.pop()
            trail.append((level, m.group(2).strip()))
        else:
            body.append(line)
    flush()

    chunks = []
    for heading, section_text in sections:
        for piece in _split_long(section_text):
            chunks.append(
                {
                    "id": "%s::%03d" % (path, len(chunks)),
                    "path": path,
                    "heading": heading,
                    "text": "%s\n%s" % (heading, piece),
                }
            )
    return chunks
