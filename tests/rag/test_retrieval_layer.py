"""Chunking, BM25, RRF and the search_corpus tool."""

import pytest

from core.registry import validate
from rag.bm25 import BM25
from rag.chunking import TARGET_CHARS, chunk_markdown
from rag.fuse import rrf
from rag.search import CorpusIndex, get_index
from tools.corpus.search import SEARCH_CORPUS

_DOC = """# Title

Intro paragraph.

## Section A

```python
# this is a code comment, not a heading
x = 1
```

Body of section A.

## Section B

""" + ("Long paragraph. " * 30 + "\n\n") * 6


def test_chunking_is_deterministic_and_heading_aware():
    a = chunk_markdown("doc.md", _DOC)
    b = chunk_markdown("doc.md", _DOC)
    assert a == b
    headings = {c["heading"] for c in a}
    assert any("Section A" in h for h in headings)
    # The '#' inside the fenced block never became a heading.
    assert not any("code comment" in h for h in headings)


def test_long_sections_split_with_overlap():
    chunks = [c for c in chunk_markdown("doc.md", _DOC) if "Section B" in c["heading"]]
    assert len(chunks) > 1
    assert all(len(c["text"]) <= TARGET_CHARS + 300 for c in chunks)
    # Overlap: the tail of one piece reappears at the head of the next.
    tail = chunks[0]["text"][-80:]
    assert tail[:40] in chunks[1]["text"]


def test_bm25_puts_the_exact_term_first():
    texts = [
        "the gate approves irreversible actions",
        "KC_WORKSPACE points tools at the workspace root",
        "balanced accuracy is the metric",
    ]
    index = BM25(texts)
    assert index.rank("KC_WORKSPACE", 3)[0] == 1
    assert index.rank("nonexistent-term", 3) == []


def test_rrf_hand_computed():
    fused = rrf([["a", "b", "c"], ["b", "a", "c"]], k=60)
    # a: 1/61+1/62, b: 1/62+1/61 — tie, broken by first appearance; c last.
    assert fused[:2] == ["a", "b"] and fused[2] == "c"
    # A doc ranked high by only one retriever still fuses in.
    assert "d" in rrf([["d"], ["a", "b"]], k=60)


def test_search_corpus_tool_schema_rejects_before_body():
    cleaned, err = validate(SEARCH_CORPUS, {"query": "gate", "k": 99})
    assert cleaned is None and err["error"]["kind"] == "bad_input"
    cleaned, err = validate(SEARCH_CORPUS, {"query": "gate"})
    assert err is None and cleaned["k"] == 5 and cleaned["rerank"] is False


def test_search_corpus_tool_happy_path_envelope():
    result = SEARCH_CORPUS.fn({"query": "who approves the gate", "k": 3, "rerank": False})
    assert result["ok"] is True
    results = result["data"]["results"]
    assert 1 <= len(results) <= 3
    assert {"chunk_id", "path", "heading", "snippet"} <= set(results[0])


def test_rerank_without_key_is_an_error_branch(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # A query no cached rerank response can exist for.
    result = SEARCH_CORPUS.fn(
        {"query": "zz-unseen-query-xyzzy-42", "k": 3, "rerank": True}
    )
    assert result["ok"] is False
    assert result["error"]["kind"] == "exec_failed"
    assert "ANTHROPIC_API_KEY" in result["error"]["msg"]


def test_index_search_order_is_deterministic():
    index = get_index()
    q = "what metric does the competition use"
    assert [c["id"] for c in index.search(q, k=8)] == [
        c["id"] for c in index.search(q, k=8)
    ]


def test_fixture_corpus_isolated_from_repo(tmp_path):
    (tmp_path / "one.md").write_text("# Only\n\ngate approval text here.")
    index = CorpusIndex(root=tmp_path, files=["one.md"])
    assert len(index.chunks) == 1
    assert index.search("gate", k=1)[0]["id"] == "one.md::000"
