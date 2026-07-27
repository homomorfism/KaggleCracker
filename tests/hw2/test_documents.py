"""Tests for the JSON-lines document store (memory/documents.py).

KC_DOCS is pointed at a file under tmp_path via the `docs` fixture, so no test
writes into the real workspace/. Each test asserts on the consequence of the
visibility rule — that a private record is genuinely absent for another user,
and that a shared one genuinely reaches them.
"""

import pytest

from memory import documents


@pytest.fixture
def docs(tmp_path, monkeypatch):
    """Point KC_DOCS at a fresh tmp file. Functions read the env default."""
    monkeypatch.setenv("KC_DOCS", str(tmp_path / "documents.jsonl"))


def test_private_doc_is_invisible_to_another_user(docs):
    doc_id = documents.save_document(
        "alice", "secret tuning notes for xgboost", "xgboost", shared=False
    )
    # Bob queries the very keyword alice used; the record still must not appear.
    for_bob = documents.retrieve_documents("bob", "xgboost")
    assert doc_id not in {r["id"] for r in for_bob}
    assert for_bob == []


def test_private_doc_is_visible_to_its_owner(docs):
    doc_id = documents.save_document(
        "alice", "secret tuning notes for xgboost", "xgboost", shared=False
    )
    for_alice = documents.retrieve_documents("alice", "xgboost")
    assert {r["id"] for r in for_alice} == {doc_id}


def test_shared_doc_is_visible_to_another_user(docs):
    doc_id = documents.save_document(
        "alice", "lightgbm learning-rate trick that helped", "lightgbm", shared=True
    )
    for_bob = documents.retrieve_documents("bob", "lightgbm")
    assert {r["id"] for r in for_bob} == {doc_id}


def test_include_shared_false_hides_others_shared_docs(docs):
    documents.save_document("alice", "shared cv trick", "cv", shared=True)
    # With include_shared off, bob sees only his own — none here.
    for_bob = documents.retrieve_documents("bob", "cv", include_shared=False)
    assert for_bob == []


def test_query_filters_within_visible_records(docs):
    keep = documents.save_document("alice", "notes about catboost depth", "catboost")
    documents.save_document("alice", "notes about lightgbm leaves", "lightgbm")
    hits = documents.retrieve_documents("alice", "catboost")
    assert {r["id"] for r in hits} == {keep}


def test_cue_matches_even_when_body_does_not(docs):
    doc_id = documents.save_document(
        "alice", "the run finally converged", "early-stopping patience", shared=False
    )
    # "patience" appears only in the cue, not the body.
    hits = documents.retrieve_documents("alice", "patience")
    assert {r["id"] for r in hits} == {doc_id}
