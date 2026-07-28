"""The Kaggle wrapper without the network: a fake KaggleApi stands in, and the
tests assert on the error taxonomy the rest of the app depends on."""

import json
import zipfile
from types import SimpleNamespace

import pytest

from ui import kaggle_client
from ui.kaggle_client import KaggleError


@pytest.fixture(autouse=True)
def fresh_api(monkeypatch):
    monkeypatch.setattr(kaggle_client, "_api_instance", None)


def _install(monkeypatch, api):
    monkeypatch.setattr(kaggle_client, "_api", lambda: api)


# --- url parsing --------------------------------------------------------------


def test_parse_accepts_urls_and_bare_slugs():
    for given in (
        "https://www.kaggle.com/competitions/playground-series-s5e7",
        "https://www.kaggle.com/competitions/playground-series-s5e7/overview",
        "https://www.kaggle.com/c/playground-series-s5e7",
        "playground-series-s5e7",
        "  Playground-Series-S5E7  ",
    ):
        assert kaggle_client.parse_competition_url(given) == "playground-series-s5e7"


def test_parse_rejects_junk():
    for given in ("https://example.com/competitions/x", "not a slug!", ""):
        with pytest.raises(KaggleError) as e:
            kaggle_client.parse_competition_url(given)
        assert e.value.kind == "bad_url"


# --- metadata -----------------------------------------------------------------


def _comp(slug, title="T"):
    return SimpleNamespace(
        ref="https://www.kaggle.com/competitions/%s" % slug,
        title=title,
        description="desc",
        evaluation_metric="RMSE",
        deadline="2026-09-01",
        category="Playground",
        reward="Swag",
    )


def test_fetch_metadata_matches_exact_slug(monkeypatch):
    # The search endpoint fuzzy-matches; only the exact ref match counts.
    resp = SimpleNamespace(competitions=[_comp("titanic-extended"), _comp("titanic")])
    _install(monkeypatch, SimpleNamespace(competitions_list=lambda search: resp))
    meta = kaggle_client.fetch_metadata("titanic")
    assert meta["slug"] == "titanic"
    assert meta["evaluation_metric"] == "RMSE"
    assert meta["url"].endswith("/titanic")


def test_fetch_metadata_no_match_is_not_found(monkeypatch):
    resp = SimpleNamespace(competitions=[_comp("titanic-extended")])
    _install(monkeypatch, SimpleNamespace(competitions_list=lambda search: resp))
    with pytest.raises(KaggleError) as e:
        kaggle_client.fetch_metadata("titanic")
    assert e.value.kind == "not_found"


def test_http_statuses_map_to_kinds(monkeypatch):
    cases = {
        "403 Forbidden": "rules_not_accepted",
        "404 Not Found": "not_found",
        "429 Too Many Requests": "rate_limited",
        "401 Unauthorized": "no_credentials",
        "connection reset": "network",
    }
    for text, kind in cases.items():
        def boom(search, _text=text):
            raise RuntimeError(_text)

        _install(monkeypatch, SimpleNamespace(competitions_list=boom))
        with pytest.raises(KaggleError) as e:
            kaggle_client.fetch_metadata("titanic")
        assert e.value.kind == kind, text


def test_no_credentials_raised_before_any_network(monkeypatch):
    monkeypatch.setattr(kaggle_client, "has_credentials", lambda: False)
    with pytest.raises(KaggleError) as e:
        kaggle_client._api()
    assert e.value.kind == "no_credentials"
    assert "KAGGLE_USERNAME" in e.value.msg and "kaggle.json" in e.value.msg


# --- download -----------------------------------------------------------------


def test_download_extracts_and_removes_the_zip(monkeypatch, tmp_path):
    def fake_download(slug, path, quiet):
        with zipfile.ZipFile(str(tmp_path / "dest" / "bundle.zip"), "w") as zf:
            zf.writestr("train.csv", "a,b\n1,2\n")
            zf.writestr("test.csv", "a\n1\n")

    _install(
        monkeypatch,
        SimpleNamespace(competition_download_files=fake_download),
    )
    extracted = kaggle_client.download_data("comp", tmp_path / "dest")
    assert sorted(extracted) == ["test.csv", "train.csv"]
    assert (tmp_path / "dest" / "train.csv").read_text() == "a,b\n1,2\n"
    assert not list((tmp_path / "dest").glob("*.zip"))


# --- kernels ------------------------------------------------------------------


def test_list_kernels_shapes_and_skips_refless(monkeypatch):
    kernels = [
        SimpleNamespace(ref="a/nb-one", title="One", author="A", total_votes=10, language="python"),
        SimpleNamespace(ref=None, title="ghost", author="", total_votes=0, language=""),
    ]
    _install(
        monkeypatch,
        SimpleNamespace(kernels_list=lambda competition, sort_by, page_size: kernels),
    )
    out = kaggle_client.list_kernels("comp")
    assert out == [
        {
            "ref": "a/nb-one",
            "title": "One",
            "author": "A",
            "votes": 10,
            "language": "python",
            "url": "https://www.kaggle.com/code/a/nb-one",
        }
    ]


def test_pull_kernel_flattens_notebook_cells(monkeypatch, tmp_path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Intro\n", "words"]},
            {"cell_type": "code", "source": ["import pandas as pd\n", "pd.read_csv('x')"]},
        ]
    }

    def fake_pull(ref, path):
        (tmp_path / "scratch" / "nb-one.ipynb").write_text(json.dumps(nb))

    _install(monkeypatch, SimpleNamespace(kernels_pull=fake_pull))
    out = kaggle_client.pull_kernel("a/nb-one", tmp_path / "scratch")
    assert out["ref"] == "a/nb-one"
    assert "import pandas as pd" in out["source"]
    # Markdown cells arrive commented, so the summarizer sees prose as prose.
    assert "# # Intro" in out["source"]
    # The scratch file is consumed, not left to shadow the next pull.
    assert not list((tmp_path / "scratch").iterdir())
