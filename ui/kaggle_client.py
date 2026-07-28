"""One seam between this repo and the official `kaggle` package.

Everything that talks to Kaggle goes through this module so the rest of the
code (and every test) sees a small, fake-able surface. The kaggle package is
imported lazily inside `_api()` — importing `kaggle` at module import time
authenticates immediately and would make merely importing this file fail on a
machine without credentials.

All failures become KaggleError(kind, msg) with a closed set of kinds; callers
turn `kind` into UI-facing, actionable text (e.g. rules_not_accepted links to
the competition's rules page).
"""

import json
import os
import re
import zipfile
from pathlib import Path

_KINDS = (
    "no_credentials",
    "rules_not_accepted",
    "not_found",
    "rate_limited",
    "network",
    "bad_url",
)

_URL_RE = re.compile(r"kaggle\.com/(?:c|competitions)/([A-Za-z0-9-]+)")
_SLUG_RE = re.compile(r"^[A-Za-z0-9-]+$")


class KaggleError(Exception):
    def __init__(self, kind, msg):
        assert kind in _KINDS, kind
        super().__init__(msg)
        self.kind = kind
        self.msg = msg


_api_instance = None


def _api():
    """Authenticated KaggleApi singleton, created on first use."""
    global _api_instance
    if _api_instance is not None:
        return _api_instance
    if not has_credentials():
        raise KaggleError(
            "no_credentials",
            "no Kaggle credentials: set KAGGLE_USERNAME and KAGGLE_KEY, or put "
            "an API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings -> "
            "Create New Token)",
        )
    # Imported here, not at module top: `import kaggle` authenticates eagerly.
    import kaggle

    _api_instance = kaggle.api
    return _api_instance


def has_credentials():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").is_file()


def _wrap(exc, slug=""):
    """Map an exception from the kaggle package onto a KaggleError.

    The 2.x SDK raises different exception types per transport; the HTTP status
    embedded in the message is the only stable signal, so classify on that.
    """
    text = str(exc)
    if "403" in text or "Forbidden" in text:
        return KaggleError(
            "rules_not_accepted",
            "Kaggle returned 403 for %r — most likely the competition rules are "
            "not accepted yet. Open https://www.kaggle.com/c/%s/rules , accept, "
            "then retry." % (slug, slug),
        )
    if "404" in text or "Not Found" in text:
        return KaggleError("not_found", "Kaggle has no competition %r" % slug)
    if "429" in text or "Too Many Requests" in text:
        return KaggleError("rate_limited", "Kaggle rate limit hit; retry in a minute")
    if "401" in text or "Unauthorized" in text:
        return KaggleError(
            "no_credentials",
            "Kaggle rejected the credentials (401); recreate the API token",
        )
    return KaggleError("network", "Kaggle call failed: %s" % text[:300])


def parse_competition_url(url):
    """Accept a full competition URL or a bare slug; return the slug."""
    url = (url or "").strip()
    m = _URL_RE.search(url)
    if m:
        return m.group(1).lower()
    if _SLUG_RE.match(url):
        return url.lower()
    raise KaggleError(
        "bad_url",
        "expected https://www.kaggle.com/competitions/<slug> or a bare slug, got %r" % url,
    )


def fetch_metadata(slug):
    api = _api()
    try:
        resp = api.competitions_list(search=slug)
    except KaggleError:
        raise
    except Exception as e:
        raise _wrap(e, slug) from e
    comps = getattr(resp, "competitions", None) or []
    for c in comps:
        ref = (getattr(c, "ref", "") or "").rstrip("/")
        if ref.rsplit("/", 1)[-1].lower() == slug:
            return {
                "slug": slug,
                "title": c.title or slug,
                "description": c.description or "",
                "evaluation_metric": getattr(c, "evaluation_metric", "") or "",
                "deadline": str(getattr(c, "deadline", "") or ""),
                "category": getattr(c, "category", "") or "",
                "reward": getattr(c, "reward", "") or "",
                "url": "https://www.kaggle.com/competitions/%s" % slug,
            }
    raise KaggleError("not_found", "no competition matches slug %r" % slug)


def list_files(slug):
    api = _api()
    try:
        resp = api.competition_list_files(slug, page_size=200)
    except KaggleError:
        raise
    except Exception as e:
        raise _wrap(e, slug) from e
    return [
        {"name": f.name, "bytes": int(getattr(f, "total_bytes", 0) or 0)}
        for f in (getattr(resp, "files", None) or [])
    ]


def download_data(slug, dest_dir):
    """Download the competition bundle into dest_dir and extract it in place.

    Blocking; the caller reports progress by watching the zip grow on disk.
    Returns the list of extracted top-level file names.
    """
    api = _api()
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        api.competition_download_files(slug, path=str(dest), quiet=True)
    except KaggleError:
        raise
    except Exception as e:
        raise _wrap(e, slug) from e
    extracted = []
    for z in sorted(dest.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
            extracted.extend(zf.namelist())
        z.unlink()
    return extracted


def list_kernels(slug, page_size=20):
    api = _api()
    try:
        kernels = api.kernels_list(
            competition=slug, sort_by="voteCount", page_size=page_size
        )
    except KaggleError:
        raise
    except Exception as e:
        raise _wrap(e, slug) from e
    out = []
    for k in kernels or []:
        ref = getattr(k, "ref", None)
        if not ref:
            continue
        out.append(
            {
                "ref": ref,
                "title": getattr(k, "title", "") or ref,
                "author": getattr(k, "author", "") or "",
                "votes": int(getattr(k, "total_votes", 0) or 0),
                "language": str(getattr(k, "language", "") or ""),
                "url": "https://www.kaggle.com/code/%s" % ref,
            }
        )
    return out


def pull_kernel(ref, scratch_dir):
    """Fetch one notebook/script's source; return {ref, source}.

    kernels_pull writes a file (ipynb or script) into scratch_dir; notebooks
    are flattened to their code cells so the summarizer sees plain code.
    """
    api = _api()
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_pull(ref, path=str(scratch))
    except KaggleError:
        raise
    except Exception as e:
        raise _wrap(e, ref) from e
    name = ref.rsplit("/", 1)[-1]
    candidates = [p for p in scratch.iterdir() if p.is_file() and p.stem == name]
    if not candidates:
        candidates = [p for p in scratch.iterdir() if p.is_file()]
    if not candidates:
        raise KaggleError("not_found", "kernels_pull produced no file for %r" % ref)
    path = candidates[0]
    text = path.read_text(errors="replace")
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(text)
            cells = nb.get("cells", [])
            parts = []
            for cell in cells:
                src = "".join(cell.get("source", []))
                if cell.get("cell_type") == "markdown":
                    src = "\n".join("# %s" % line for line in src.splitlines())
                parts.append(src)
            text = "\n\n".join(parts)
        except (ValueError, AttributeError):
            pass  # ship the raw JSON rather than nothing
    path.unlink()
    return {"ref": ref, "source": text}
