"""Project storage for the UI: a project is a directory shaped like workspace/.

Plain functions raising ValueError / FileNotFoundError on bad input; the HTTP
layer maps those to 400 / 404. The envelope contract is for tools the model
calls — this module is called by the server, never by the loop, so ordinary
exceptions are the honest interface here.
"""

import json
import re
import sqlite3
import time

from ui import journal
from ui.paths import projects_root

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,59}$")
_RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,59}$")
# Uploaded names must be safe both as filesystem names and as the profiler's
# `path` argument: no separators, no leading dot, csv only (nothing else is
# profilable anyway).
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\.csv$")


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]


def project_dir(slug):
    if not _SLUG_RE.match(slug or ""):
        raise ValueError("bad project slug: %r" % (slug,))
    return projects_root() / slug


def create_project(name, description="", target=""):
    slug = slugify(name)
    if not _SLUG_RE.match(slug):
        raise ValueError("project name must contain at least one letter or digit")
    d = projects_root() / slug
    if d.exists():
        raise ValueError("project %r already exists" % slug)
    (d / "data").mkdir(parents=True)
    (d / "plans").mkdir()
    (d / "runs").mkdir()
    meta = {
        "name": name.strip(),
        "slug": slug,
        "description": description.strip(),
        "target": target.strip(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "project.json").write_text(json.dumps(meta, indent=2))
    return dict(meta, files=[], runs=[])


def get_project(slug):
    d = project_dir(slug)
    meta_path = d / "project.json"
    if not meta_path.is_file():
        raise FileNotFoundError("no project %r" % slug)
    meta = json.loads(meta_path.read_text())
    meta["files"] = _data_files(d)
    meta["runs"] = list_runs(slug)
    return meta


def list_projects():
    root = projects_root()
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        # A directory without project.json is not a project — a stray folder,
        # not an error worth failing the whole listing over.
        if (d / "project.json").is_file():
            out.append(get_project(d.name))
    return out


def save_data_file(slug, filename, data):
    if not _FILENAME_RE.match(filename or ""):
        raise ValueError(
            "file name must be a plain *.csv name (letters, digits, . _ -), got %r"
            % (filename,)
        )
    if not data:
        raise ValueError("upload body is empty")
    d = project_dir(slug)
    if not (d / "project.json").is_file():
        raise FileNotFoundError("no project %r" % slug)
    (d / "data" / filename).write_bytes(data)
    return {"name": filename, "bytes": len(data)}


def _data_files(d):
    data = d / "data"
    if not data.is_dir():
        return []
    return [
        {"name": p.name, "bytes": p.stat().st_size}
        for p in sorted(data.iterdir())
        if p.is_file() and p.suffix == ".csv"
    ]


# --- data preparation: plans + findings --------------------------------------


def list_plans(slug):
    """The preprocessing plans the agent has written for this project."""
    d = project_dir(slug)
    if not (d / "project.json").is_file():
        raise FileNotFoundError("no project %r" % slug)
    plans = d / "plans"
    if not plans.is_dir():
        return []
    out = []
    for p in sorted(plans.glob("*.plan.md")):
        out.append(
            {
                "dataset": p.name[: -len(".plan.md")],
                "markdown": p.read_text(),
                "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime)),
            }
        )
    return out


def list_findings(slug, flagged_only=True):
    """The recon findings recorded for this project, straight from its store.

    Read directly from the project's SQLite file in read-only mode rather than
    through tools.recon.store: the store resolves its path via KC_WORKSPACE at
    call time, and mutating that env var per-request inside a threaded server
    would race. The table name and shape are the store's documented schema.
    """
    if not _SLUG_RE.match(slug or ""):
        raise ValueError("bad project slug: %r" % (slug,))
    db = project_dir(slug) / "experiments.db"
    if not db.is_file():
        return []
    sql = (
        "SELECT dataset, check_name, column_name, value, flagged, rows_profiled, "
        "created_at FROM findings"
    )
    if flagged_only:
        sql += " WHERE flagged = 1"
    sql += " ORDER BY dataset, check_name, column_name"
    conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        try:
            rows = conn.execute(sql).fetchall()
        except sqlite3.OperationalError:
            # The db exists but another slice created it and the findings
            # table does not: genuinely zero findings, not a server error.
            return []
        out = []
        for row in rows:
            d = dict(row)
            d["value"] = json.loads(d["value"])
            d["flagged"] = bool(d["flagged"])
            out.append(d)
        return out
    finally:
        conn.close()


# --- runs --------------------------------------------------------------------


def new_run_id(slug):
    """Pick a fresh run id and claim it by creating its directory.

    mkdir is the claim: two callers in the same second cannot both create the
    same directory, so uniqueness needs no lock.
    """
    base = "run-" + time.strftime("%Y%m%d-%H%M%S")
    root = project_dir(slug) / "runs"
    candidate, n = base, 1
    while True:
        try:
            (root / candidate).mkdir(parents=True)
            return candidate
        except FileExistsError:
            n += 1
            candidate = "%s-%d" % (base, n)


def run_dir(slug, run_id):
    if not _RUN_RE.match(run_id or ""):
        raise ValueError("bad run id: %r" % (run_id,))
    return project_dir(slug) / "runs" / run_id


def get_run(slug, run_id):
    d = run_dir(slug, run_id)
    if not d.is_dir():
        raise FileNotFoundError("no run %r in project %r" % (run_id, slug))
    return d


def list_runs(slug):
    root = project_dir(slug) / "runs"
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        out.append(
            {
                "run_id": d.name,
                "status": journal.run_status(d / "journal.jsonl"),
                "has_report": (d / "report.md").is_file(),
            }
        )
    return out
