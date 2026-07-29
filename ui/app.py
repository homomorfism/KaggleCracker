"""The UI's HTTP API, now on FastAPI.

Route contract and JSON shapes are carried over from the retired stdlib
server verbatim so the frontend's api.ts keeps working. Long jobs (setup,
analysis runs, observer, EDA turns) are still detached subprocesses journaling
to files — the server holds no job state and answers every poll by reading
disk, which is what lets it restart mid-run without losing anything.

Run from the repo root:  python -m ui.app [--port 8123]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from ui import experiments, journal, kaggle_client, projects
from ui.paths import REPO_ROOT

# Big enough for a Playground train.csv, small enough that a runaway upload
# cannot fill the disk.
_MAX_BODY = 200 * 1024 * 1024

_DIST = REPO_ROOT / "ui" / "frontend" / "dist"

app = FastAPI(title="KaggleCracker UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(ValueError)
async def _bad_request(request, exc):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(FileNotFoundError)
async def _not_found(request, exc):
    return JSONResponse(status_code=404, content={"error": str(exc)})


@app.exception_handler(kaggle_client.KaggleError)
async def _kaggle_error(request, exc):
    status = 404 if exc.kind == "not_found" else 400
    return JSONResponse(status_code=status, content={"error": exc.msg, "kind": exc.kind})


def _spawn(module, args, log_path):
    """Detached worker subprocess, stdout+stderr to its own log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        subprocess.Popen(
            [sys.executable, "-m", module, *args],
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def start_run(slug, pace=0.8, mode="demo"):
    """Claim a run id and launch the recon analysis as a detached subprocess."""
    if mode not in ("demo", "live"):
        raise ValueError("mode must be 'demo' or 'live', got %r" % (mode,))
    if mode == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError(
            "live mode needs ANTHROPIC_API_KEY in the server's environment; "
            "set it and restart the server, or run in demo mode"
        )
    project = projects.get_project(slug)
    if not project["files"]:
        raise ValueError("project has no data files; upload a csv before starting an analysis")
    run_id = projects.new_run_id(slug)
    rdir = projects.run_dir(slug, run_id)
    with (rdir / "runner.log").open("w") as log:
        subprocess.Popen(
            [sys.executable, "-m", "ui.run_analysis", slug, run_id,
             "--mode", mode, "--pace", str(pace)],
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return {"run_id": run_id, "status": "running", "mode": mode}


def answer_gate(slug, run_id, answer):
    """Deliver a human's gate answer to a waiting live run.

    Written atomically (tmp + replace) so the polling runner can never read a
    half-written answer. What counts as approval is decided by core/gate.py
    alone — this endpoint transports text, it does not interpret it.
    """
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    if len(answer) > 100:
        raise ValueError("answer is suspiciously long; the gate wants 'yes' or a refusal")
    rdir = projects.get_run(slug, run_id)
    tmp = rdir / "gate_answer.txt.tmp"
    tmp.write_text(answer)
    os.replace(tmp, rdir / "gate_answer.txt")
    return {"answer": answer}


async def _json_body(request):
    raw = await request.body()
    if len(raw) > _MAX_BODY:
        raise ValueError("body too large: %d bytes (max %d)" % (len(raw), _MAX_BODY))
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("request body must be JSON")
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


# --- api routes ---------------------------------------------------------------


@app.get("/api/capabilities")
def capabilities():
    # Read at request time so exporting a key and restarting is enough.
    return {
        "live": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "kaggle": kaggle_client.has_credentials(),
    }


@app.get("/api/projects")
def list_projects():
    return {"projects": projects.list_projects()}


@app.post("/api/projects", status_code=201)
async def create_project(request: Request):
    body = await _json_body(request)
    url = (body.get("competition_url") or "").strip()
    if url:
        slug = kaggle_client.parse_competition_url(url)
        meta = projects.create_project(
            body.get("name", "") or slug,
            body.get("description", ""),
            body.get("target", ""),
            competition=slug,
        )
        setup_dir = projects.project_dir(slug) / "setup"
        _spawn("ui.run_setup", [slug], setup_dir / "setup.log")
        meta["setup_state"] = "pending"
        return meta
    return projects.create_project(
        body.get("name", ""), body.get("description", ""), body.get("target", "")
    )


@app.get("/api/projects/{slug}")
def get_project(slug: str):
    return projects.get_project(slug)


@app.post("/api/projects/{slug}/files", status_code=201)
async def upload_file(slug: str, request: Request, name: str = ""):
    raw = await request.body()
    if len(raw) > _MAX_BODY:
        raise ValueError("body too large: %d bytes (max %d)" % (len(raw), _MAX_BODY))
    return projects.save_data_file(slug, name, raw)


@app.get("/api/projects/{slug}/competition")
def competition(slug: str):
    return projects.competition_meta(slug)


@app.get("/api/projects/{slug}/setup")
def setup_status(slug: str):
    projects.project_dir(slug)  # validates the slug shape
    return projects.setup_status(slug)


@app.post("/api/projects/{slug}/setup/retry")
def setup_retry(slug: str):
    project = projects.get_project(slug)
    if not project.get("competition"):
        raise ValueError("project %r is not linked to a Kaggle competition" % slug)
    setup_dir = projects.project_dir(slug) / "setup"
    if (setup_dir / "lock").is_dir():
        return JSONResponse(status_code=409, content={"error": "setup already running"})
    _spawn("ui.run_setup", [slug], setup_dir / "setup.log")
    return {"state": "pending"}


@app.get("/api/projects/{slug}/plans")
def plans(slug: str):
    return {"plans": projects.list_plans(slug)}


@app.get("/api/projects/{slug}/findings")
def findings(slug: str, flagged: str = "1"):
    return {"findings": projects.list_findings(slug, flagged_only=flagged != "0")}


# --- data preparation runs ----------------------------------------------------


def _prep_running(prep_dir):
    """Lock held, or spawned moments ago and not yet locked — the same
    spawn-race guard the EDA agent uses."""
    if (prep_dir / "lock").is_dir():
        return True
    marker = prep_dir / "spawn_pending"
    try:
        return time.time() - marker.stat().st_mtime < 60
    except OSError:
        return False


@app.get("/api/projects/{slug}/prep")
def prep_status(slug: str, since: int = 0):
    projects.get_project(slug)
    prep_dir = projects.project_dir(slug) / "prep"
    path = prep_dir / "journal.jsonl"
    if _prep_running(prep_dir):
        status = "running"
    elif path.is_file():
        status = journal.run_status(path)
    else:
        status = "none"
    events = journal.read_events(path, since)
    manifest = {}
    manifest_path = prep_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError:
            pass  # torn read mid-replace; the writer is atomic, transient
    # The worker's own stdout/stderr, for the interface's raw-logs view.
    # Usually empty — pandas warnings or a traceback land here when not.
    log_path = prep_dir / "prep.log"
    log_tail = ""
    if log_path.is_file():
        log_tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-100:])
    return {
        "status": status,
        "events": events,
        "next_since": events[-1]["seq"] if events else since,
        "manifest": manifest,
        "log": log_tail,
    }


@app.post("/api/projects/{slug}/prep/run")
async def prep_run(slug: str, request: Request):
    projects.get_project(slug)
    if not projects.list_plans(slug):
        raise ValueError("no preprocessing plan yet — run a recon analysis first")
    prep_dir = projects.project_dir(slug) / "prep"
    if _prep_running(prep_dir):
        return JSONResponse(status_code=409, content={"error": "preparation already running"})
    body = await _json_body(request)
    pace = min(max(float(body.get("pace", 0.6)), 0.0), 10.0)
    prep_dir.mkdir(exist_ok=True)
    (prep_dir / "spawn_pending").touch()
    _spawn("ui.run_prep", [slug, "--pace", str(pace)], prep_dir / "prep.log")
    return {"status": "running"}


@app.get("/api/projects/{slug}/prep/files/{name}")
def prep_file(slug: str, name: str):
    if not projects._FILENAME_RE.match(name or ""):
        raise ValueError("bad file name: %r" % (name,))
    path = projects.project_dir(slug) / "prep" / name
    if not path.is_file():
        raise FileNotFoundError("no prepared file %r" % name)
    return FileResponse(path, media_type="text/csv", filename=name)


def _eda_running(d):
    """Lock held, or spawned moments ago and not yet locked (see
    ui/experiments.is_running for the reasoning)."""
    if (d / "eda" / "lock").is_dir():
        return True
    marker = d / "eda" / "spawn_pending"
    try:
        return time.time() - marker.stat().st_mtime < 60
    except OSError:
        return False


def _eda_latest_run(d):
    runs_root = d / "eda" / "runs"
    if not runs_root.is_dir():
        return None
    turns = sorted(p.name for p in runs_root.iterdir() if p.is_dir())
    return turns[-1] if turns else None


@app.get("/api/projects/{slug}/eda")
def eda(slug: str, version: int = -1):
    """Dashboard + chat + agent status in one poll.

    ``version`` lets the poller skip the payload when nothing changed — but
    only while the agent is idle, since chat messages and journal events move
    without bumping the dashboard version.
    """
    projects.get_project(slug)
    d = projects.project_dir(slug)
    running = _eda_running(d)
    dash_path = d / "eda" / "dashboard.json"
    dashboard = {"version": 0, "updated": "", "panels": []}
    if dash_path.is_file():
        try:
            dashboard = json.loads(dash_path.read_text())
        except ValueError:
            pass
    if not running and version == dashboard.get("version", 0):
        return {"unchanged": True, "version": version}

    from ui.run_eda import read_messages

    return {
        "dashboard": dashboard,
        "chat": read_messages(slug),
        "agent": {"status": "running" if running else "idle", "run_id": _eda_latest_run(d)},
    }


@app.post("/api/projects/{slug}/eda/messages", status_code=201)
async def eda_message(slug: str, request: Request):
    projects.get_project(slug)
    body = await _json_body(request)
    text = (body.get("text") or "").strip()
    if not text:
        raise ValueError("message text is empty")
    if len(text) > 4000:
        raise ValueError("message is too long (max 4000 chars)")
    d = projects.project_dir(slug)
    if (d / "eda" / "lock").is_dir():
        return JSONResponse(
            status_code=409, content={"error": "the EDA agent is already working on a turn"}
        )
    from ui.run_eda import append_message

    record = append_message(slug, "user", text)
    (d / "eda").mkdir(exist_ok=True)
    (d / "eda" / "spawn_pending").touch()
    _spawn("ui.run_eda", [slug], d / "eda" / "eda.log")
    return {"message": record, "agent": {"status": "running"}}


@app.get("/api/projects/{slug}/eda/runs/{run_id}/events")
def eda_events(slug: str, run_id: str, since: int = 0):
    projects.get_project(slug)
    if not projects._RUN_RE.match(run_id or ""):
        raise ValueError("bad run id: %r" % (run_id,))
    path = projects.project_dir(slug) / "eda" / "runs" / run_id / "journal.jsonl"
    if not path.parent.is_dir():
        raise FileNotFoundError("no EDA turn %r in project %r" % (run_id, slug))
    events = journal.read_events(path, since)
    return {
        "events": events,
        "status": journal.run_status(path),
        "next_since": events[-1]["seq"] if events else since,
    }


@app.get("/api/projects/{slug}/observer")
def observer(slug: str):
    """Everything the Discussions page needs in one poll."""
    projects.get_project(slug)
    d = projects.project_dir(slug)
    journal_path = d / "observer" / "journal.jsonl"
    # No journal yet means the observer has never run — distinct from a
    # journal-less subprocess that is about to write ("running").
    status = journal.run_status(journal_path) if journal_path.is_file() else "none"
    notebooks = []
    store = d / "knowledge" / "notebooks.jsonl"
    if store.is_file():
        for line in store.read_text().splitlines():
            try:
                notebooks.append(json.loads(line))
            except ValueError:
                continue
    notebooks.sort(key=lambda r: r.get("votes", 0), reverse=True)
    summary_path = d / "knowledge" / "summary.md"
    return {
        "status": status,
        "notebooks": notebooks,
        "summary_md": summary_path.read_text() if summary_path.is_file() else "",
    }


@app.post("/api/projects/{slug}/observer/refresh")
def observer_refresh(slug: str):
    projects.get_project(slug)
    obs_dir = projects.project_dir(slug) / "observer"
    if (obs_dir / "lock").is_dir():
        return JSONResponse(status_code=409, content={"error": "observer already running"})
    _spawn("ui.run_observer", [slug], obs_dir / "observer.log")
    return {"status": "running"}


@app.get("/api/projects/{slug}/observer/events")
def observer_events(slug: str, since: int = 0):
    projects.get_project(slug)
    path = projects.project_dir(slug) / "observer" / "journal.jsonl"
    events = journal.read_events(path, since)
    return {
        "events": events,
        "status": journal.run_status(path) if path.is_file() else "none",
        "next_since": events[-1]["seq"] if events else since,
    }


@app.get("/api/projects/{slug}/experiments")
def list_experiments(slug: str):
    return {"experiments": experiments.list_experiments(slug)}


@app.post("/api/projects/{slug}/experiments", status_code=201)
async def create_experiment(slug: str, request: Request):
    body = await _json_body(request)
    prompt = (body.get("prompt") or "").strip()
    if len(prompt) > 4000:
        raise ValueError("prompt is too long (max 4000 chars)")
    meta = experiments.create_experiment(slug, prompt)
    exp_dir = experiments.experiment_dir(slug, meta["id"])
    (exp_dir / "spawn_pending").touch()
    _spawn("ui.run_experiment_turn", [slug, meta["id"]], exp_dir / "turns.log")
    return dict(meta, running=True)


@app.get("/api/projects/{slug}/experiments/{exp_id}")
def experiment_detail(slug: str, exp_id: str):
    return experiments.get_detail(slug, exp_id)


@app.post("/api/projects/{slug}/experiments/{exp_id}/messages", status_code=201)
async def experiment_message(slug: str, exp_id: str, request: Request):
    body = await _json_body(request)
    text = (body.get("text") or "").strip()
    if not text:
        raise ValueError("message text is empty")
    if len(text) > 4000:
        raise ValueError("message is too long (max 4000 chars)")
    exp_dir = experiments.experiment_dir(slug, exp_id)
    if (exp_dir / "lock").is_dir():
        return JSONResponse(
            status_code=409, content={"error": "the experiment agent is already working"}
        )
    record = experiments.append_message(slug, exp_id, "user", text)
    (exp_dir / "spawn_pending").touch()
    _spawn("ui.run_experiment_turn", [slug, exp_id], exp_dir / "turns.log")
    return {"message": record, "running": True}


@app.post("/api/projects/{slug}/experiments/{exp_id}/stop")
def experiment_stop(slug: str, exp_id: str):
    """Kill the running turn. The conversation survives; the human resumes by
    sending another message."""
    experiments.get_meta(slug, exp_id)
    lock = experiments.experiment_dir(slug, exp_id) / "lock"
    pid_file = lock / "pid"
    if not pid_file.is_file():
        return {"stopped": False, "reason": "no turn is running"}
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 15)
    except (ValueError, ProcessLookupError):
        pass
    shutil.rmtree(lock, ignore_errors=True)
    experiments.append_message(slug, exp_id, "assistant", "(turn stopped by you)")
    return {"stopped": True}


@app.get("/api/projects/{slug}/experiments/{exp_id}/runs/{turn}/events")
def experiment_events(slug: str, exp_id: str, turn: str, since: int = 0):
    path = experiments.turn_journal(slug, exp_id, turn)
    events = journal.read_events(path, since)
    return {
        "events": events,
        "status": journal.run_status(path),
        "next_since": events[-1]["seq"] if events else since,
    }


@app.get("/api/projects/{slug}/leaderboard")
def leaderboard(slug: str):
    """Recorded experiment results for this project, best first."""
    projects.get_project(slug)
    path = projects.project_dir(slug) / "experiments" / "leaderboard.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    rows.sort(key=lambda r: r.get("cv_score", float("-inf")), reverse=True)
    return {"entries": rows}


@app.get("/api/projects/{slug}/runs")
def list_runs(slug: str):
    return {"runs": projects.list_runs(slug)}


@app.post("/api/projects/{slug}/runs", status_code=201)
async def create_run(slug: str, request: Request):
    body = await _json_body(request)
    pace = min(max(float(body.get("pace", 0.8)), 0.0), 10.0)
    mode = body.get("mode", "demo")
    return start_run(slug, pace=pace, mode=mode)


@app.get("/api/projects/{slug}/runs/{run_id}/events")
def run_events(slug: str, run_id: str, since: int = 0):
    rdir = projects.get_run(slug, run_id)
    path = rdir / "journal.jsonl"
    events = journal.read_events(path, since)
    return {
        "events": events,
        "status": journal.run_status(path),
        "next_since": events[-1]["seq"] if events else since,
    }


@app.get("/api/projects/{slug}/runs/{run_id}/report")
def run_report(slug: str, run_id: str):
    rdir = projects.get_run(slug, run_id)
    report = rdir / "report.md"
    if not report.is_file():
        raise FileNotFoundError("no report for run %r yet" % run_id)
    return PlainTextResponse(report.read_text(), media_type="text/markdown; charset=utf-8")


@app.post("/api/projects/{slug}/runs/{run_id}/gate")
async def run_gate(slug: str, run_id: str, request: Request):
    body = await _json_body(request)
    return answer_gate(slug, run_id, body.get("answer", ""))


# --- frontend -----------------------------------------------------------------


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    if not _DIST.is_dir():
        return PlainTextResponse(
            "KaggleCracker UI API is running, but the frontend is not built.\n\n"
            "Either build it once:   cd ui/frontend && npm run build\n"
            "or develop against it:  cd ui/frontend && npm run dev\n"
        )
    candidate = (_DIST / path).resolve() if path else _DIST / "index.html"
    try:
        candidate.relative_to(_DIST.resolve())
    except ValueError:
        # Path escaped dist (..-tricks) — serve the app shell, never anything
        # outside the build.
        candidate = _DIST / "index.html"
    if not candidate.is_file():
        # Unknown paths get the shell too: routing lives in the URL hash.
        candidate = _DIST / "index.html"
    return FileResponse(candidate)


def main(argv=None):
    import uvicorn

    parser = argparse.ArgumentParser(description="KaggleCracker UI API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
