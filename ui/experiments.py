"""Experiment records for the UI: one directory per experiment.

<project>/experiments_agent/<exp_id>/
    experiment.json   {id, name, description, state, cv_score, ...}
    plan.md           written by propose_experiment_plan
    messages.jsonl    the conversation with the human
    code/attempt-N.py every training script the agent ran, extracted from the
                      turn journals so the human can read exactly what executed
    runs/turn-*/      journals, one per agent turn

States: draft (planning / Q&A) -> plan_proposed -> finished. "running" is not
a stored state — a held lock means a turn is in flight, and the API reports
that separately, so a crashed turn can never wedge an experiment.

Same error vocabulary as ui/projects.py: ValueError for bad input,
FileNotFoundError for absent things; the HTTP layer maps them.
"""

import json
import re
import time

from ui import projects

_EXP_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,59}$")
_TURN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,59}$")


def _root(slug):
    return projects.project_dir(slug) / "experiments_agent"


def is_running(d):
    """A turn is in flight: its lock is held, or it was just spawned and has
    not claimed the lock yet (the marker bridges the subprocess-startup gap so
    the UI never flashes 'idle' right after a send). A stale marker — spawn
    crashed before ever locking — expires rather than wedging the experiment."""
    if (d / "lock").is_dir():
        return True
    marker = d / "spawn_pending"
    try:
        return time.time() - marker.stat().st_mtime < 60
    except OSError:
        return False


def experiment_dir(slug, exp_id):
    if not _EXP_RE.match(exp_id or ""):
        raise ValueError("bad experiment id: %r" % (exp_id,))
    return _root(slug) / exp_id


def create_experiment(slug, prompt=""):
    """Claim a new experiment directory and seed its record + conversation.

    An empty prompt means "you decide": the agent picks the most promising
    experiment itself from the project's insights.
    """
    projects.get_project(slug)
    root = _root(slug)
    base = "exp-" + time.strftime("%Y%m%d-%H%M%S")
    exp_id, n = base, 1
    while True:
        try:
            (root / exp_id).mkdir(parents=True)
            break
        except FileExistsError:
            n += 1
            exp_id = "%s-%d" % (base, n)
    d = root / exp_id
    (d / "runs").mkdir()
    (d / "code").mkdir()
    meta = {
        "id": exp_id,
        "name": "untitled experiment",
        "description": prompt.strip(),
        "state": "draft",
        "cv_score": None,
        "result_summary": "",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "experiment.json").write_text(json.dumps(meta, indent=2))
    first = prompt.strip() or (
        "Design the most promising experiment yourself from the dataset "
        "insights and community knowledge. Ask me only what you genuinely "
        "cannot decide."
    )
    append_message(slug, exp_id, "user", first)
    return meta


def get_meta(slug, exp_id):
    d = experiment_dir(slug, exp_id)
    path = d / "experiment.json"
    if not path.is_file():
        raise FileNotFoundError("no experiment %r in project %r" % (exp_id, slug))
    return json.loads(path.read_text())


def list_experiments(slug):
    projects.get_project(slug)
    root = _root(slug)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if (d / "experiment.json").is_file():
            meta = json.loads((d / "experiment.json").read_text())
            meta["running"] = is_running(d)
            out.append(meta)
    return out


def get_detail(slug, exp_id):
    from ui import journal

    d = experiment_dir(slug, exp_id)
    meta = get_meta(slug, exp_id)
    plan = d / "plan.md"
    code = []
    code_dir = d / "code"
    if code_dir.is_dir():
        for p in sorted(code_dir.glob("*.py")):
            code.append({"name": p.name, "source": p.read_text()})
    turns = []
    runs_root = d / "runs"
    if runs_root.is_dir():
        for t in sorted(p.name for p in runs_root.iterdir() if p.is_dir()):
            turns.append(
                {"id": t, "status": journal.run_status(runs_root / t / "journal.jsonl")}
            )
    return {
        "experiment": dict(meta, running=is_running(d)),
        "plan_md": plan.read_text() if plan.is_file() else "",
        "messages": read_messages(slug, exp_id),
        "latest_turn": turns[-1]["id"] if turns else None,
        "turns": turns,
        "code": code,
    }


# --- conversation -------------------------------------------------------------


def read_messages(slug, exp_id):
    path = experiment_dir(slug, exp_id) / "messages.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def append_message(slug, exp_id, role, text):
    d = experiment_dir(slug, exp_id)
    if not (d / "experiment.json").is_file():
        raise FileNotFoundError("no experiment %r in project %r" % (exp_id, slug))
    messages = read_messages(slug, exp_id)
    record = {
        "seq": messages[-1]["seq"] + 1 if messages else 1,
        "role": role,
        "text": text,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with (d / "messages.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


# --- turns --------------------------------------------------------------------


def new_turn_dir(slug, exp_id):
    root = experiment_dir(slug, exp_id) / "runs"
    root.mkdir(exist_ok=True)
    base = "turn-" + time.strftime("%Y%m%d-%H%M%S")
    turn, n = base, 1
    while True:
        try:
            (root / turn).mkdir()
            return root / turn
        except FileExistsError:
            n += 1
            turn = "%s-%d" % (base, n)


def latest_turn(slug, exp_id):
    root = experiment_dir(slug, exp_id) / "runs"
    if not root.is_dir():
        return None
    turns = sorted(p.name for p in root.iterdir() if p.is_dir())
    return turns[-1] if turns else None


def turn_journal(slug, exp_id, turn):
    if not _TURN_RE.match(turn or ""):
        raise ValueError("bad turn id: %r" % (turn,))
    d = experiment_dir(slug, exp_id) / "runs" / turn
    if not d.is_dir():
        raise FileNotFoundError("no turn %r in experiment %r" % (turn, exp_id))
    return d / "journal.jsonl"


def save_code_attempts(slug, exp_id, journal_events):
    """Copy every run_experiment script out of a turn journal into code/.

    The journal's act events carry the exact `code` argument the sandbox
    executed — extracting here (not trusting the model to file its own code)
    guarantees what the human reads is what actually ran.
    """
    code_dir = experiment_dir(slug, exp_id) / "code"
    code_dir.mkdir(exist_ok=True)
    existing = len(list(code_dir.glob("*.py")))
    saved = []
    for event in journal_events:
        if event.get("type") != "act" or event.get("tool") != "run_experiment":
            continue
        code = (event.get("args") or {}).get("code")
        if not isinstance(code, str) or not code.strip():
            continue
        existing += 1
        name = "attempt-%02d.py" % existing
        (code_dir / name).write_text(code)
        saved.append(name)
    return saved
