"""The EDA slice's tools: run analysis code, keep the dashboard current.

Modeled on tools/exec/experiments.run_experiment: the model writes a Python
script, the tool runs it in a subprocess confined to the workspace, and the
output contract is machine-checked. Here the contract is not a printed score
but a JSON file of dashboard panels (see tools/eda/spec.py) — the script
aggregates with pandas and dumps panels to the path in KC_EDA_OUT; the tool
validates them and upserts them into <workspace>/eda/dashboard.json, which the
UI polls. All three tools are reversible (panels can be replaced or removed),
so none of them is gated.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from core.contracts import err, ok
from core.registry import ToolSpec
from tools.eda.spec import validate_panels

# The dashboard file is polled by the UI every ~1.5s; this cap keeps a
# runaway script from turning every poll into a multi-MB read.
_MAX_DASHBOARD_BYTES = 512 * 1024


def _workspace_root():
    return Path(os.environ.get("KC_WORKSPACE", "workspace"))


def _eda_dir():
    return _workspace_root() / "eda"


def dashboard_path():
    return _eda_dir() / "dashboard.json"


def load_dashboard():
    path = dashboard_path()
    if not path.is_file():
        return {"version": 0, "updated": "", "panels": []}
    try:
        return json.loads(path.read_text())
    except ValueError:
        # Writes are atomic, so this is a hand-edited or truncated file;
        # starting over beats crashing every later tool call.
        return {"version": 0, "updated": "", "panels": []}


def _save_dashboard(dashboard):
    dashboard["version"] = int(dashboard.get("version", 0)) + 1
    dashboard["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = dashboard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dashboard))
    os.replace(tmp, path)
    return dashboard["version"]


def _tail(text, limit=800):
    return (text or "").strip()[-limit:]


def run_eda_script(args):
    code = args["code"]
    timeout_s = args["timeout_s"]

    ws = _workspace_root()
    script_id = uuid.uuid4().hex[:12]
    scripts = _eda_dir() / "scripts"
    out_dir = _eda_dir() / "out"
    script = scripts / (script_id + ".py")
    out_path = out_dir / (script_id + ".json")
    try:
        scripts.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        script.write_text(code)
    except OSError as e:
        return err("exec_failed", "could not write EDA script: %s" % e)

    env = dict(os.environ)
    env["KC_EDA_OUT"] = str(out_path)
    try:
        proc = subprocess.run(
            [sys.executable, str(script.relative_to(ws))],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return err(
            "timeout",
            "EDA script exceeded %ss; sample the data or split the work" % timeout_s,
            retryable=False,
        )

    if proc.returncode != 0:
        return err(
            "exec_failed",
            "EDA script exited %d: %s" % (proc.returncode, _tail(proc.stderr)),
        )

    if not out_path.is_file():
        return err(
            "bad_input",
            "script exited 0 but wrote nothing to KC_EDA_OUT; json.dump your "
            "panel list to the path in os.environ['KC_EDA_OUT']",
            retryable=True,
        )
    try:
        panels = json.loads(out_path.read_text())
    except ValueError as e:
        return err("bad_input", "KC_EDA_OUT is not valid JSON: %s" % e, retryable=True)
    finally:
        out_path.unlink(missing_ok=True)

    problem = validate_panels(panels)
    if problem:
        return err("bad_input", problem, retryable=True)

    dashboard = load_dashboard()
    by_id = {p["id"]: p for p in dashboard["panels"]}
    for panel in panels:
        by_id[panel["id"]] = panel
    merged = list(by_id.values())
    if len(merged) > len(dashboard["panels"]) and len(json.dumps(merged)) > _MAX_DASHBOARD_BYTES:
        return err(
            "bad_input",
            "dashboard would exceed %d bytes; emit smaller panels or remove some first"
            % _MAX_DASHBOARD_BYTES,
            retryable=True,
        )
    dashboard["panels"] = merged
    version = _save_dashboard(dashboard)

    return ok(
        panel_ids=[p["id"] for p in panels],
        version=version,
        stdout_tail=_tail(proc.stdout),
    )


def remove_dashboard_panels(args):
    ids = args["panel_ids"]
    if not ids:
        return err("bad_input", "panel_ids is empty")
    if not all(isinstance(i, str) for i in ids):
        return err("bad_input", "panel_ids must be strings")
    dashboard = load_dashboard()
    have = {p["id"] for p in dashboard["panels"]}
    missing = [i for i in ids if i not in have]
    if missing:
        return err("not_found", "no such panels: %s" % ", ".join(sorted(missing)))
    dashboard["panels"] = [p for p in dashboard["panels"] if p["id"] not in set(ids)]
    version = _save_dashboard(dashboard)
    return ok(removed=list(ids), remaining_ids=[p["id"] for p in dashboard["panels"]], version=version)


def read_dashboard(args):
    dashboard = load_dashboard()
    return ok(
        version=dashboard.get("version", 0),
        panels=[
            {"id": p["id"], "type": p["type"], "title": p["title"]}
            for p in dashboard.get("panels", [])
        ],
    )


RUN_EDA_SCRIPT = ToolSpec(
    name="run_eda_script",
    description=(
        "Run a Python analysis script in the workspace sandbox and publish the "
        "dashboard panels it computes. The script reads datasets from data/ "
        "(pandas and numpy are available), AGGREGATES them (bin, count, "
        "correlate, sample), and json.dumps a list of panel objects to the file "
        "path in os.environ['KC_EDA_OUT']. Panels with an existing id replace "
        "that panel; new ids are added. Call this to create or update charts. "
        "Do NOT call it to train models or compute CV scores, and do NOT emit "
        "raw rows — panels carry aggregates only."
    ),
    parameters={
        "code": {"type": "str", "required": True},
        "timeout_s": {"type": "int", "min": 1, "max": 600, "default": 120},
    },
    fn=run_eda_script,
)

REMOVE_PANELS = ToolSpec(
    name="remove_dashboard_panels",
    description=(
        "Remove dashboard panels by id. Call this only when the user asked for "
        "panels to go away or a panel is superseded and NOT being replaced "
        "under the same id. Do NOT call it before re-creating a panel — "
        "run_eda_script already replaces panels that reuse an id."
    ),
    parameters={
        "panel_ids": {"type": "list", "required": True},
    },
    fn=remove_dashboard_panels,
)

READ_DASHBOARD = ToolSpec(
    name="read_dashboard",
    description=(
        "List the current dashboard's panels (id, type, title — no data). Call "
        "this first when modifying an existing dashboard so ids are reused "
        "rather than duplicated. Do NOT call it to get chart data back — it "
        "returns structure only."
    ),
    parameters={},
    fn=read_dashboard,
)


def register(registry):
    """Register the EDA slice's tools into the given registry."""
    registry.register(RUN_EDA_SCRIPT)
    registry.register(REMOVE_PANELS)
    registry.register(READ_DASHBOARD)
    return registry
