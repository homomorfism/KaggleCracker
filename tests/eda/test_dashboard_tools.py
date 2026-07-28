"""The EDA tools end to end in a tmp workspace: real subprocesses, real files,
and every error branch returning the envelope — never an exception."""

import json

from tools.eda import dashboard
from tools.eda.spec import MAX_SCATTER_POINTS


def _script(panels_expr):
    return (
        "import json, os\n"
        "panels = %s\n"
        "with open(os.environ['KC_EDA_OUT'], 'w') as f:\n"
        "    json.dump(panels, f)\n"
        "print('done')\n" % panels_expr
    )


def _stat(pid, value=1):
    return {
        "id": pid,
        "type": "stat",
        "title": "T",
        "commentary": "",
        "data": {"label": "l", "value": value},
    }


def _run(code, timeout_s=60):
    return dashboard.run_eda_script({"code": code, "timeout_s": timeout_s})


def test_script_publishes_panels_and_bumps_version(eda_workspace):
    out = _run(_script(json.dumps([_stat("rows", 891)])))
    assert out["ok"], out
    assert out["data"]["panel_ids"] == ["rows"]
    assert out["data"]["version"] == 1
    assert "done" in out["data"]["stdout_tail"]

    saved = json.loads((eda_workspace / "eda" / "dashboard.json").read_text())
    assert saved["panels"][0]["data"]["value"] == 891

    # Same id replaces, new id appends; version moves every publish.
    out = _run(_script(json.dumps([_stat("rows", 900), _stat("cols", 12)])))
    assert out["data"]["version"] == 2
    saved = json.loads((eda_workspace / "eda" / "dashboard.json").read_text())
    assert {p["id"]: p["data"]["value"] for p in saved["panels"]} == {"rows": 900, "cols": 12}


def test_crash_is_exec_failed_with_stderr(eda_workspace):
    out = _run("raise SystemExit('boom in cell 3')")
    assert not out["ok"]
    assert out["error"]["kind"] == "exec_failed"
    assert "boom in cell 3" in out["error"]["msg"]


def test_forgetting_the_output_file_is_bad_input(eda_workspace):
    out = _run("print('I computed so much')")
    assert not out["ok"]
    assert out["error"]["kind"] == "bad_input" and out["error"]["retryable"]
    assert "KC_EDA_OUT" in out["error"]["msg"]


def test_invalid_panels_are_rejected_before_any_write(eda_workspace):
    fat = {
        "id": "fat",
        "type": "scatter",
        "title": "T",
        "commentary": "",
        "data": {"points": [[i, i] for i in range(MAX_SCATTER_POINTS + 1)]},
    }
    out = _run(_script(json.dumps([fat])))
    assert not out["ok"] and out["error"]["kind"] == "bad_input"
    assert "downsample" in out["error"]["msg"]
    # Consequence: nothing was published.
    assert not (eda_workspace / "eda" / "dashboard.json").exists()


def test_timeout_maps_to_timeout_kind(eda_workspace):
    out = _run("import time\ntime.sleep(5)", timeout_s=1)
    assert not out["ok"] and out["error"]["kind"] == "timeout"


def test_remove_and_read_dashboard(eda_workspace):
    _run(_script(json.dumps([_stat("a"), _stat("b")])))

    view = dashboard.read_dashboard({})
    assert view["ok"]
    assert [p["id"] for p in view["data"]["panels"]] == ["a", "b"]
    # Structure only — no data payloads leak into the transcript.
    assert "data" not in view["data"]["panels"][0]

    out = dashboard.remove_dashboard_panels({"panel_ids": ["a"]})
    assert out["ok"] and out["data"]["remaining_ids"] == ["b"]

    out = dashboard.remove_dashboard_panels({"panel_ids": ["ghost"]})
    assert not out["ok"] and out["error"]["kind"] == "not_found"


def test_scripts_can_read_workspace_data(eda_workspace):
    (eda_workspace / "data" / "train.csv").write_text("a,b\n1,2\n3,4\n")
    code = (
        "import json, os\n"
        "import pandas as pd\n"
        "df = pd.read_csv('data/train.csv')\n"
        "panels = [{'id': 'rows', 'type': 'stat', 'title': 'Rows',\n"
        "           'commentary': '', 'data': {'label': 'rows', 'value': int(len(df))}}]\n"
        "with open(os.environ['KC_EDA_OUT'], 'w') as f:\n"
        "    json.dump(panels, f)\n"
    )
    out = _run(code)
    assert out["ok"], out
    saved = json.loads((eda_workspace / "eda" / "dashboard.json").read_text())
    assert saved["panels"][0]["data"]["value"] == 2
