"""Competition setup runner: metadata + data download for one project.

Spawned detached by the API server (same pattern as ui.run_analysis):

    python -m ui.run_setup <slug>

All observable state lives in <project>/setup/status.json, written atomically,
so the server answers polls by reading a file — no shared process state, and a
crashed setup leaves an honest last state behind. On success this spawns the
observer and the EDA bootstrap so a fresh project starts working on itself
immediately.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from ui import kaggle_client, projects
from ui.paths import REPO_ROOT


def _setup_dir(slug):
    d = projects.project_dir(slug) / "setup"
    d.mkdir(exist_ok=True)
    return d


def write_status(slug, state, **fields):
    status = dict(
        {
            "state": state,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        **fields,
    )
    path = _setup_dir(slug) / "status.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, path)
    return status


def _failed(slug, err):
    actionable = {
        "no_credentials": "Add Kaggle credentials, then retry.",
        "rules_not_accepted": "Accept the rules at https://www.kaggle.com/c/%s/rules , then retry." % slug,
        "not_found": "Check the competition URL.",
        "rate_limited": "Wait a minute, then retry.",
    }.get(err.kind, "Retry; if it persists, check the setup log.")
    write_status(
        slug,
        "failed",
        error={"kind": err.kind, "msg": err.msg, "actionable": actionable},
    )


def _watch_download(slug, data_dir, total, stop):
    """Report the growing zip's size while kaggle's blocking download runs."""
    while not stop.wait(0.5):
        done = sum(p.stat().st_size for p in data_dir.glob("*.zip") if p.is_file())
        write_status(slug, "downloading", bytes_done=done, bytes_total=total)


def _spawn(module, args, log_path):
    with open(log_path, "a") as log:
        subprocess.Popen(
            [sys.executable, "-m", module, *args],
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def run_setup(slug, spawn_followups=True):
    project_dir = projects.project_dir(slug)
    data_dir = project_dir / "data"
    try:
        write_status(slug, "fetching_metadata")
        meta = kaggle_client.fetch_metadata(slug)
        meta["files"] = kaggle_client.list_files(slug)
        (project_dir / "competition.json").write_text(json.dumps(meta, indent=2))

        total = sum(f["bytes"] for f in meta["files"])
        write_status(slug, "downloading", bytes_done=0, bytes_total=total)
        stop = threading.Event()
        watcher = threading.Thread(
            target=_watch_download, args=(slug, data_dir, total, stop), daemon=True
        )
        watcher.start()
        try:
            kaggle_client.download_data(slug, data_dir)
        finally:
            stop.set()
            watcher.join(timeout=2)

        write_status(slug, "extracting")
        files = [f["name"] for f in projects._data_files(project_dir)]
        write_status(slug, "done", files=files)
    except kaggle_client.KaggleError as e:
        _failed(slug, e)
        return 1

    if spawn_followups:
        # Detached, logged next to the setup log: the project starts observing
        # the community and exploring its data without another click.
        setup = _setup_dir(slug)
        _spawn("ui.run_observer", [slug], setup / "observer_spawn.log")
        _spawn("ui.run_eda", [slug, "--bootstrap"], setup / "eda_spawn.log")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download competition metadata + data.")
    parser.add_argument("slug")
    args = parser.parse_args(argv)
    lock = _setup_dir(args.slug) / "lock"
    try:
        lock.mkdir()
    except FileExistsError:
        print("setup already running for %s" % args.slug)
        return 1
    try:
        return run_setup(args.slug)
    finally:
        shutil.rmtree(lock, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
