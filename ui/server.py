"""The UI's HTTP API: stdlib only, one file.

http.server instead of a web framework on purpose: the repo's no-dependency
rule stays true for every line of Python, and a handful of JSON routes does
not need more. The React frontend (ui/frontend) talks to this over localhost;
CORS is wide open because the server binds 127.0.0.1 and holds nothing secret.

Run from the repo root:  python -m ui.server [--port 8123]
"""

import argparse
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ui import journal, projects
from ui.paths import REPO_ROOT

# Big enough for a Playground train.csv (~63 MB), small enough that a runaway
# upload cannot fill the disk.
_MAX_BODY = 200 * 1024 * 1024

# The built frontend. When it exists, this server IS the whole app on one
# port; when it does not (dev flow: Vite on 5173 proxying /api), any non-API
# GET explains how to get it instead of 404ing cryptically.
_DIST = REPO_ROOT / "ui" / "frontend" / "dist"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".css": "text/css",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".map": "application/json",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


def start_run(slug, pace=0.8, mode="demo"):
    """Claim a run id and launch the analysis as a detached subprocess.

    A subprocess, not a thread: the run mutates KC_WORKSPACE in its own
    environment and can take minutes; the server must keep answering polls
    (and surviving restarts) while it works. Its stdout/stderr land in the
    run directory so a crashed run leaves evidence next to its journal.
    """
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


class Handler(BaseHTTPRequestHandler):
    # http.server logs every request to stderr; at a one-second poll interval
    # that buries real output. The journal is the record of a run, not this.
    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def _route(self, method):
        # ValueError / FileNotFoundError are the projects module's vocabulary
        # for "your fault" / "does not exist"; everything else is a genuine
        # server bug and should produce a loud 500, not be silenced.
        try:
            handled = self._dispatch(method)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except FileNotFoundError as e:
            self._send_json(404, {"error": str(e)})
        else:
            if not handled:
                self._send_json(404, {"error": "no such route: %s %s" % (method, self.path)})

    def _dispatch(self, method):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]

        if parts == ["api", "capabilities"] and method == "GET":
            # Read at request time so exporting the key and restarting the
            # server is enough — no config file to edit.
            self._send_json(200, {"live": bool(os.environ.get("ANTHROPIC_API_KEY"))})
            return True

        if parts == ["api", "projects"]:
            if method == "GET":
                self._send_json(200, {"projects": projects.list_projects()})
            else:
                body = self._json_body()
                meta = projects.create_project(
                    body.get("name", ""), body.get("description", ""), body.get("target", "")
                )
                self._send_json(201, meta)
            return True

        if len(parts) == 3 and parts[:2] == ["api", "projects"] and method == "GET":
            self._send_json(200, projects.get_project(parts[2]))
            return True

        if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "files":
            if method != "POST":
                return False
            name = (query.get("name") or [""])[0]
            saved = projects.save_data_file(parts[2], name, self._raw_body())
            self._send_json(201, saved)
            return True

        if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "plans":
            if method != "GET":
                return False
            self._send_json(200, {"plans": projects.list_plans(parts[2])})
            return True

        if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "findings":
            if method != "GET":
                return False
            flagged_only = (query.get("flagged") or ["1"])[0] != "0"
            self._send_json(
                200, {"findings": projects.list_findings(parts[2], flagged_only=flagged_only)}
            )
            return True

        if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "runs":
            if method == "GET":
                self._send_json(200, {"runs": projects.list_runs(parts[2])})
            else:
                body = self._json_body()
                pace = min(max(float(body.get("pace", 0.8)), 0.0), 10.0)
                mode = body.get("mode", "demo")
                self._send_json(201, start_run(parts[2], pace=pace, mode=mode))
            return True

        if len(parts) == 6 and parts[:2] == ["api", "projects"] and parts[3] == "runs":
            slug, run_id, leaf = parts[2], parts[4], parts[5]
            if leaf == "events" and method == "GET":
                rdir = projects.get_run(slug, run_id)
                since = int((query.get("since") or ["0"])[0])
                path = rdir / "journal.jsonl"
                events = journal.read_events(path, since)
                self._send_json(
                    200,
                    {
                        "events": events,
                        "status": journal.run_status(path),
                        "next_since": events[-1]["seq"] if events else since,
                    },
                )
                return True
            if leaf == "report" and method == "GET":
                rdir = projects.get_run(slug, run_id)
                report = rdir / "report.md"
                if not report.is_file():
                    raise FileNotFoundError("no report for run %r yet" % run_id)
                self._send_text(200, report.read_text())
                return True
            if leaf == "gate" and method == "POST":
                body = self._json_body()
                self._send_json(200, answer_gate(slug, run_id, body.get("answer", "")))
                return True

        if method == "GET" and (not parts or parts[0] != "api"):
            return self._serve_frontend(parts)

        return False

    def _serve_frontend(self, parts):
        if not _DIST.is_dir():
            self._send_text(
                200,
                "KaggleCracker UI API is running, but the frontend is not built.\n\n"
                "Either build it once:   cd ui/frontend && npm run build\n"
                "or develop against it:  cd ui/frontend && npm run dev\n",
            )
            return True
        candidate = (_DIST / "/".join(parts)).resolve() if parts else _DIST / "index.html"
        try:
            candidate.relative_to(_DIST.resolve())
        except ValueError:
            # Path escaped dist (e.g. ..-tricks) — fall back to the app shell
            # rather than serving anything outside the build.
            candidate = _DIST / "index.html"
        if not candidate.is_file():
            # Unknown paths get the shell too: routing lives in the URL hash,
            # so every real page is index.html anyway.
            candidate = _DIST / "index.html"
        body = candidate.read_bytes()
        self.send_response(200)
        self._cors()
        ctype = _CONTENT_TYPES.get(candidate.suffix, "application/octet-stream")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True

    # --- plumbing ------------------------------------------------------------

    def _raw_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > _MAX_BODY:
            raise ValueError("body too large: %d bytes (max %d)" % (length, _MAX_BODY))
        return self.rfile.read(length)

    def _json_body(self):
        raw = self._raw_body()
        if not raw:
            return {}
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError("request body must be JSON")
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status, text):
        body = text.encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(host="127.0.0.1", port=8123):
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="KaggleCracker UI API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    print("kagglecracker-ui listening on http://%s:%d" % server.server_address)
    server.serve_forever()


if __name__ == "__main__":
    main()
