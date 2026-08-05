"""Read-only viewer over the tree, plus the one write the human is allowed to make.

    worker process ──writes──► SQLite (WAL) ◄──reads── web process
                                              │
                                              └── POST /approve/<id>  the ONE write

The asymmetry is deliberate. The worker is the sole writer of nodes; the web
process opens the database read-only for everything except approving a
submission. WAL mode is what lets both run at once without the reader blocking
the writer mid-node.

Layout follows docs/design.html (light/dark, numbered sections). Panels that
the mockup shows but this system cannot honestly populate — inference time,
a public-LB estimate, a feature store — are left out rather than filled with
plausible-looking numbers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from kagglecracker.config import Settings, load_settings
from kagglecracker.db.store import Node, Protocol, ProtocolStore, connect
from kagglecracker.engine.contract import ContractError, SubmissionValidator
from kagglecracker.engine.ranking import RankedNode, leaderboard

WEB_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

STATUS_CLASS = {
    "ok": "ok",
    "error": "bad",
    "timeout": "warn",
    "truncated": "warn",
    "contract_violation": "warn",
    "infra_error": "infra",
}


def build_tree(nodes: list[Node], best_id: int | None = None) -> list[dict]:
    """Nested node structure the template renders as HTML cards, server-side.

    Module-level pure function so its shape can be tested without standing up
    the app. Each entry: {"node", "cls", "scored", "children"}.
    """
    entries: dict[int, dict] = {}
    for n in nodes:
        # Key off status, not off cv_score being present — the same rule
        # ranking.py enforces. A node that failed must never display a score,
        # however it came to have one.
        scored = n.status == "ok" and n.cv_score is not None
        if n.id == best_id:
            cls = "best"
        elif n.status == "ok":
            cls = "ok"
        elif n.status == "infra_error":
            cls = "infra"
        else:
            cls = "bad"
        entries[n.id] = {"node": n, "cls": cls, "scored": scored, "children": []}

    roots: list[dict] = []
    for n in nodes:
        if n.parent_id and n.parent_id in entries:
            entries[n.parent_id]["children"].append(entries[n.id])
        else:
            roots.append(entries[n.id])
    return roots


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="KaggleCracker")
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    def read() -> sqlite3.Connection:
        """A fresh read-only connection per request.

        Read-only is enforced at the connection, not by convention — the viewer
        cannot corrupt a running search even if a handler is wrong.
        """
        return connect(settings.db_path, read_only=True)

    # -- data ------------------------------------------------------------

    def current_protocol(conn: sqlite3.Connection) -> Protocol | None:
        return ProtocolStore(conn).current()

    def all_nodes(conn: sqlite3.Connection, protocol: Protocol) -> list[Node]:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE protocol_id = ? ORDER BY id", (protocol.id,)
        )
        return [Node.from_row(r) for r in rows]

    def run_summary(conn: sqlite3.Connection) -> sqlite3.Row | None:
        return conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()

    def worker_state(conn: sqlite3.Connection) -> str:
        row = conn.execute(
            "SELECT pid, (julianday('now') - julianday(heartbeat_at)) * 86400 AS age "
            "FROM worker_lock WHERE id = 1"
        ).fetchone()
        if row is None:
            return "idle"
        if (row["age"] or 0) > settings.worker_lock_ttl_s:
            return f"stale (pid {row['pid']}, no heartbeat for {row['age']:.0f}s)"
        return f"running (pid {row['pid']})"

    def context_docs() -> list[tuple[str, str]]:
        docs = []
        eda = settings.context_dir / "eda_findings.md"
        if eda.is_file():
            docs.append(("EDA findings", eda.read_text()))
        d = settings.context_dir / "discussions"
        if d.is_dir():
            for p in sorted(d.glob("*.md"), reverse=True):
                docs.append((f"Discussions {p.stem}", p.read_text()))
        return docs

    def submission_for(conn: sqlite3.Connection, node_id: int) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM submissions WHERE node_id = ?", (node_id,)
        ).fetchone()

    # -- routes ----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, node: int | None = None) -> HTMLResponse:
        conn = read()
        protocol = current_protocol(conn)
        if protocol is None:
            return templates.TemplateResponse(
                request,
                "empty.html",
                {
                    "settings": settings,  # base.html renders the header from it
                    "message": "No CV protocol seeded. Run `kagglecracker init`.",
                },
            )

        nodes = all_nodes(conn, protocol)
        board: list[RankedNode] = leaderboard(conn, protocol)
        best = board[0].node if board else None
        selected = next((n for n in nodes if n.id == node), None) or best
        if selected is None and nodes:
            selected = nodes[-1]
        selected_rank = (
            next((r for r in board if r.node.id == selected.id), None) if selected else None
        )

        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "settings": settings,
                "protocol": protocol,
                "nodes": nodes,
                "board": board,
                "best": best,
                "selected": selected,
                "selected_rank": selected_rank,
                "selected_submission": submission_for(conn, selected.id) if selected else None,
                "tree": build_tree(nodes, best.id if best else None),
                "run": run_summary(conn),
                "worker": worker_state(conn),
                "context_docs": context_docs(),
                "stdout_tail": read_stdout(selected),
                "status_class": STATUS_CLASS,
            },
        )

    def read_stdout(node: Node | None, lines: int = 40) -> str:
        if node is None or not node.stdout_path:
            return ""
        p = Path(node.stdout_path)
        if not p.is_file():
            return ""
        return "\n".join(p.read_text().splitlines()[-lines:])

    @app.post("/approve/{node_id}", response_class=HTMLResponse)
    def approve(request: Request, node_id: int, confirm: str = Form("")) -> HTMLResponse:
        """Validate a node's submission and record the intent to submit.

        Day 5 stops at recording. The actual `kaggle competitions submit` call is
        Day 6 — the point of splitting them is that the validation and the
        idempotency guard exist and are testable before anything can spend one
        of the day's submissions.
        """
        conn = connect(settings.db_path)  # the one writable path
        node = Node.from_row(
            conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        )

        if node.status != "ok":
            return _result(
                request,
                False,
                f"node {node_id} has status {node.status}; only a scored node can be submitted",
            )

        existing = submission_for(conn, node_id)
        if existing is not None:
            # UNIQUE(node_id) makes this structural, but reporting it beats
            # letting the insert raise — a double-clicked button is the
            # expected case, not an error.
            return _result(
                request, False,
                f"node {node_id} was already submitted at {existing['submitted_at']} "
                f"(state {existing['state']}). Not spending a second submission.",
            )

        run_dir = Path(node.stdout_path).parent if node.stdout_path else None
        if run_dir is None or not (run_dir / "submission.csv").is_file():
            return _result(request, False, f"no submission.csv on disk for node {node_id}")

        try:
            SubmissionValidator(settings.data_dir / "sample_submission.csv").validate(run_dir)
        except ContractError as exc:
            return _result(request, False, f"submission failed validation: {exc}")

        with conn:
            conn.execute(
                "INSERT INTO submissions (node_id, message, state) VALUES (?, ?, 'pending')",
                (node_id, f"KaggleCracker node {node_id} cv={node.cv_score:.5f}"),
            )
        return _result(
            request, True,
            f"node {node_id} validated and queued (cv {node.cv_score:.5f}). "
            f"Run `kagglecracker submit` to send it to Kaggle.",
        )

    def _result(request: Request, ok: bool, message: str) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_approve_result.html", {"ok": ok, "message": message}
        )

    return app


app = create_app()
