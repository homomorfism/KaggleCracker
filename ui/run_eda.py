"""One EDA chat turn: read the conversation, run the agent, update the dashboard.

    python -m ui.run_eda <slug> [--bootstrap]

The conversation is the durable thing (<project>/eda/chat/messages.jsonl);
each turn is a stateless subprocess primed from that history plus the current
dashboard structure, so nothing has to stay alive between messages and a
server restart loses nothing. --bootstrap injects the canned first request
that builds the initial dashboard after setup finishes.

Unlike the recon runner this loop has no gate: every EDA tool is reversible
(panels can be replaced or removed), so there is nothing a human needs to
approve mid-turn.
"""

import argparse
import json
import os
import shutil
import sys
import time

from core.registry import Registry
from tools.corpus import search as corpus_search
from tools.eda import dashboard as eda_dashboard
from tools.exec import shell
from ui import projects
from ui.journal import JournalWriter
from ui.live_model import DEFAULT_MODEL, AnthropicModel
from ui.runner import run_with_journal

_HISTORY_WINDOW = 12

_BOOTSTRAP_MESSAGE = (
    "Build the initial EDA dashboard for this competition: dataset shapes and "
    "dtypes, missingness per column, target distribution, distributions of the "
    "most informative features, a correlation heatmap of numeric columns, and "
    "train/test drift for the strongest features. Finish with a short markdown "
    "panel summarizing what a modeller must watch out for."
)


def _chat_dir(slug):
    d = projects.project_dir(slug) / "eda" / "chat"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _messages_path(slug):
    return _chat_dir(slug) / "messages.jsonl"


def read_messages(slug):
    path = _messages_path(slug)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def append_message(slug, role, text):
    messages = read_messages(slug)
    seq = messages[-1]["seq"] + 1 if messages else 1
    record = {
        "seq": seq,
        "role": role,
        "text": text,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with _messages_path(slug).open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _task(slug, project, messages):
    try:
        meta = projects.competition_meta(slug)
    except FileNotFoundError:
        meta = {}
    files = ", ".join(f["name"] for f in project["files"]) or "(none)"
    current = eda_dashboard.read_dashboard({})["data"]

    lines = [
        "You are the EDA agent for a Kaggle competition project. You maintain a",
        "live dashboard the human is looking at RIGHT NOW; your job this turn is",
        "to satisfy their latest message by updating it.",
        "",
        "Competition: %s" % (meta.get("title") or project["name"]),
    ]
    if meta.get("evaluation_metric"):
        lines.append("Evaluation metric: %s" % meta["evaluation_metric"])
    if project.get("target"):
        lines.append("Target column: %r" % project["target"])
    lines += [
        "Files under data/: %s" % files,
        "Dashboard now: version %d, panels %s"
        % (
            current["version"],
            ", ".join("%s(%s)" % (p["id"], p["type"]) for p in current["panels"]) or "(empty)",
        ),
        "",
        "Conversation so far (oldest first):",
    ]
    for m in messages[-_HISTORY_WINDOW:]:
        lines.append("%s: %s" % (m["role"].upper(), m["text"]))
    lines += [
        "",
        "Rules:",
        "- Use run_eda_script to compute panels; aggregate with pandas, never",
        "  emit raw rows. Reuse a panel id to replace that panel.",
        "- pandas, numpy, scikit-learn, lightgbm, xgboost are installed; for",
        "  anything else use run_bash with 'pip install <pkg>' (network works).",
        "- Use read_dashboard before modifying panels you did not just create.",
        "- Panel data shapes: histogram {column, bins:[{x0,x1,count}]},",
        "  bar {categories, series:[{name,values}]}, scatter {x_label,y_label,",
        "  points:[[x,y]]}, heatmap {x_labels,y_labels,values}, line {x,series},",
        "  table {columns,rows}, stat {label,value,unit?}, markdown {text}.",
        "  Every panel: {id, type, title, commentary, data}.",
        "- A tool error is information: read the message, fix the script, retry.",
        "- When the dashboard reflects the request, STOP: reply with no tool",
        "  calls, in 1-3 sentences describing what changed, addressed to the",
        "  human in the chat.",
    ]
    return "\n".join(lines)


def run_eda_turn(slug, model_id=None):
    project = projects.get_project(slug)
    project_path = projects.project_dir(slug)
    messages = read_messages(slug)
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("no pending user message for %r" % slug)

    # EDA turns get their own run space so recon run history stays recon-only.
    runs_root = project_path / "eda" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    base = "turn-" + time.strftime("%Y%m%d-%H%M%S")
    run_id, n = base, 1
    while True:
        try:
            (runs_root / run_id).mkdir()
            break
        except FileExistsError:
            n += 1
            run_id = "%s-%d" % (base, n)
    rdir = runs_root / run_id
    writer = JournalWriter(rdir / "journal.jsonl")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        writer.write("run_failed", reason="ANTHROPIC_API_KEY is not set in the server environment")
        writer.close()
        raise ValueError("ANTHROPIC_API_KEY is not set")

    previous = os.environ.get("KC_WORKSPACE")
    os.environ["KC_WORKSPACE"] = str(project_path)
    try:
        task = _task(slug, project, messages)
        registry = Registry()
        eda_dashboard.register(registry)
        shell.register(registry)
        corpus_search.register(registry)
        final = run_with_journal(
            [],
            # 3000 tokens (the recon default) truncates mid-script here: a
            # whole pandas program rides inside one tool_use block, and a
            # truncated block reaches the loop as an empty input dict.
            AnthropicModel(api_key, task, model=model_id or DEFAULT_MODEL, max_tokens=16000),
            registry,
            writer,
            max_steps=12,
            prompt=messages[-1]["text"],
        )
        append_message(slug, "assistant", final or "(done)")
        return final
    except Exception as e:
        # The chat must show the failure where the user is looking — the
        # journal alone is not enough once the turn subprocess is gone.
        append_message(slug, "assistant", "The turn failed: %s" % str(e)[:300])
        raise
    finally:
        writer.close()
        if previous is None:
            os.environ.pop("KC_WORKSPACE", None)
        else:
            os.environ["KC_WORKSPACE"] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one EDA chat turn.")
    parser.add_argument("slug")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    eda_dir = projects.project_dir(args.slug) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    lock = eda_dir / "lock"
    try:
        lock.mkdir()
    except FileExistsError:
        print("an EDA turn is already running for %s" % args.slug)
        return 1
    # Lock claimed: the spawn-pending marker has done its job.
    (eda_dir / "spawn_pending").unlink(missing_ok=True)
    try:
        if args.bootstrap:
            if read_messages(args.slug):
                # Setup retries must not re-bootstrap over a real conversation.
                print("chat already exists; skipping bootstrap")
                return 0
            append_message(args.slug, "user", _BOOTSTRAP_MESSAGE)
        run_eda_turn(args.slug, model_id=args.model)
        return 0
    except (ValueError, RuntimeError) as e:
        print("eda turn failed: %s" % e)
        return 1
    finally:
        shutil.rmtree(lock, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
