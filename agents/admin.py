"""The admin path: privileged commands, exactly one authorized id, all recorded.

The boundary here is the same shape as the executor/critic split. The critic has
no registry, so it cannot reach a tool no matter what it decides; the executor
holds the registry and is therefore the only side that can act. This module
holds a SECOND, PRIVILEGED registry that the ordinary agents are never handed —
run_agent is called with the ordinary registry, so an ordinary run cannot name
these tools even if the model asks for them by name. Privilege is a function of
which registry you were given, never of what you say.

Three things the ordinary path cannot do:

    quota          read the remaining daily submission quota
    dry_run on|off record whether live submissions are permitted
    stop <turn>    ask a running turn to halt

Authorization is exact equality against KC_ADMIN_USER, checked BEFORE the
command is parsed. An unset or blank KC_ADMIN_USER authorizes nobody: the
default is closed, not open. Every command, allowed or denied, is appended to
runs/hw3/admin_journal.jsonl with the caller's id, so the boundary leaves
evidence rather than being taken on trust.

The command string is DATA. It is matched against a closed list of exact forms
and never executed, split into code, or eval'd — a caller who sends
"dry_run off; submit everything" gets an unknown-command error, not a second
action. This is the HW2 data-not-instructions rule applied to a control channel.

Callers branch on the fields of the returned object:

    {"status": "ok" | "denied" | "error",
     "result": <prose for a human; never parsed>,
     "needs_approval": False,
     "envelope": <the tool's {ok,data} envelope, or None>}

needs_approval is False on every admin command, and that is deliberate: none of
these three is irreversible. dry_run is flipped back by one more command, a stop
request is advisory, and reading the quota changes nothing. The single human
gate stays where the irreversible action is — submit_to_kaggle. Gating
everything would erase the contrast the gate exists to show.
"""

import json
import os
from collections import namedtuple

from core.contracts import err, ok
from core.loop import RunState, dispatch
from core.registry import Registry, ToolSpec
from hw3_paths import hw3_dir
from tools.exec import submit
from tools.exec.experiments import _workspace_root

# The single authorized id lives in the environment, not in code, so rotating it
# is a deployment change and no id is baked into a graded file.
ADMIN_ID_ENV = "KC_ADMIN_USER"

_SUBMIT_STATE_FILE = "submit_state.json"
_STOP_FILE = "stop_requests.json"
_JOURNAL_FILE = "admin_journal.jsonl"

# dispatch() only needs .name and .args, same as the executor uses.
_Call = namedtuple("_Call", "name args")


# --- state other code reads --------------------------------------------------


def dry_run_enabled():
    """Whether submissions are still restricted to dry runs.

    True when no admin has said otherwise. The safe default survives a missing
    file, a fresh checkout, and a wiped runs/ directory — the only way to get
    False is an explicit "dry_run off" from the configured admin.

    This is a RECORDED DECISION, not an override: submit_to_kaggle still
    declares dry_run=True as its schema default and is still irreversible, so a
    live submission still stops at the human gate. Nothing here can spend a
    submission on its own.
    """
    path = hw3_dir() / _SUBMIT_STATE_FILE
    if not path.exists():
        return True
    # Written only by this module, so unparseable JSON is a real fault to raise
    # on, not something to paper over by defaulting — same stance the trigger
    # takes on its last_seen.json.
    return bool(json.loads(path.read_text()).get("dry_run", True))


def stop_requested(turn_id):
    """Has an admin asked this turn to halt?

    Shaped to be polled: run_agent's should_stop is called at the top of every
    step, so `lambda state, msgs: stop_requested(tid)` turns an out-of-band admin
    command into an ordinary stopping condition. A halt is cooperative — the
    running turn notices at its next step; nothing here kills a process.
    """
    path = hw3_dir() / _STOP_FILE
    if not path.exists():
        return False
    return turn_id in json.loads(path.read_text())


# --- the privileged tools ----------------------------------------------------


def _read_quota(args):
    ws = _workspace_root()
    # Read through submit's own loader so there is one definition of the quota
    # file's shape and one place that refuses a corrupt count. A second reader
    # here could disagree with the precheck that actually blocks submissions.
    quota, e = submit._load_quota(ws)
    if e is not None:
        return e
    today = submit._today()
    spent = quota.get(today, 0)
    if isinstance(spent, bool) or not isinstance(spent, int):
        return err("exec_failed", "quota count for %s is not an integer" % today)
    return ok(date=today, spent=spent,
              remaining=submit.DAILY_QUOTA - spent, daily_quota=submit.DAILY_QUOTA)


def _set_dry_run(args):
    enabled = args["enabled"]
    path = hw3_dir() / _SUBMIT_STATE_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dry_run": enabled}) + "\n")
    except OSError as e:
        return err("exec_failed", "could not write submit state: %s" % e)
    return ok(dry_run=enabled)


def _stop_precheck(args):
    """Machine-checkable guard before the body: a stop needs something to stop.

    A blank turn id would append an empty string to the request list and then
    match nothing forever, which looks like a working command and silently is
    not. Rejecting it here means the body never runs.
    """
    if not args["turn_id"].strip():
        return err("bad_input", "stop requires a turn id, e.g. 'stop turn-7'")
    return None


def _stop_turn(args):
    turn_id = args["turn_id"]
    path = hw3_dir() / _STOP_FILE
    try:
        requested = json.loads(path.read_text()) if path.exists() else []
        if turn_id not in requested:  # asking twice is not two stops
            requested.append(turn_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(requested) + "\n")
    except OSError as e:
        return err("exec_failed", "could not write stop requests: %s" % e)
    return ok(turn_id=turn_id, pending=requested)


READ_SUBMISSION_QUOTA = ToolSpec(
    name="read_submission_quota",
    description=(
        "Read how many of today's capped Kaggle submissions are still "
        "available. Call this to answer 'can we still submit today'. Do NOT "
        "call it to submit, to reserve a slot, or to change the count — it only "
        "reads. Admin-only: it is not in the ordinary agents' registry."
    ),
    parameters={},
    fn=_read_quota,
    # Reversible: a read. Ungated on purpose.
)


SET_SUBMIT_DRY_RUN = ToolSpec(
    name="set_submit_dry_run",
    description=(
        "Record whether submissions may go live (enabled=False) or must stay "
        "dry runs (enabled=True). Call this only when a human admin has decided "
        "to open or close live submitting. Do NOT call it to make a submission "
        "happen — it writes a flag, submits nothing, and the human gate on "
        "submit_to_kaggle still runs either way. Admin-only."
    ),
    # The one constrained parameter: a strict bool, so "off", 0 and None are all
    # rejected by validate() before the body can write an ambiguous flag.
    parameters={"enabled": {"type": "bool", "required": True}},
    fn=_set_dry_run,
    # Reversible: the opposite command undoes it in full.
)


REQUEST_TURN_STOP = ToolSpec(
    name="request_turn_stop",
    description=(
        "Ask a named running turn to halt at its next step. Call this to stop a "
        "turn that is looping or working on something no longer wanted. Do NOT "
        "call it to kill a process, undo work already done, or cancel a "
        "submission that has already been approved — it only sets a flag the "
        "turn polls. Admin-only."
    ),
    parameters={"turn_id": {"type": "str", "required": True}},
    fn=_stop_turn,
    precheck=_stop_precheck,
    # Reversible: an unheeded flag. The turn stops itself; nothing is destroyed.
)


def privileged_registry():
    """A registry holding ONLY the admin tools.

    Built fresh per call and never merged into the ordinary agents' registry.
    The absence of submit_to_kaggle here is as much the point as the presence of
    the three: the admin path cannot submit, and the ordinary path cannot flip
    the dry-run flag or read the quota. Neither side is a superset of the other.
    """
    reg = Registry()
    reg.register(READ_SUBMISSION_QUOTA)
    reg.register(SET_SUBMIT_DRY_RUN)
    reg.register(REQUEST_TURN_STOP)
    return reg


# --- the boundary ------------------------------------------------------------


def _obj(status, result, envelope=None):
    return {"status": status, "result": result,
            "needs_approval": False, "envelope": envelope}


def _journal(caller_id, command, decision, detail):
    """Append one line per command — allowed and denied alike.

    Journaling only the denials would record attacks but not authority: a reader
    could not tell whether the admin path had been used at all. Both directions
    go in, so the file answers 'who asked for what, and what happened'.
    """
    path = hw3_dir() / _JOURNAL_FILE
    record = {"caller_id": caller_id, "command": command,
              "decision": decision, "detail": detail}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _authorized(caller_id):
    admin = os.environ.get(ADMIN_ID_ENV)
    if not admin or not admin.strip():
        # Unconfigured means nobody is admin. Failing open here would make an
        # empty environment the most privileged state the system can be in.
        return False
    # Exact equality: no case folding, no stripping, no prefix match. Every
    # loosening of this comparison is a way for a near-miss id to get in.
    return caller_id == admin


def _parse(command):
    """Map a command string to (tool_name, args), or None if it is not one of
    the four exact forms. The string is data being recognised, never executed."""
    if not isinstance(command, str):
        return None
    text = command.strip()
    if text == "quota":
        return "read_submission_quota", {}
    if text == "dry_run on":
        return "set_submit_dry_run", {"enabled": True}
    if text == "dry_run off":
        return "set_submit_dry_run", {"enabled": False}
    if text == "stop" or text.startswith("stop "):
        # A bare "stop" is recognised but empty, so _stop_precheck can say
        # "stop requires a turn id" instead of the useless "unknown command".
        return "request_turn_stop", {"turn_id": text[len("stop"):].strip()}
    return None


def handle_command(caller_id, command, registry=None, output_fn=print):
    """Run one admin command on behalf of caller_id. Returns the object above.

    Order is the point, and it mirrors dispatch's: identity first, because an
    unauthorized caller must not get so much as a parse error telling them which
    commands exist; then parsing; then dispatch, which validates arguments and
    runs the precheck before any body executes.
    """
    if not _authorized(caller_id):
        reason = "caller %r is not the configured admin; command refused" % (caller_id,)
        _journal(caller_id, command, "denied", reason)
        return _obj("denied", reason)

    parsed = _parse(command)
    if parsed is None:
        reason = ("unknown admin command: %r; known commands are 'quota', "
                  "'dry_run on', 'dry_run off', 'stop <turn>'" % (command,))
        _journal(caller_id, command, "error", reason)
        return _obj("error", reason)

    name, args = parsed
    env = dispatch(_Call(name, args), registry or privileged_registry(),
                   RunState(), output_fn=output_fn)
    if env["ok"]:
        _journal(caller_id, command, "allowed", env["data"])
        return _obj("ok", "%s: %s" % (name, env["data"]), envelope=env)

    # An authorized command that failed is not a denial — the caller was the
    # admin and the boundary held; the tool itself reported an error. Keeping
    # the two apart is what makes the journal readable as a security record.
    _journal(caller_id, command, "allowed", env["error"])
    return _obj("error", "%s failed: %s" % (name, env["error"]["msg"]), envelope=env)
