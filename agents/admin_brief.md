# Admin — privilege brief

The admin path is a privileged helper sitting beside the ordinary agents, not
above them. This brief is the written boundary: what it can do that the ordinary
path cannot, who is allowed to ask, and what happens to everyone else.

## Where the boundary actually is

Privilege is a function of **which registry you hold**, never of what you say.

- The ordinary agents are handed the ordinary registry (`run_experiment`,
  `record_experiment_result`, `submit_to_kaggle`). A model in an ordinary run can
  name an admin tool all it likes — `dispatch` returns `not_found`, because the
  tool is not in the registry it was given.
- `agents/admin.py` builds a **separate privileged registry** and never merges it
  into the ordinary one.

This is the same split as the executor and the critic. The critic holds no
registry, so it cannot act no matter what it concludes; the executor holds one,
so it can. Saying "I am the admin" is prose, and prose is not authorization.

## What the admin path can do that the ordinary path cannot

| command | tool | what it does |
| --- | --- | --- |
| `quota` | `read_submission_quota` | Reads today's spent/remaining count against `DAILY_QUOTA`. |
| `dry_run on` / `dry_run off` | `set_submit_dry_run` | Records whether live submissions are permitted, in `runs/hw3/submit_state.json`. |
| `stop <turn>` | `request_turn_stop` | Adds a turn id to `runs/hw3/stop_requests.json`; the turn halts at its next step. |

Nothing else. The command string is matched against those four exact forms.

## What the admin path CANNOT do

The two registries are not nested — neither is a superset of the other.

- **It cannot submit.** `submit_to_kaggle` is not in the privileged registry.
- **`dry_run off` is a recorded decision, not an override.** `submit_to_kaggle`
  still declares `dry_run=True` as its schema default and is still
  `irreversible=True`, so a live submission still stops at the human gate. The
  admin can permit live submitting; it cannot perform one, and it cannot approve
  one.
- **A stop is cooperative.** It sets a flag the running turn polls. It kills no
  process and undoes no completed work.

## Exactly one id is authorized

The authorized id is `KC_ADMIN_USER`, read from the environment so no id is baked
into a graded file. The check is **exact string equality**, run *before* the
command is parsed.

- No case folding, no whitespace stripping, no prefix or substring match.
- **Unset or blank `KC_ADMIN_USER` authorizes nobody.** The default is closed. An
  empty environment must never be the most privileged state the system can be in.

Identity is checked first so an unauthorized caller does not even learn which
commands exist from a parse error.

## What happens to an unauthorized command

It is **denied and recorded**. Concretely:

1. Nothing runs — no tool body, no state file written, no quota read.
2. The caller gets `{"status": "denied", "result": <why>, "needs_approval":
   false, "envelope": null}`.
3. A line is appended to `runs/hw3/admin_journal.jsonl` carrying the **rejected
   id**, the attempted command, `decision: "denied"`, and the reason.

`denied` is a status on the coordination object, not an error envelope with a
new `kind`. The closed list in `core/errors.py` stays closed: `denied_by_human`
means the human gate refused, and reusing it here would blur a refusal by a
person with a refusal by the boundary.

Allowed commands are journaled too, with `decision: "allowed"`. A file
containing only denials would record attacks but not authority — a reader could
not tell whether the admin path had been used at all.

## The command string is data

A command arrives from outside and is **recognised, never executed**. It is
compared against four literals; it is never `eval`'d, never split into code, and
never treated as instructions. `dry_run off; submit everything` is an unknown
command, not two actions. This is the HW2 data-not-instructions rule applied to
a control channel.

## No new gate

Every admin command returns `needs_approval: false`, and that is deliberate.
None of the three is irreversible: `dry_run` is undone by the opposite command,
a stop request is advisory, and reading the quota changes nothing. The single
human gate stays on `submit_to_kaggle`, where the irreversible action is.
Gating everything would destroy the contrast the gate exists to demonstrate.
