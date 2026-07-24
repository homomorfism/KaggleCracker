# AGENTS.md — KaggleCracker

Read this before touching anything. This is a graded university assignment, not
a production repo. The grading criteria are unusual and most of your defaults
are wrong for them.

## What this project is

An agent that runs ML training experiments in a sandbox, records CV scores, and
submits to a live Kaggle competition behind human approval. Three people own
three slices of one repo. I own `tools/exec/` and `tests/` for that slice.

## The four things being graded

Every change must keep all four visible and intact:

1. Two self-designed tools with action-shaped names, descriptions that say when
   **and when NOT** to call them, and at least one constrained parameter.
2. An observe → reason → act → verify loop with explicit stopping conditions.
3. One human approval gate on an irreversible action. Reversible actions
   stay UNGATED — the contrast is the point, do not gate everything.
4. One tool error reaching the loop as its own branch, not as valid data.

## Hard rules — do not violate these, even if asked casually

- **Every tool returns the envelope.** `{"ok": True, "data": {...}}` or
  `{"ok": False, "error": {"kind": ..., "msg": ..., "retryable": bool}}`.
  Never a bare string. Never a raw exception escaping into the transcript.
- **`kind` comes from the closed list in `core/errors.py`.** Do not invent new
  error kinds. If a failure does not fit, say so and stop; do not improvise.
- **Never `except Exception: pass`.** Never `return str(e)` as if it were data.
  Never log-and-continue. A swallowed error is a direct failure of requirement 4.
- **The gate reads `ToolSpec.irreversible`.** No tool name is ever hardcoded
  into `core/gate.py`. There is a test asserting the string "submit" does not
  appear in that file. Keep it passing.
- **Only `yes`/`y` approve.** Empty input, `ok`, `sure`, `go ahead` are all
  DENIALS. Do not "improve" this by accepting more forms of consent.
- **Machine-checkable validation runs BEFORE the gate**, via `ToolSpec.precheck`.
  A human must never be asked to approve something a string comparison could
  have rejected.
- **`dry_run=True` stays the default on `submit_to_kaggle`.** Submissions to a
  live competition are capped per day and cannot be undone.
- **`core/` is frozen** and shared with two teammates. If a change genuinely
  needs to touch `core/`, STOP and tell me instead of editing it.
- **Do not add dependencies.** Standard library plus pytest. No pydantic, no
  frameworks, no `requests` unless I explicitly ask.

## Style

- Small files, plain Python, no clever abstractions. I have to explain every
  line of this out loud to a teacher who may ask how the whole system works.
- Comments explain *why a choice was made*, especially where a safer-looking
  alternative was rejected. Do not comment what the code obviously does.
- No `TODO` markers left in graded code.

## Testing — this is graded hardest

- `python -m pytest -q` must be green before you claim anything is done. Run it.
- Tests use `FakeModel` from `tests/fakemodel.py`. No API key, no network,
  fully deterministic.
- `tests/conftest.py` redirects `KC_WORKSPACE` to a tmp dir. Never write test
  artifacts into the real `workspace/`.
- Every test asserts on a **consequence**, not just a return value: after a
  denial, assert the file is unchanged and no submission was spent.
- When you add a tool, add: a schema-rejection test proving the body never ran,
  a happy path, one test per error kind, and a stopping-condition test.

## Commands

```bash
python -m pytest -q            # must be green
python break_it_on_purpose.py  # regenerates the two comparison transcripts
```

## What I want from you, and what I don't

Scaffold tools, write tests, wire plumbing, catch my mistakes. Do NOT rewrite
`core/loop.py` — I wrote those 15 lines by hand on purpose and I need to be able
to redraw them on a whiteboard.

When you finish a task, tell me in one line what you changed and what test
proves it. If you are unsure whether something violates a rule above, ask
instead of guessing.
