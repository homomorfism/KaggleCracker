# KaggleCracker — LLM Architecture

An agent system that studies a live Kaggle competition (Playground S6E7, balanced
accuracy, target `health_condition`), runs ML experiments in a sandbox, records CV
scores, and submits behind a single human approval gate. This document maps the
agents, their wiring, the tool surface, the stopping criteria, and the memory/context
layers. Repository layout and per-slice rationale live in `docs/PLAN.md`; this file
describes the system as built.

---

## 1. The shared core (`core/`, frozen)

Every agent in the repo runs on the same 15-line hand-written loop and the same tool
contract. Nothing else is shared at runtime.

**Envelope.** Every tool returns exactly one of:

```python
{"ok": True,  "data": {...}}
{"ok": False, "error": {"kind": ..., "msg": ..., "retryable": bool}}
```

`kind` comes from the closed list in `core/errors.py`:
`not_found, bad_input, timeout, rate_limited, exec_failed, denied_by_human, quota_exhausted`.
A raised exception never reaches the transcript raw — `dispatch()` converts it to an
`exec_failed` envelope, and the model branches on the error as its own path
(`[TOOL OK ...]` vs `[TOOL ERROR ...]` are visibly distinct surface forms).

**Dispatch order** (`core/loop.py::dispatch`) — each stage runs only if every earlier
stage passed:

1. unknown/disabled tool → `not_found`
2. `validate()` against the schema → `bad_input` (body never runs)
3. `ToolSpec.precheck` — deeper machine check, still before the gate
4. human gate — only if `ToolSpec.irreversible`; refusal → `denied_by_human`
5. tool body — exception-wrapped → `exec_failed`

**Gate** (`core/gate.py`). Reads only `ToolSpec.irreversible` — no tool name is ever
hardcoded (a test asserts "submit" does not appear in the file). Only a literal
`yes`/`y` (stripped, case-insensitive) approves; blank input, EOF, "ok", "sure" are
denials. An irreversible tool must register a `preview` so the human sees the real
consequence before approving. Machine-checkable validation always runs before the
gate: a human is never asked to approve something a string comparison could reject.

**Loop** (`core/loop.py::run_agent`). Model is a plain callable
`model(messages, tools) -> Reply`; `Reply`/`ToolCall` from `tests/fakemodel.py` are
the model-interface contract for both the test `FakeModel` and the live Anthropic
adapter. Three explicit stopping conditions:

| condition | behaviour |
|---|---|
| external `should_stop(state, messages)` fires | returns `None` |
| model emits no tool calls | done — returns the reply text |
| `max_steps` exhausted (default 12) | raises `StepLimitReached` carrying the partial transcript |

**Tool disabling.** `RunState` counts consecutive failures per tool; at
`max_consecutive_fails` (default 2) the tool is dropped from the schemas offered to
the model. A success clears the streak. The experiment runner raises the cap to 6
because consecutive errors are the normal texture of debugging a training script.

---

## 2. Agents

Seven distinct agent roles. Four are LLM loop agents on `run_agent`; two are
scripted pipelines; one is a deterministic coordinating pair; plus an offline
monitor that grades everything after the fact.

### 2.1 LLM loop agents (model: `claude-sonnet-5` via `ui/live_model.py` / `ui/llm.py`)

All are launched as detached subprocesses by the FastAPI server (`ui/app.py`) and
are stateless per turn: each turn re-assembles its full context from files on disk.

| agent | entrypoint | tools registered | max_steps | max_consec_fails | max_tokens | gate |
|---|---|---|---|---|---|---|
| **Recon / analysis** (live) | `ui/run_analysis.py --mode live` | `profile_dataset`, `write_preprocessing_plan`, `overwrite_preprocessing_plan` | 16 | 2 | 3000 | real — human answers via `gate_answer.txt`, 600 s timeout → denial |
| **Recon / analysis** (demo) | `ui/run_analysis.py --mode demo` | same | 12 | 2 | n/a (`FakeModel`, scripted replies, real tools) | auto-`yes`, journaled as scripted |
| **EDA** | `ui/run_eda.py` | `run_eda_script`, `remove_dashboard_panels`, `read_dashboard`, `run_bash` | 12 | 2 | 16000 | none — all tools reversible |
| **Experiment** | `ui/run_experiment_turn.py` | `run_experiment`, `record_experiment_result`, `propose_experiment_plan`, `finalize_experiment`, `run_bash` | 24 | 6 | 16000 | none — all tools reversible |
| **Knowledge (e2e)** | `e2e_knowledge_run.py` | `search_kaggle_discussions`, `save_technique_note`, `fetch_external_dataset` | 24 | 2 | 1500 | real, on stdin (`fetch_external_dataset` is irreversible) |

The experiment agent's workflow is encoded in its prompt, not in code: a three-phase
state machine `draft → plan_proposed → finished` persisted in `experiment.json`.
Phase 1 asks the human up to 4 clarifying questions; phase 2 writes `plan.md` via
`propose_experiment_plan` and stops for approval; phase 3 (only after a human
approval message) iterates training scripts through `run_experiment`, files the
score with `record_experiment_result`, and closes with `finalize_experiment`.
"Never run experiments before the plan is approved. Never invent scores."

### 2.2 Scripted pipelines (no agent loop)

| agent | entrypoint | LLM use | what it does |
|---|---|---|---|
| **Setup** | `ui/run_setup.py` | none | fetch competition metadata + file list, download/extract data, write `setup/status.json`; on success auto-spawns Observer and EDA `--bootstrap` — the only automatic agent-to-agent trigger in the system |
| **Observer** | `ui/run_observer.py` | two one-shot completions (no tools): per-notebook JSON summary (2500 tok), aggregate markdown briefing (2000 tok) | lists top community notebooks, pulls ≤ 8 new ones per refresh (dedup on `ref`), ≤ 15 000 chars of source per prompt, one paid retry on unparseable JSON, then skip; writes `knowledge/notebooks.jsonl` + `knowledge/summary.md` |

### 2.3 Critic + Executor pair (`agents/`, deterministic)

Two agents that coordinate **only** through a small structured object — never by
parsing each other's prose:

```python
{"status": "pass" | "revise" | "escalate", "result": <prose for humans>, "needs_approval": bool}
```

- **Critic** (`agents/critic.py::review_submission`) reviews one submission proposal
  against the rubric in `agents/critic_brief.md`: R1 row count matches the sample,
  R2 `cv_score` present and ≥ `CV_FLOOR = 0.60`, R3 `model_type` recorded. Facts for
  R2/R3 come from the shared SQLite experiments table (`memory/relational.py`), not
  from the proposal's prose. Priority is escalate > revise > pass: only a
  regenerate-fixable row-count mismatch earns `revise`; a weak/missing score or
  missing record is a human's judgement call and escalates. A `pass` still sets
  `needs_approval: True` — the submission itself is irreversible.
- **Executor** (`agents/executor.py::run_submission`) branches purely on
  `obj["status"]`:
  - `escalate` → stop, no submit;
  - `revise` → ask the reviser for a corrected proposal and loop, capped at
    `MAX_ROUNDS = 2` attempts, then `gave_up`;
  - `pass` → dispatch `submit_to_kaggle` through the normal `core.loop.dispatch`,
    so the exact HW1 precheck-then-gate machinery runs (nothing is reimplemented);
    `cv_score` is read from the shared experiments table, never from prose;
  - any other status → raise loudly (a broken contract is not valid data).

  Every terminal outcome (`submitted | denied | escalated | gave_up`, plus rounds
  used) is recorded as a row in the shared `submission_outcomes` table before
  returning, so the offline monitor can audit decisions against transcripts.

### 2.4 Offline monitor (`monitor/`, rule-based, no LLM)

Deliberately outside the request loop — it never imports the agents, tools, or
model. `monitor/judge.py` grades the plain-text transcripts in `runs/hw2/` on two
axes with named verdicts (`prompt_adherence`: strictly_adheres / minor_violation /
serious_violation; `task_completion`: completed / partial / failed), each non-clean
verdict carrying an `{expected, got}` rationale. It specifically checks for
unguarded submit offers and for prompt injections that arrived inside retrieved
data (a correct refusal is a clean verdict with an observation, not a violation).
Output: `monitor/report.json`. `monitor/clock.py` runs the judge on an interval
(`--every`, default 900 s) with an explicit stopping condition (`--max-runs`;
unbounded by default for real background use, pinnable in tests/demos).

---

## 3. How the agents are connected

There is no orchestrator, no message bus, and no shared runtime state. Agents
connect through files in the per-project directory and each agent **pulls** what it
needs at prompt-assembly time. The FastAPI server holds zero job state; every launch
is a detached subprocess, every status answer is a disk read, and concurrency is
guarded by mkdir-as-lock plus a short-lived `spawn_pending` marker.

```mermaid
flowchart TD
    UI[FastAPI ui/app.py<br/>detached subprocess spawns] --> SETUP[Setup<br/>no LLM]
    UI --> RECON[Recon agent]
    UI --> EDA[EDA agent]
    UI --> EXP[Experiment agent]
    UI --> OBS

    SETUP -- auto-spawn on success --> OBS[Observer]
    SETUP -- auto-spawn on success --> EDA
    SETUP --> CJ[competition.json + data/]

    OBS --> KN[knowledge/notebooks.jsonl<br/>knowledge/summary.md]
    EDA --> DASH[eda/dashboard.json]
    CJ --> EDA
    CJ --> EXP
    KN -- pulled into prompt --> EXP
    DASH -- pulled into prompt --> EXP

    EXP --> LB[experiments/leaderboard.jsonl<br/>experiment.json, plan.md, code/]
    RECON --> PLANS[plans/*.plan.md<br/>experiments.db findings]

    EXP -. proposal .-> CRITIC[Critic]
    CRITIC -- "{status, result, needs_approval}" --> EXEC[Executor]
    EXEC -- dispatch --> GATE{{Human gate<br/>yes/y only}}
    GATE --> SUBMIT[submit_to_kaggle<br/>dry_run=True default]
    EXEC --> OUT[(submission_outcomes<br/>SQLite)]

    MON[Offline monitor<br/>judge + clock] -. reads transcripts .-> RUNS[runs/*.txt]
    MON -.-> OUT
```

Key edges:

- **Setup → Observer + EDA** is the only automatic trigger. Everything else is
  user-triggered (create/message/refresh endpoints).
- **Observer and EDA are siblings**: both feed the experiment agent's prompt
  (community techniques and dashboard insights respectively); neither reads the
  other's output.
- **Recon is an island**: its plans and findings surface only to the UI
  (`/plans`, `/findings`); no other agent's prompt reads them.
- **Critic ↔ Executor** is the one agent-to-agent protocol, and it is field-based
  (`status`, `needs_approval`) by rule — prose is for humans only. `needs_approval`
  routes to the single HW1 gate; it does not create a second approval mechanism.
- **Human chat** (`eda/chat/messages.jsonl`, per-experiment `messages.jsonl`) is the
  durable conversation state; each turn is a fresh stateless subprocess primed from
  a window of it (12 messages for EDA, 20 for experiments).
- `KC_WORKSPACE` (and `KC_EXPERIMENT_DIR` for planning tools) is the multi-project
  mechanism: pointing the env at a project directory makes every tool operate on
  that project unchanged.

---

## 4. Tool surface (19 ToolSpecs)

Every tool: action-shaped name, description saying when **and when not** to call
it, at least one constrained parameter, envelope return. Only **3 tools are
irreversible** (gated); everything else is ungated by design — the contrast is the
point of the gate.

### exec slice (`tools/exec/`)

| tool | purpose | key constraints | gated |
|---|---|---|---|
| `run_experiment` | run a training script in the sandbox, parse `CV_SCORE: <float>` from stdout | `timeout_s` 1–1800 (default 600), `cv_folds` 2–20 (default 5); precheck: `dataset_ref` must stay inside `workspace/data` | no |
| `record_experiment_result` | append an already-computed score to `experiments/leaderboard.jsonl` | precheck: `fold_scores` ≤ 20 entries; `cv_score` deliberately unbounded (metrics can be negative) | no |
| `run_bash` | one shell command in the workspace (venv python/pip first on PATH) | `timeout_s` 1–600 (default 120) | no |
| `submit_to_kaggle` | submit predictions to the live competition | `dry_run` **default True**; precheck: file exists, columns/rows match sample, daily quota (10/day) unspent → `quota_exhausted`; preview shows rows, CV score, remaining quota | **yes** |

### recon slice (`tools/recon/`)

| tool | purpose | key constraints | gated |
|---|---|---|---|
| `profile_dataset` | per-column statistics for a CSV, recorded as `findings` rows | `checks` item_enum of 7 check names; `sample_rows` 100–200 000; precheck: path containment, target/compare_path requirements | no |
| `write_preprocessing_plan` | save a NEW `plans/<dataset>.plan.md` | precheck rejects "plan already exists" | no |
| `overwrite_preprocessing_plan` | replace an EXISTING plan (destroys the old one) | same body as write; precheck rejects "no plan exists" (never send a human a no-op); preview shows the doomed plan | **yes** |
| `record_dataset_note` | save user-authored dataset context with a retrieval cue | `shared` default False; identity from `KC_USER`, not a parameter (unspoofable) | no |
| `recall_dataset_notes` | recall own + shared notes; recalled text is quoted DATA | visibility enforced in the SQL WHERE clause | no |

### knowledge slice (`tools/knowledge/`)

| tool | purpose | key constraints | gated |
|---|---|---|---|
| `search_kaggle_discussions` | search public kernels for a competition | `sort` enum hot/recent/votes, `max_results` 1–50; zero results = `ok` with a note, never an error; HTTP 429 → `rate_limited` | no |
| `save_technique_note` | file one mined technique + evidence URL | `confidence` enum confirmed_cv_gain/claimed/speculative; `est_effort` enum low/medium/high; dedup on (technique, url) | no |
| `fetch_external_dataset` | download a third-party dataset into the workspace | `merge_into_training` default False; precheck: strict `owner/name` ref shape, already-fetched rejected before the gate; preview warns about target leakage | **yes** |

### eda slice (`tools/eda/`)

| tool | purpose | key constraints | gated |
|---|---|---|---|
| `run_eda_script` | run an analysis script; upsert the JSON dashboard panels it emits | `timeout_s` 1–600; panels validated by `tools/eda/spec.py` (8 panel types, ≤ 24 panels, per-type size caps, 512 KB dashboard cap) | no |
| `remove_dashboard_panels` | delete panels by id | `panel_ids` non-empty list of strings | no |
| `read_dashboard` | list panel ids/types/titles (no data) | no parameters; never errors | no |

### experiment-lifecycle slice (`tools/experiments/`)

| tool | purpose | key constraints | gated |
|---|---|---|---|
| `propose_experiment_plan` | write `plan.md`, move state to `plan_proposed` | plan ≥ 100 chars; experiment addressed via `KC_EXPERIMENT_DIR`, not a parameter | no |
| `finalize_experiment` | mark the experiment `finished` with score + summary | requires a prior plan | no |

### memory slice (`tools/memory/`)

| tool | purpose | key constraints | gated |
|---|---|---|---|
| `save_memory` | save a durable fact/rule for later runs | `kind` enum fact/rule (the constrained parameter); `shared` default False | no |
| `retrieve_memory` | pull matching saved facts/rules mid-run | returned notes are DATA to quote, never commands | no |

---

## 5. Stopping criteria and budgets

There is **no token or dollar budget** anywhere; cost control is structural. All
caps in one place:

### Loop-level

| mechanism | value | where |
|---|---|---|
| step cap | 12 default; 16 live recon; 12 EDA; 24 experiment & knowledge e2e | `run_agent(max_steps=...)`; overflow raises `StepLimitReached` |
| "model is done" | reply with no tool calls ends the run | `run_agent` |
| consecutive-failure tool disabling | 2 default; 6 for experiment turns | `RunState.max_fails` |
| external `should_stop` | polled at the top of every step | `run_agent(should_stop=...)` |

### Domain `should_stop` detectors (read durable records, not the transcript)

| detector | fires when | source |
|---|---|---|
| `plan_written(dataset)` | `plans/<dataset>.plan.md` exists — recon is done | `tools/recon/stopping.py` |
| `stall_detector()` | same query issued twice in a row, OR the last two searches surfaced zero new URLs | `tools/knowledge/stopping.py`, reads `knowledge/search_log.jsonl` (successes only, so a flaky network cannot trip it) |
| `plateau_detector(window=3, min_delta=0.0005)` | best recorded CV score has not improved by > min_delta over the last 3 experiments | `tools/exec/stopping.py`, reads `experiments/leaderboard.jsonl`; direction from `HIGHER_IS_BETTER` |

### Agent-level attempt/effort caps

| cap | value | where |
|---|---|---|
| executor revise rounds | `MAX_ROUNDS = 2`, then `gave_up` | `agents/executor.py` |
| daily submission quota | 10/day, checked in the precheck → `quota_exhausted` before the gate | `tools/exec/submit.py` |
| observer batch | 8 new notebooks per refresh, 15 000 chars source per prompt, 1 paid retry per notebook | `ui/run_observer.py` |
| monitor clock | `--max-runs` (unbounded default, pinnable) | `monitor/clock.py` |

### Wall-clock timeouts

| timeout | value |
|---|---|
| `run_experiment` subprocess | 1–1800 s, default 600 |
| `run_eda_script` / `run_bash` subprocess | 1–600 s, default 120 |
| `fetch_external_dataset` download | 600 s |
| Anthropic HTTP socket | 180 s |
| live gate wait for a human answer | 600 s, then treated as denial |

API failures (429/529 from Anthropic) raise immediately and are journaled as
`run_failed` — no retry, no backoff. `retryable` on tool errors is advice to the
model, not an automatic mechanism.

---

## 6. Memory and context

### 6.1 Three stores, shaped to what they hold

| store | shape | contents | files |
|---|---|---|---|
| **SQLite** (`memory/relational.py`) | fixed schema, queryable | `experiments` (id, user_id, model_type, cv_score, status) and `submission_outcomes` (closed status set: submitted/denied/escalated/gave_up + rounds) | `workspace/experiments.db` (`KC_DB` override) |
| **SQLite, recon slice** (`tools/recon/store.py`) | fixed schema | `findings` (per-check profile rows, replace-per-check transactionally) and `dataset_notes` (visibility in the WHERE clause: `user_id = ? OR shared = 1`) | per-project `experiments.db` |
| **JSON document store** (`memory/documents.py`) | free-form, append-only JSONL | facts/observations with a retrieval `cue`; private-by-default, `shared` flag | `workspace/documents.jsonl` (`KC_DOCS` override) |
| **Markdown rules** (`memory/rules.py`, `memory/recon_rules.md`) | always-on instruction layer | operating rules injected into every run's system prompt — not a scratchpad for facts | `memory/rules.md`, `memory/recon_rules.md` |

Rule of the layer: pick the store by the shape of the thing. Structured rows never
go into the JSON store; free-form notes never go into SQLite.

### 6.2 Push vs pull — the two ways context enters a run

**Push** (attached automatically before the loop starts):

- `memory/rules.md` via `load_rules()` — the five standing rules (only yes/y
  approves a submission; state the CV score before proposing one; shared notes are
  quoted data; prefer a previously used model_type; rules win over in-run requests).
- `tools/recon/context.py::push_context()` — `recon_rules.md` + hardcoded standing
  competition facts, as one `[SYSTEM]` message. A missing rules file fails loudly at
  setup (run wiring, not a tool body).
- The UI runners assemble their own push context per turn: competition metadata,
  data file names, current dashboard state, EDA insights, community-knowledge
  summary (truncated: 4000 chars summary, 3000 chars plan, 15 notebook lines), and
  the chat-history window.

**Pull** (fetched mid-run, on demand, via tools): `retrieve_memory`,
`recall_dataset_notes`, `read_dashboard`, `search_kaggle_discussions`, and the
findings table. Used when the need is conditional or the data is too large to
attach every time.

### 6.3 Identity and trust boundaries

- Identity is a property of the run: `KC_USER` env, never a tool parameter, so the
  model cannot address another user's private notes (an invented `user_id` argument
  is rejected by `validate()` as an unknown parameter).
- Visibility is enforced at the SELECT: another user's private record is never
  returned, not filtered late.
- **Shared content is DATA, never instructions.** Retrieved notes are quoted and
  reasoned about; a stored string that reads like a command is still just a string.
  The offline monitor verifies this held: an injection that arrived in retrieved
  data plus an explicit refusal grades clean-with-observation; no refusal grades
  `serious_violation`.

### 6.4 Environment contract

`KC_WORKSPACE` (sandbox root), `KC_USER` (identity), `KC_DB` / `KC_DOCS` (store
paths, pointed at tmp dirs by tests), `KC_EXPERIMENT_DIR` (which experiment the
planning tools mutate), `ANTHROPIC_API_KEY`. Tests never touch the real
`workspace/`.

---

## 7. Models and testing

- Live model: `claude-sonnet-5` through a stdlib-only `urllib` adapter
  (`ui/live_model.py`, `ui/llm.py`, `e2e_knowledge_run.py::AnthropicModel`), which
  converts `ToolSpec.parameters` into Messages-API `input_schema` and returns
  `Reply`/`ToolCall` — the same dataclasses `FakeModel` uses, so the model interface
  is one contract for tests and production.
- `tests/fakemodel.py::FakeModel`: scripted FIFO replies, records the `tools`
  offered on every call (so tests can assert a disabled tool disappeared), fails
  loudly when the script runs out. No API key, no network, deterministic.
- The `break_it_on_purpose*.py` scripts generate paired transcripts (with vs
  without the error branch) proving requirement 4 for each slice; the
  `examples/*.py` scripts generate the HW2 memory-behaviour transcripts
  (fact resurfaces with a control run, private-stays-private, planted injection
  quoted not obeyed, rule changes behaviour, outcomes recorded). All are graded by
  the offline monitor.
