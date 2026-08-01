# KaggleCracker — Implementation Plan (7-day PoC)

Derived from `~/.gstack/projects/homomorfism-KaggleCracker/shamilarslanov-main-design-20260731-190129.md`
Reviewed by `/plan-eng-review` on 2026-07-31. Branch: `main`. Repo: `KaggleCracker2` (greenfield).

Legacy `../KaggleCracker` is **not** a source. Nothing is ported. See "What already exists".

---

## Deltas from the design doc

Ten decisions were taken during review. Everything else in the design doc stands as written.

| # | Decision | Effect |
|---|---|---|
| D1 | One `AgentRuntime` (plain LLM completion) for engine **and** context feeders. smolagents moved to a Day-6 swap test. | Removes a second LLM integration path from Day 4 |
| D2 | Full test suite + `FakeExecutor`/`FakeRuntime` harness | Success criterion 4 becomes a passing test, not an interface review |
| 1 | Discussion ingestion via official `kaggle forums` / `kaggle competitions topics` CLI | Deletes design Open Question 3; kills the scraper dependency and the manual-paste step |
| 2 | `SubmissionValidator` at node completion **and** at approve | Protects the ~5/day submission quota and criterion 1 |
| 3 | Leaderboard surfaces fold mean, std, min | Makes CV overfit visible to the human at approve time |
| 4 | Hard `max_nodes` / `max_wall_clock_s` / `max_usd` caps + resumable worker | Overnight run survives a crash and cannot run away on cost |
| 5 | `protocols` table + `protocol_id` on every node | Makes the design's own staleness rule implementable |
| 6 | Composable prompt blocks, single output-contract constant | Prompt and validator cannot drift apart |
| 7 | Snapshot tests always + opt-in live prompt eval (`pytest -m eval`) | Measures contract-violation rate per action |
| 9 | **No scheduler.** Two processes (`worker`, `web`), SQLite WAL, advisory lock | One experiment at a time is structural, not a config value |
| 10 | Experiments chatbot **cut to v2** | Protects Day 5 (tree view) and Day 6 (submitter) |
| 11 | Web UI keeps its full Day 5, **unconditionally** | Outside voice argued for cutting it; rejected — the tree view is why Approach C was chosen over A |
| 12 | Five outside-voice hardening items adopted (below) | Closes every hole that can silently waste a night, a submission, or the budget |

### Adopted from the outside voice (Codex)

1. **Kaggle auth + rules preflight.** Credentials present, `kaggle competitions list` succeeds (proves
   auth **and** network), target competition actually downloadable (proves **competition rules accepted
   in a browser** — a hard Day-0 blocker with no code fix), `kaggle-cli >= 2.2.0`.
2. **Approve/submit idempotency.** `UNIQUE(node_id)` on `submissions` plus a disabled-on-submit button.
   The validator stops a *bad* submission; nothing stopped the *same good* one going twice.
3. **Stale worker-lock recovery.** Heartbeat TTL: a lock older than 3 missed heartbeats is stealable
   and the steal is logged loudly. Without this a crashed worker bricks the run.
4. **Circuit breaker.** Stop the run after 5 consecutive non-ok nodes or 3 consecutive `infra_error`s.
   Budget caps alone let 20 identical failures burn the whole budget and report a completed run.
5. **Context size budget.** Context blocks truncated to a configured char budget, oldest discussion
   summaries dropped first, assembled prompt size logged per node. This is the real cost curve — the
   Day-2 per-node measurement otherwise underestimates the Day-4 bill.

### Corrected prior learning

`[kaggle-api-no-discussions-endpoint]` (was 8/10, cross-model) is **wrong as of kaggle-cli v2.2.0**.
Verified in the changelog: `feat: add forums commands for browsing Kaggle discussions (#993)` and
`Add competitions topics CLI command (#982)`. Commands: `kaggle competitions topics <comp>`,
`kaggle forums topics list <forum>`, `kaggle forums topics show <ref>`, all with `--csv`.

**Correction to that correction (verified against the installed CLI on Day 0):** an earlier draft of
this plan claimed `kaggle competitions submit --wait --poll-interval` removes the need for a poll
loop. It does not — those flags **do not exist** in kaggle-cli 2.2.4. `submit` takes only
`-f/-k/-m/-v/-q/--sandbox`. Day 6 **does** need a poll loop: submit, then poll
`kaggle competitions submissions <comp> --csv` until the row leaves `SubmissionStatus.PENDING`.
Two other doc-vs-binary gaps found the same way: `competitions download` has no `--unzip` in 2.x
(unzip yourself), and `competitions files` is the cheapest proof that competition rules were accepted.

The forums claim held up: `kaggle competitions topics list|show` is real and works.

## Day-0 status (2026-07-31) — COMPLETE

| Item | Result |
|---|---|
| Competition | **`spaceship-titanic`** — rolling deadline, never closes. Chosen over `playground-series-s6e7`, which expired the same day. |
| Data | 8,693 train x 14, 4,277 test, target `Transported` (bool), balance 50.4/49.6. Nulls in every feature column (~2%). |
| Metric | Classification accuracy, higher is better. |
| CV protocol | `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, grouping on the `PassengerId` prefix. |
| Naive baseline (sample_submission, all False) | **0.4964** |
| Day-0 baseline (CryoSleep rule) — public LB | **0.72550** |
| Submission quota | 10/day for this competition (9 remaining after the boundary walk). |

**The protocol was settled on Day 0, not deferred to the Day-4 EDA agent.** `PassengerId` is
`gggg_pp`, where `gggg` is the travelling party. **Zero of 3,063 test groups appear in train** —
Kaggle split by group. Within a party the outcome is correlated: P(transported | all groupmates
transported) = 0.616 against a 0.504 base rate. A plain `KFold` therefore lets a model read a
groupmate's label that does not exist at test time, inflating CV without moving the leaderboard.

**The Day-0 baseline does not test that call, and it would be wrong to claim it does.** The
CryoSleep rule reads one column; it has no mechanism to memorize group identity, so grouping the
holdout cannot change its score. Measured three ways on the same rule: group-disjoint holdout
**0.7277**, random-row holdout **0.7066**, full train **0.7183**, public LB **0.72550**. At
n≈1,750 the standard error is about 0.011, so that whole spread is sampling noise. The 0.002 gap
between the group-disjoint holdout and the leaderboard is luck, not evidence.

The justification for `StratifiedGroupKFold` is structural, not empirical: the train/test split is
group-disjoint and within-party outcomes correlate, so any model that *can* exploit group membership
will be scored optimistically by a row-random fold. It also costs nothing when the model can't
exploit it. The protocol will be genuinely tested on Day 2, by the first gradient-boosted node with
group-derived features — that is the point to re-measure the CV-to-LB gap and believe it.

---

## Process architecture

Two processes. No queue, no broker, no scheduler. "One experiment at a time" is a consequence of
there being exactly one worker, enforced by an advisory lock.

```
┌─────────────────────────┐         ┌─────────────────────────┐
│  worker  (sole writer)  │         │  web  (readers only)    │
│                         │         │                         │
│  SearchLoop             │         │  FastAPI + HTMX + Jinja │
│   ├ budget caps         │         │   ├ tree (mermaid)      │
│   ├ resume(run_id)      │         │   ├ leaderboard         │
│   └ SearchPolicy        │         │   ├ node detail         │
│        │                │         │   ├ context panel       │
│        ▼                │         │   └ approve ──┐         │
│  AgentRuntime.generate  │         │               │         │
│        │                │         └───────────────┼─────────┘
│        ▼                │                         │
│  Executor.run           │                         ▼
│   └ LocalDockerExecutor │                   SubmissionValidator
│        │                │                         │
│        ▼                │                         ▼
│  validate + persist     │                   kaggle submit --wait
└───────────┬─────────────┘                         │
            │                                       │
            ▼                                       ▼
      ┌───────────────────────────────────────────────────┐
      │  SQLite (WAL mode)                                │
      │  protocols · runs · nodes · submissions           │
      │  worker_lock (pid + heartbeat)                    │
      └───────────────────────────────────────────────────┘
```

WAL mode is load-bearing: without it the web process's reads block the worker's writes and you get
`database is locked` mid-run. The advisory lock makes a second `worker` start fail loudly rather than
silently doubling CPU load and forking the tree.

## Node lifecycle

```
                    SearchPolicy picks action
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           draft          improve           debug
              └───────────────┼───────────────┘
                              ▼
                    PromptBuilder.build(action, parent, tree, context)
                              │
                              ▼
                    AgentRuntime.generate  ──► API error/429 ──► retry×3 ──► node NOT created
                              │
                              ▼
                    ast.parse(code)  ──fail──► status=contract_violation
                              │                (error text = SyntaxError)
                              ▼
                    LocalDockerExecutor.run(code, timeout_s)
                              │
        ┌─────────────┬───────┴────────┬──────────────────┐
        ▼             ▼                ▼                  ▼
    exit 0        exit != 0      SIGKILL @ timeout    docker error
        │             │                │                  │
        ▼             ▼                ▼                  ▼
  parse metrics   status=error    status=timeout    status=infra_error
        │         (traceback)     (NO traceback —   (not the agent's
        ▼                          own prompt)       fault, retry)
  validate_submission
        │
   ┌────┴────┐
   ▼         ▼
  ok    mismatch ──► status=contract_violation (error = the diff)
   │
   ▼
status=ok, cv_score + fold_scores + protocol_id persisted
```

The four terminal statuses are why `RunResult` needs a status enum rather than a bare error string:
a 900s SIGKILL produces **no traceback**, so a debug node built from `""` regenerates garbage.
Each status gets its own debug prompt template.

## Prompt composition (single contract source)

```
build(action) = [ task_spec        ]  shared
                [ cv_protocol      ]  shared — rendered from the current protocols row
                [ OUTPUT_CONTRACT  ]  shared — ONE module constant
                [ context_docs     ]  shared — eda_findings.md + discussions/*.md
                [ tree_summary     ]  shared — what has been tried, scores, direction
                [ action_block     ]  ── draft:   "models already tried: [...], pick a different one"
                                       ── improve: parent code + parent score + what to change
                                       ── debug:   parent code + status-specific failure block
                                                     error   → traceback
                                                     timeout → "killed at Ns, no traceback, make it faster"
                                                     contract→ the validator diff
```

`OUTPUT_CONTRACT` is imported by `PromptBuilder`, `parse_metrics`, and `SubmissionValidator`.
One constant, three consumers — prompt and validator cannot drift.

## Data model

```sql
protocols(id, metric_name, direction, fold_scheme, fold_params_json, seed, created_at, is_current)
runs(id, protocol_id, max_nodes, max_wall_clock_s, max_usd,
     max_consecutive_failures, max_consecutive_infra_errors,
     nodes_done, wall_clock_s, usd_spent, consecutive_failures, consecutive_infra_errors,
     state, stop_reason, started_at, ended_at)
nodes(id, run_id, protocol_id, parent_id, action, depth, code,
      status, cv_score, fold_scores_json, metrics_json,
      stdout_path, run_flags_json, prompt_chars, tokens_in, tokens_out, usd, created_at)
submissions(id, node_id UNIQUE, kaggle_ref, message, public_score, state, submitted_at)
worker_lock(id=1, pid, hostname, heartbeat_at)   -- stealable after 3 missed heartbeats
```

`protocol_id` on every node is what makes the design's staleness rule real: changing the fold scheme
inserts a new `protocols` row and flips `is_current`. Nothing is mutated or deleted; old nodes simply
stop matching the ranking filter.

**All node ranking goes through one helper.** `ranking.best_node()` and `ranking.leaderboard()` are the
only places that sort nodes. Both filter on `is_current` and `status='ok'` and honor `direction`.
Ad-hoc ORDER BY elsewhere is a bug — it is how the protocol filter gets forgotten.

`run_flags_json` per node is the evidence for success criterion 5 (no container ever had network).

---

## Agent memory

There is **no memory subsystem** — no conversation history, no vector store, no episodic recall.
Every `AgentRuntime.generate()` call is stateless. That is deliberate: in AIDE-style search the
**tree is the memory structure**, and bolting a second memory layer beside it duplicates the thing
that makes the approach work.

What each prompt actually carries, and which memory role it plays:

```
                    ┌──────────────────────────────────────────────┐
  static            │ task_spec · cv_protocol · OUTPUT_CONTRACT     │  no memory,
                    │                                               │  pure instruction
                    ├──────────────────────────────────────────────┤
  semantic /        │ context_docs                                  │  written once by the
  long-term         │   eda_findings.md · discussions/<date>.md      │  feeders, read every call
                    ├──────────────────────────────────────────────┤
  episodic          │ tree_summary                                  │  rebuilt from SQLite on
                    │   what was tried · scores · what failed        │  every call, never held
                    ├──────────────────────────────────────────────┤
  working           │ parent code · parent score OR parent failure   │  one hop only
                    └──────────────────────────────────────────────┘
```

Two gaps this leaves, both fixed rather than papered over with a memory framework:

1. **Cross-run amnesia.** `nodes.run_id` scopes a node to a run. If `tree_summary` filters on
   `run_id`, then restarting the worker after a crash — or starting run 2 the next evening — makes
   the agent redraft models it already tried last night. **Fix:** `tree_summary` spans *all* nodes on
   the **current protocol**, not the current run. `runs` stays the unit of budget accounting;
   `protocols` is the unit of comparability and therefore of memory. One `WHERE` clause, and it is
   already the same filter `ranking.py` uses.
2. **No failure digest.** If nodes 5, 9, and 11 all die the same way (OOM, same missing column, same
   timeout), node 17 learns nothing unless it happens to be a debug node on one of them. **Fix:** a
   `known_failures` block in `tree_summary` — failure signatures grouped by `RunStatus` and the first
   line of the error, with counts. Directly serves the debug action and costs a GROUP BY.

Explicitly **not** doing: vector store over past nodes, summarized long-term memory, cross-competition
recall. One competition, 20-40 nodes, full history fits in the context budget. Revisit only if
`prompt_chars` shows the tree summary crowding out the rest.

## LLM providers

Three supported: **OpenAI, DeepSeek, Anthropic.** Provider is a config value, not a code path.
No LangChain — `AgentRuntime.generate` is one completion returning a string, and `langchain-core` +
`langchain-openai` for that is the same overbuild that got smolagents demoted. Provider SDKs directly.

**Two adapters, not three.** DeepSeek's API is deliberately OpenAI-shaped, so it and OpenAI differ
only by base URL, model name, and price. Anthropic is genuinely different and gets its own.

```
                        AgentRuntime (protocol)
                 generate(prompt, action) -> GenerateResult
                                 │
                 ┌───────────────┴────────────────┐
                 ▼                                ▼
    OpenAICompatibleRuntime              AnthropicRuntime
      openai SDK, base_url                 anthropic SDK
      ├─ openai    (default base)          ├─ system prompt outside messages
      └─ deepseek  (api.deepseek.com)      ├─ max_tokens REQUIRED
                                            └─ usage: input_tokens / output_tokens
      usage: prompt_tokens /
             completion_tokens                       │
      (+ DeepSeek cache hit/miss)                    │
                 └───────────────┬────────────────────┘
                                 ▼
              GenerateResult(code, tokens_in, tokens_out, usd, truncated)
```

Both normalize to one `GenerateResult`, so `SearchLoop` never learns which provider it is on. That
normalization is the whole point — `max_usd` is computed from it, so it has to be exact per provider,
which is why the Anthropic compatibility-endpoint shortcut was rejected.

### Price table: fail loudly, never silently

`Settings.price_per_1m` maps `(provider, model) -> (input_usd, output_usd)`, filled on Day 0 from each
provider's live pricing page. **Do not hardcode prices in this plan** — they move, and a stale number
in a budget cap is worse than no cap.

Startup asserts every configured model has a price entry. Without that assert the failure mode is
silent and expensive: an unknown model computes `$0.00` per node, `max_usd` never trips, and the
overnight run bills against a cap that was never armed.

### Per-action model override

`Settings` allows `model_draft`, `model_improve`, `model_debug`, all defaulting to one model. Lets you
run the bulk of the search on something cheap (DeepSeek) and reserve a stronger model for the debug
action, where reasoning quality actually changes the outcome. A dict lookup, not an architecture.

### Truncation is its own failure, not a contract violation

Anthropic **requires** `max_tokens`; OpenAI defaults it. Set too low, the model returns a `train.py`
that stops mid-function. That reads as malformed code and gets filed as `contract_violation`, so the
debug action tries to fix a syntax error that is really a config bug — and it will do it every time.

Detect it explicitly: OpenAI `finish_reason == "length"`, Anthropic `stop_reason == "max_tokens"`.
Both map to a distinct `truncated` outcome with its own message ("output hit the token ceiling, raise
`max_tokens`"), never to `contract_violation`. Default `max_tokens` generously (8192+) — a full
`train.py` with imports, feature engineering, and a 5-fold loop is not short.

### Retries

Both SDKs retry on 429/5xx internally via `max_retries`. Configure it there rather than hand-rolling
backoff. Exhausted retries mean the node is **never created** — no half-node in the tree — and the
`SearchLoop` circuit breaker counts it as an `infra_error`.

### Day-2 verification

First live call of the project prints the raw usage object for whichever provider is configured, and
the assembled `GenerateResult`. Confirms token accounting is real before any budget logic depends on
it. Repeat once per provider you actually intend to use.

## Module layout

```
kagglecracker/
  config.py              Settings: competition slug, model, budgets, docker flags, paths
  db/
    schema.sql
    store.py             ProtocolStore, RunStore, NodeStore, SubmissionStore, WorkerLock
  engine/
    contract.py          OUTPUT_CONTRACT, parse_metrics, SubmissionValidator
    prompts.py           blocks + PromptBuilder
    policy.py            SearchPolicy (seeded, deterministic)
    ranking.py           best_node, leaderboard  ← sole ranking authority
    loop.py              SearchLoop: budgets, resume, lock
  runtime/
    base.py              AgentRuntime protocol, GenerateResult, TokenUsage
    openai_compat.py     OpenAICompatibleRuntime — serves openai + deepseek
    anthropic_rt.py      AnthropicRuntime — native anthropic SDK
    pricing.py           price_per_1m table + startup assert
    fake.py              FakeRuntime (canned responses)
  executors/
    base.py              Executor protocol, RunResult, RunStatus
    local_docker.py      LocalDockerExecutor
    fake.py              FakeExecutor
  feeders/
    discussions.py       kaggle forums fetch → summarize → context/discussions/<date>.md
    eda.py               pandas summary in sandbox → summarize → context/eda_findings.md
  kaggle/
    client.py            subprocess wrapper: download, topics, submit --wait, submissions
  web/
    app.py routes.py templates/ static/
  cli.py                 worker / web / preflight / init
tests/
sandbox/Dockerfile
```

### Inline ASCII diagram placement (per your doc preference)

- `engine/loop.py` — the node lifecycle diagram above, at module top
- `engine/policy.py` — the action decision tree including both determinism fallbacks
- `engine/prompts.py` — the block composition table
- `db/store.py` — the protocol/node staleness relationship
- `executors/base.py` — the four terminal statuses and what produces each
- `tests/test_loop_integration.py` — what the fake harness wires together and why

---

## Test coverage plan

Greenfield: every path is a gap. Framework: **pytest** (no existing test infra in this repo).

```
CODE PATHS                                              USER FLOWS
[+] engine/policy.py                                    [+] Overnight run
  ├── next_action()                                       ├── [GAP] start → 20 nodes → clean stop on max_nodes
  │   ├── [GAP] step < 3 → draft                          ├── [GAP] stop on max_wall_clock_s
  │   ├── [GAP] improve sampled, no scored node → draft   ├── [GAP] stop on max_usd
  │   ├── [GAP] debug sampled, no buggy node → draft      ├── [GAP] kill mid-node → --resume continues
  │   ├── [GAP] improve, best at depth cap → best d<5     └── [GAP] second worker start → fails on lock
  │   ├── [GAP] improve, all at depth cap → draft
  │   └── [GAP] seeded RNG reproducibility              [+] Approve & submit
  │                                                       ├── [GAP] [→E2E] approve valid node → submitted
[+] engine/contract.py                                    ├── [GAP] approve node w/ bad submission → blocked
  ├── parse_metrics()                                     │                 + reason shown
  │   ├── [GAP] valid                                     └── [GAP] Kaggle rejects → error surfaced, not silent
  │   ├── [GAP] file missing
  │   ├── [GAP] malformed JSON                          [+] Dashboard
  │   ├── [GAP] cv_score missing / NaN / inf              ├── [GAP] empty tree renders
  │   └── [GAP] fold_scores length != protocol n_folds    ├── [GAP] 20-node tree renders
  └── SubmissionValidator                                 ├── [GAP] node detail 404
      ├── [GAP] valid                                     └── [GAP] leaderboard shows mean/std/min
      ├── [GAP] file missing
      ├── [GAP] column names differ from sample         [+] Protocol change (Day 4)
      ├── [GAP] row count differs                         └── [GAP] [→E2E] new protocol → old nodes drop
      ├── [GAP] id set differs                                            out of best + leaderboard
      ├── [GAP] duplicate ids
      ├── [GAP] NaN/inf in prediction column
      └── [GAP] unnamed index column written

[+] executors/local_docker.py                           [+] engine/ranking.py
  └── run()                                               ├── [GAP] higher_is_better
      ├── [GAP] exit 0 → ok                               ├── [GAP] lower_is_better
      ├── [GAP] exit != 0 → error + traceback             ├── [GAP] filters to is_current protocol
      ├── [GAP] SIGKILL @ timeout → timeout, no tb        ├── [GAP] excludes non-ok nodes
      ├── [GAP] daemon down → infra_error, clear msg      ├── [GAP] empty tree → None
      ├── [GAP] image missing → clear msg                 └── [GAP] tie-break is deterministic
      └── [GAP] run_flags recorded (criterion 5)

[+] runtime/llm.py                                      [+] kaggle/client.py
  └── generate()                                          ├── [GAP] download
      ├── [GAP] fenced code → extracted                   ├── [GAP] topics list/show parse
      ├── [GAP] prose only → contract_violation           ├── [GAP] submit success
      ├── [GAP] non-parseable → ast.parse rejects         ├── [GAP] submit rejected by Kaggle
      ├── [GAP] 429/500 → retry ×3 → node not created     └── [GAP] --wait returns public score
      └── [GAP] tokens + usd recorded

INTEGRATION: [GAP] [→E2E] full SearchLoop on FakeExecutor + FakeRuntime — draft ×3, improve, debug,
             depth cap, budget stop, resume. Runs in <1s, no Docker, no API cost.
             THIS IS SUCCESS CRITERION 4 AS A TEST.

LLM:  [GAP] [→EVAL] prompt snapshot per action (free, default suite)
      [GAP] [→EVAL] pytest -m eval — each action prompt × 5 vs real model against a 200-row fixture:
                    parses, imports only pinned packages, writes valid metrics.json + submission.csv.
                    Reports contract-violation rate per action. Opt-in, costs tokens.

COVERAGE: 0/47 paths tested (greenfield)  |  Code paths: 0/32  |  User flows: 0/15
GAPS: 47 (3 E2E, 2 eval)
```

Legend: ★★★ behavior + edge + error · ★★ happy path · ★ smoke · [→E2E] integration · [→EVAL] LLM eval

### Failure modes

| Codepath | Realistic failure | Test? | Error handling? | User sees? |
|---|---|---|---|---|
| `LocalDockerExecutor.run` | Docker Desktop not running (**currently true on this machine**) | yes | yes — `infra_error`, preflight catches it Day 1 | clear message |
| `LocalDockerExecutor.run` | 900s SIGKILL, no traceback | yes | yes — `timeout` status, own debug template | node marked timeout |
| `SubmissionValidator` | pandas `to_csv` without `index=False` | yes | yes — `contract_violation`, agent debugs it | node marked, diff shown |
| `LLMRuntime.generate` | 429 during overnight run | yes | yes — retry ×3, then stop run cleanly | run ends, budget preserved |
| `SearchLoop` | macOS sleeps the laptop at 3am | yes | yes — `caffeinate -i` + `--resume` | resumable |
| `ranking.best_node` | protocol filter forgotten in an ad-hoc query | yes | structural — single ranking helper | wrong node ranked |
| `worker_lock` | second worker started by accident | yes | yes — startup fails loudly | clear message |
| `worker_lock` | worker crashed, lock never released | yes | yes — heartbeat TTL, stealable after 3 misses | steal is logged |
| `kaggle submit` | Kaggle rejects the file | yes | yes — state persisted, error surfaced | error in UI |
| `kaggle download` | competition rules not accepted in browser | yes | yes — Day-0 preflight, clear instruction | blocked with a fix |
| approve flow | double-click submits the same node twice | yes | yes — `UNIQUE(node_id)` + disabled button | second click no-ops |
| `SearchLoop` | 20 consecutive `infra_error` nodes burn the budget | yes | yes — circuit breaker, `stop_reason` recorded | run stops, reason shown |
| `PromptBuilder` | context grows until it blows the context window | yes | yes — char budget, oldest dropped first | `prompt_chars` per node |

**Critical gaps (no test + no handling + silent): 0.** Every failure above is caught, tested, and visible.

---

## Build order (7 days)

**Day 0** — Pick playground competition (needs active submissions + tabular). Install `kaggle` CLI
(**not currently installed**), verify `>= 2.2.0`. **Accept the competition rules in a browser** — the
API refuses download until you do, and there is no code fix for it. Run the auth preflight
(`kaggle competitions list` proves credentials + network; a successful download proves rules accepted).
Smoke-test `competitions topics` + `forums topics show`. `kaggle competitions download`. Hardcode the
CV protocol from the evaluation page into the `protocols` table. Repo skeleton, `uv` project pinned to
Python 3.12, pytest wired.

**Day 0 live-boundary smoke test** (also from the outside voice): before any engine code exists, walk
the whole external boundary by hand once — download data, train a trivial model locally, write a
submission, submit it, read the public score back. That single pass proves auth, rules, quota, file
format, and score polling all work, on the day you can still react to it. It also gives you the naive
baseline number that success criterion 3 is measured against.

**Day 1 — COMPLETE.** `sandbox/` is its own uv project with its own `uv.lock`, deliberately separate
from the host project so neither can acquire the other's dependencies (the host has no sklearn; the
sandbox has no typer/openai). `LocalDockerExecutor` with the four terminal statuses. Run flags:
`--network none --cpus 4 --memory 8g --pids-limit 256 --cap-drop ALL --security-opt
no-new-privileges --user <host uid:gid> -v data:/data:ro -v run_dir:/workspace:rw`, recorded per
node. `contract.py`: `render_output_contract`, `parse_metrics`, `SubmissionValidator`. `FakeExecutor`.
`kagglecracker exec <file>` drives any local file through the sandbox and validates it.

| Result | |
|---|---|
| Reference node, end to end | `status ok`, 13.1s, **cv_score 0.81054** (folds 0.799–0.825, std 0.0094) |
| Tests | 44 unit in **0.11s** (no Docker, no API) + 5 `live` in 11.3s |
| Preflight | 8/8 pass |
| Image | 1.4 GB |

Two things the Day-1 run taught, both now permanent:

- **`python:3.12-slim` has no `libgomp1`**, so LightGBM/XGBoost/CatBoost all die at import with
  `libgomp.so.1: cannot open shared object file`. Inside the sandbox this is unfixable — there is no
  network, so the agent cannot install it — and the traceback points at the import line rather than
  at the image, so the debug action would retry forever against code that was never wrong. The
  Dockerfile now installs it **and** imports every library at build time, so a missing shared object
  fails the build instead of a node at 3am.
- **`--user <host uid:gid>` is required, not hygiene.** The image's non-root `runner` does not own the
  bind-mounted workspace, so without it every write fails with a permission error that also reads
  like a bug in the generated code. `HOME`/`MPLCONFIGDIR`/`XDG_CACHE_HOME` are pointed at
  `/workspace/.cache` for the same reason.

The `live` suite proves the security posture rather than asserting it from flags: a container that
tries to open a socket reports no network, and one that tries to write `/data` cannot.

**Day 2 — COMPLETE.** `AgentRuntime` + `OpenAICompatibleRuntime` (OpenAI, DeepSeek) +
`AnthropicRuntime` + `FakeRuntime` + `build_runtime` factory. `PromptBuilder` with composable blocks.
`NodeStore`/`RunStore`. `kagglecracker node` drives one node end to end.

**The agentic loop closed, and it self-healed:**

| node | action | status | cv_score | cost |
|---|---|---|---|---|
| 1 | draft | `error` — `NameError: lgb_early_stopping` | — | $0.0008 |
| 2 | debug | `error` — LightGBM 4.7 deprecated `eval_set` | — | $0.0009 |
| 3 | debug | **`ok`** | **0.80697** | $0.0010 |

**Design Open Question 4 is answered with a measurement: ~$0.0009 per node, so a 20-node run is
about $0.02.** The `max_usd` default of $5 is three orders of magnitude of headroom. Cost is not the
binding constraint on tree size; nothing is, at this price.

Token accounting verified live on all three providers before any budget logic was written — all
report usage, so `max_usd` is safe. DeepSeek and OpenAI report cached input under different names
(`prompt_cache_hit_tokens` vs `prompt_tokens_details.cached_tokens`); both are normalized, and cache
hits are already landing (768–896 tokens by node 3).

### The `truncated` status earned itself on the first real call

It fired immediately, twice. The draft consumed **all 8,192** output tokens and then **all 32,000**,
returning `stop_reason=length` and not one line of `train.py` — 5m24s and $0.009 for nothing.

The cause was not the ceiling. `deepseek-v4-flash` takes a `reasoning_effort` of low/high/max and
defaults high, and at that level these models expect an output budget in the *hundreds of thousands*
of tokens. Writing one training script is a well-specified job, not an open research problem.
Setting `reasoning_effort="low"`:

| | before | after |
|---|---|---|
| output tokens | 32,000 (all reasoning) | 2,442 |
| cost | $0.0090 | $0.0008 |
| wall clock | 5m 24s | 29s |
| result | nothing | working code |

Had `TRUNCATED` not been its own status, this would have been filed as "the model returned no code",
and the debug action would have spent the night trying to repair a config value. `reasoning_effort`
is now a setting, sent as `reasoning_effort` on OpenAI-compatible providers and as
`output_config.effort` on Anthropic — the two spell the same concept differently, which is precisely
why they are separate adapters.

Second confirmed instance of the version-drift risk flagged on Day 1: the model wrote LightGBM 4.0
idioms against 4.7. The debug loop fixed it unaided, which is the system working as designed.

**Day 3 — COMPLETE.** `SearchPolicy` (seeded, both fallbacks), `ranking.py` as sole ranking
authority, `SearchLoop` with three budget caps + circuit breaker + `--resume`, `worker_lock` with a
stealable heartbeat TTL. **Success criterion 4 is now a green test**: the whole loop runs on
`FakeExecutor` + `FakeRuntime` with no Docker, no network and no API key.

### The 20-node run

**20 nodes, 15.3 minutes, $0.0344, stopped on `max_nodes` as intended. 16/20 scored (80%).**

| | |
|---|---|
| Best node | **17 — cv 0.81997** (improve, depth 3) |
| Hand-written Day-1 reference | 0.81054 |
| Day-0 CryoSleep baseline | 0.7255 public LB |
| Action mix | improve 13, draft 5, debug 2 — close to the configured 70/15/15 |
| Max depth reached | 4 of a cap of 5 |

Success criterion 2 is satisfied **inside a single run**: node 8 (0.81721) and node 9 (0.81985) each
beat their parent, and node 11 recovered a crashed node 4 into a scoring 0.81882.

The leaderboard shows fold spread beside every score, which is the CV-overfit mitigation earning its
place: node 17 leads on mean (0.81997) but carries std 0.01070, while node 15 scores 0.81951 at std
0.00759. Those are different bets, and the human approving a submission can now see that rather than
taking the top row on faith.

### Three findings from the run

### Experiment: pandas 2.3.3, and feeding failures back into the prompt

Two arms, 20 nodes each, each in its own database so the tree summary could not
leak failures between arms.

| arm | sandbox | contract | scored | failures | version-drift failures | best cv | cost | wall |
|---|---|---|---|---|---|---|---|---|
| baseline | pandas 3.0.5 | stock | 16/20 | 4 (20%) | **3** | 0.81997 | $0.0344 | 15.3m |
| **A** | pandas 2.3.3 | stock | 16/19 | 3 (16%) | **0** | 0.82135 | $0.0297 | 15.6m |
| **B** | pandas 2.3.3 | + API notes | 17/20 | 3 (15%) | **0** | 0.81951 | $0.0339 | 19.6m |

**The failure-rate differences are noise.** 95% CIs: baseline [2%, 38%], A [0%, 32%],
B [0%, 31%] — near-total overlap. Detecting a genuine 20%→10% halving at 80% power needs
**~199 nodes per arm**, not 20. What the arms *can* resolve is the change in failure
*composition*, which is unambiguous.

**Verdict 1 — keep pandas 2.3.3.** The version-drift class went 3 → 0 → 0 and did not
reappear. Baseline failures were `Categorical categories cannot be null`,
`bad pandas dtypes: Cabin: str`, `ArrowStringArray has no attribute categories`. After the
downgrade every failure in both arms is a generic logic bug — wrong column lengths, a
mixed-type encoder input, an index error. The downgrade is also **free**: re-resolving
pinned only pandas 3.0.5 → 2.3.3, leaving numpy 2.5.1, scikit-learn 1.9.0, lightgbm 4.7.0,
catboost 1.2.10 and pyarrow 25.0.0 byte-identical.

**Verdict 2 — the API notes did not earn their place; flag stays off.** They cost +569
prompt chars (+6%), +163 input tokens per node, +14% money, and bought a difference
indistinguishable from zero.

**The experiment design had a flaw, and it explains the null result.** The notes were
written from baseline failures — which were *predominantly pandas drift*. Arm A had
already eliminated that class, so arm B was testing a remedy for a problem that no longer
existed. The honest conclusion is not "feeding failures back into the prompt doesn't
work", it is "these particular notes addressed an already-solved problem." A fair retest
would derive notes from arm A's *actual* residual failures (encoder dtype handling, column
alignment) and measure those.

`contract_api_notes` stays in the codebase as a config flag with the null result recorded,
rather than being deleted. Re-running it properly costs about $0.68 for two 200-node arms.

---

**1. pandas 3.0 was the dominant failure mode (now fixed — see the experiment above).** All five failures across the tree had *distinct*
signatures — the failure digest is working, nothing was rediscovered — but three of five were pandas
3.0 behaviour changes: `Categorical categories cannot be null`, `Fields with bad pandas dtypes:
Cabin: str`, and `'ArrowStringArray' object has no attribute 'categories'`. The contract states the
version but not what changed. **Day-4 action:** name the pandas 3.0 string-dtype shift explicitly in
the contract and measure the before/after failure rate. This is the highest-value prompt change
available and it is measurable.

**2. `improve` can produce a semantically-null change.** Node 16's fold scores are byte-identical to
its parent node 9's, with different code — it edited the file without changing a single prediction.
All 20 code bodies were distinct, so this is not duplication; it is a no-op edit the tree cannot
detect, and it costs a full node. Cheap fix worth considering: compare an improve's fold scores to
its parent's and, when identical, tell the next prompt that the previous change had no effect.

**3. The depth cap never bound.** Max depth reached was 4 against a cap of 5, so fallback (b) never
fired in the real run — it is exercised only by tests. Worth knowing before trusting it.

Cost remains a non-constraint: $0.0017/node, so the $5 cap allows roughly 2,900 nodes.

**Day 4 — COMPLETE, with two negative results that are worth more than the feature.**

Both feeders built on the single `AgentRuntime`. The EDA script is fixed, generic and mechanical —
tests assert it names no column of this competition and prescribes no treatment — so the statistics
are computed and the *interpretation* is the agent's. That line is the point: writing the
preprocessing into the prompt myself would raise the score and prove only that I can preprocess the
data.

**The EDA agent independently found the fact that drove the Day-0 protocol decision:** `PassengerId`'s
first component has 6,217 unique values, none shared with test. Derived from statistics, not told.

### Result 1 — context made the search significantly WORSE

| | A (no context) | C (EDA + discussions) |
|---|---|---|
| scored | 16/19 (84%) | 16/20 (80%) |
| **best cv** | **0.82135** | 0.81479 |
| **mean cv** | **0.81607** (sd 0.00395) | 0.81000 (sd 0.00499) |
| prompt | 8,919 ch | 13,594 ch (+52%) |
| cost | $0.0297 | $0.0379 (+28%) |
| wall clock | 15.6 min | 19.4 min (+24%) |

Mean difference **−0.00607**, 95% CI [0.00295, 0.00919], Welch t = 3.81 (df ≈ 28). Worse on every
axis, and unlike the pandas experiment this one is not noise.

**Caveat, stated plainly:** the t-test treats nodes as independent draws and they are not — an
improve chain inherits its parent's code, so the effective sample is smaller than 16 and the test
overstates significance. What survives that objection is the consistency: best, mean, cost and time
all moved the same way, and the failure log shows the mechanism.

**The mechanism is visible in C's failures.** The discussion summary recommended group features and
the CryoSleep/zero-spend rule; the agent built them (`GroupCryoMean`) and got the dtypes wrong —
twice. Context bought ambition the agent could not execute cleanly, then charged 52% more prompt for
it. Better context is not automatically better performance, and this is the design doc's own premise
2 ("the experiment loop is the core; feeders are context providers") coming back with evidence
attached.

### Result 2 — the error digest does not work, even on exact repeats

Nodes 13 and 15 failed with the identical string `Fields with bad pandas dtypes: GroupCryoMean:
object`. Node 15's prompt is reconstructable and it **contained that exact line**, under the heading
`Failures so far` and followed by `Do not reintroduce a failure that already appears above`.

It reintroduced it.

This is the first exact-repeat failure the project has produced, and the existing memory failed it.
Listing past errors and instructing the model not to repeat them **does not prevent repetition**.
That directly settles a design question: error memory has to carry the *fix*, not the error.
Lesson extraction from confirmed repairs — node 4 → node 11 was a clean 3-line diff — is the design
this evidence supports, and this run is the argument for it.

### What this changes

- Feeders stay built and tested, but **context is off by default** until it can be shown to pay.
- The next context experiment should test them **separately** (EDA alone vs discussions alone). C
  bundled both, so it cannot say which caused the regression — a design flaw I should have avoided
  after making the same mistake in experiment B.

**Day 5 — COMPLETE.** FastAPI + HTMX + Jinja, light numbered-sections theme per approved mockup
variant C. Four panels: solution tree (mermaid, server-rendered spec), leaderboard with fold
mean/std/min, node detail (code, folds, failure text, run log), context panel. `kagglecracker web`.

**The viewer is read-only by construction.** Every request opens SQLite with `mode=ro`, so a wrong
handler cannot corrupt a running search. The single exception is `POST /approve/<id>`, which is the
only write the human is allowed to make — and it validates the submission and refuses a duplicate
before recording anything.

**Panels the mockup shows and this system cannot honestly populate were left out** rather than filled
with plausible numbers: inference time per row, a public-LB estimate, a feature store, "23
experiments running" (we run one at a time, by design).

### The fold-spread column immediately earned its place

Live leaderboard against experiment A:

| rank | cv | fold std | fold min |
|---|---|---|---|
| 1 | 0.82135 | 0.01028 | 0.81024 |
| **2** | **0.82100** | **0.00799** | **0.81300** |
| 3 | 0.82066 | 0.01069 | 0.80610 |

Rank 2 is 0.00035 behind on the mean but has **lower variance and a higher worst fold**. On a search
that takes many draws against one fixed fold split, that is plausibly the better submission — and the
column exists so a human can see that instead of taking the top row on faith. This is the concrete
form of the CV-overfitting mitigation agreed in the eng review.

### Two bugs the tests caught

- **The tree displayed a score for a failed node.** `build_mermaid` keyed off `cv_score is not None`
  rather than `status == "ok"` — the invariant `ranking.py` already enforces. Latent today (failed
  nodes carry no score) but wrong, and the tree is the panel a demo actually shows.
- **The empty-database page crashed** in the header, which renders from `settings` the error path
  never passed.

Browser screenshotting was blocked by a Chrome tab-group restriction, so the layout was verified by
parsing the rendered HTML instead: mermaid spec decoded and checked edge-by-edge, leaderboard values
read back from the live page.

**Day 6** — `kaggle/client.py` submitter: approve → `SubmissionValidator` re-check → **idempotency
check (`UNIQUE(node_id)`, button disabled on submit)** → `kaggle competitions submit` → **poll
`kaggle competitions submissions <comp> --csv` until the row leaves `SubmissionStatus.PENDING`**
(no `--wait` flag exists — verified Day 0) → persist public score. Record the **CV-to-public-LB gap**
for the submitted node —
that one number is your only calibration on whether the fixed protocol was the right one. Error
handling on executor timeouts and crashes. **smolagents swap test**: reimplement one feeder on
`smolagents.CodeAgent` behind the identical `AgentRuntime` interface, engine untouched. README. Demo run.

**Day 7 / buffer** — Live prompt eval (`pytest -m eval`), coverage gaps, demo polish.

---

## NOT in scope

| Deferred | Rationale |
|---|---|
| Experiments chatbot | Cut to v2 (Issue 10). Protects Day 5 tree view and Day 6 submitter. |
| `SshDockerExecutor` / remote GPU | No machine exists. Interface proven by `FakeExecutor` test instead. |
| Parallel experiment execution | You explicitly want one at a time; CPU is the constraint. |
| Job queue / scheduler / Redis | `SearchPolicy` already decides order; a queue would be a second, competing ordering mechanism. |
| Enqueue-from-browser | Follows from no-queue. "More nodes" = rerun worker with a larger budget. |
| Search-policy tuning | Design's own named rabbit hole. Policy is frozen for week 1. |
| smolagents as the default runtime | Demoted to a Day-6 swap test (D1). Proves swappability better than using it by default. |
| gVisor / Firecracker / microVM | Threat model is accidental damage, not adversarial multi-tenancy. |
| CI/CD, registry publishing, docker-compose | Personal tool, one laptop. `git clone` + `uv sync` + `docker build`. |
| Daily discussion scheduler | v2. Now trivial since the CLI path is unattended. |
| Cytoscape.js tree | Only if mermaid proves too static on Day 5. |
| Porting anything from `../KaggleCracker` | Fresh start, per your instruction. |
| Cutting the web UI to a static page | Outside voice recommended it; **rejected**. The live tree is why Approach C was chosen over Approach A. Day 5 keeps its full day unconditionally. |
| A Day-4 go/no-go tripwire on the UI | Offered and declined. Day 5 proceeds regardless of Day 4 state. |
| Fixing a wrong CV protocol choice | Outside voice is right that `protocol_id` preserves history without helping you pick correctly. Mitigation is the Day-6 CV-to-LB gap measurement, not more machinery. |
| Using `aideml` as the pipeline | Its interpreter runs generated code via `exec()` on the host with no container. Incompatible with the sandbox constraint, which is half the learning goal. Read its prompts, do not import it. |
| Any agent framework that owns execution | Same reason as aideml and as smolagents `CodeAgent`. The Docker-sandbox contract means *we* execute; a pipeline that executes for us has to be fought, not configured. |
| Vector store / episodic memory layer | The tree is the memory structure. A second memory beside it duplicates the thing that makes AIDE work. |
| Cross-competition memory | One competition in week 1. Nothing to recall across. |
| LangChain for the LLM call | `generate(prompt) -> str` is one completion. `langchain-core` + `langchain-openai` for that is the same overbuild smolagents was demoted for. Provider SDKs directly. |
| OpenCode Zen | Dropped at your request. Any OpenAI-compatible host still works by changing `base_url`. |
| Anthropic's OpenAI-compatibility endpoint | Documented as a testing convenience with reduced coverage. `max_usd` is computed from the usage object, so it must be exact — native SDK instead. |
| Providers beyond OpenAI / DeepSeek / Anthropic | Not needed. `OpenAICompatibleRuntime` already absorbs any OpenAI-shaped host (OpenRouter, local vLLM) via config. |

## What already exists

- **`../KaggleCracker` (legacy, 173 Python files)** — a different system (`memory/`, `monitor/`,
  `triggers/`, `break_it_on_purpose_*.py`). No overlap with tree-search-over-ML-code. **Not reused.**
- **`KaggleCracker2`** — empty, one commit, no tracked files. Truly greenfield.
- **`kaggle` CLI ≥ 2.2.0** — provides discussion reading (`forums topics`, `competitions topics`),
  data download, submission, and score polling (`submit --wait`). **Reused for all four.** The design
  doc's manual-paste path, scraper option, and custom poll loop are all replaced by it.
- **`WecoAI/aideml`** — reference implementation of the same algorithm. **Cannot be used as the
  pipeline.** Verified in `aide/interpreter.py`: it executes generated code with
  `exec(compile(code, self.agent_file_name, "exec"), global_scope)` inside a `multiprocessing.Process`
  on the host. No container, no filesystem restriction, no network restriction — only a `SIGINT`
  timeout. That is the direct opposite of this project's #1 constraint (Docker, `--network none`,
  read-only data mount). Adopting it means deleting the sandbox learning goal, which is half the
  project. **Still read its prompt templates and journal schema** before inventing yours.
- **uv 0.11.26, Docker 29.5.2** — present. Docker **daemon not currently running**; `kaggle` CLI **not
  installed**. Both are Day-0/Day-1 preflight items.

---

## Implementation Tasks

Synthesized from this review's findings. Each derives from a specific finding above.

- [ ] **T1 (P1, human: ~3h / CC: ~15min)** — engine/contract.py — Build `SubmissionValidator` against `sample_submission.csv`
  - Surfaced by: Architecture Issue 2 — design doc:93 validates metrics.json but never submission.csv
  - Files: `kagglecracker/engine/contract.py`, `tests/test_contract.py`
  - Verify: `pytest tests/test_contract.py` — 8 validator branches green
- [ ] **T2 (P1, human: ~2h / CC: ~10min)** — db — Add `protocols` table + `protocol_id` on nodes
  - Surfaced by: Code Quality Issue 5 — design doc:96 staleness rule has no column to express it
  - Files: `kagglecracker/db/schema.sql`, `kagglecracker/db/store.py`, `kagglecracker/engine/ranking.py`
  - Verify: `pytest tests/test_ranking.py::test_protocol_change_drops_old_nodes`
- [ ] **T3 (P1, human: ~4h / CC: ~20min)** — engine/loop.py — Budget caps + `--resume` + `worker_lock`
  - Surfaced by: Architecture Issue 4 — design doc:111 caps iterations, not cost; no recovery path
  - Files: `kagglecracker/engine/loop.py`, `kagglecracker/db/store.py`, `kagglecracker/cli.py`
  - Verify: `pytest tests/test_loop_integration.py` — three stop conditions + resume + double-start
- [ ] **T4 (P1, human: ~1.5d / CC: ~1h)** — tests — `FakeExecutor` + `FakeRuntime` full-loop harness
  - Surfaced by: D2 — success criterion 4 was "passes on paper: interface review"
  - Files: `kagglecracker/executors/fake.py`, `kagglecracker/runtime/fake.py`, `tests/test_loop_integration.py`
  - Verify: `pytest tests/test_loop_integration.py` runs in <1s with no Docker and no API key
- [ ] **T5 (P1, human: ~2h / CC: ~10min)** — executors/base.py — `RunStatus` enum + per-status debug templates
  - Surfaced by: Architecture folded finding — a SIGKILL at 900s produces no traceback
  - Files: `kagglecracker/executors/base.py`, `kagglecracker/engine/prompts.py`
  - Verify: `pytest tests/test_executor.py::test_timeout_has_no_traceback`
- [ ] **T6 (P1, human: ~2h / CC: ~10min)** — feeders/discussions.py — Official `kaggle forums` ingestion
  - Surfaced by: Architecture Issue 1 — kaggle-cli v2.2.0 made the manual-paste workaround obsolete
  - Files: `kagglecracker/feeders/discussions.py`, `kagglecracker/kaggle/client.py`
  - Verify: `kaggle competitions topics <comp>` smoke test Day 0; `pytest tests/test_kaggle_client.py`
- [ ] **T7 (P2, human: ~3h / CC: ~15min)** — engine/prompts.py — Composable blocks, one `OUTPUT_CONTRACT`
  - Surfaced by: Code Quality Issue 6 — three actions sharing 80% of a natural-language contract
  - Files: `kagglecracker/engine/prompts.py`, `tests/test_prompts.py`
  - Verify: `pytest tests/test_prompts.py` — 3 snapshots; contract constant imported by validator too
- [ ] **T8 (P2, human: ~2h / CC: ~10min)** — web — Leaderboard shows fold mean / std / min
  - Surfaced by: Architecture Issue 3 — 20 draws against one fixed fold split overfits CV
  - Files: `kagglecracker/engine/ranking.py`, `kagglecracker/web/templates/leaderboard.html`
  - Verify: `pytest tests/test_ranking.py::test_leaderboard_reports_variance`
- [ ] **T9 (P2, human: ~1h / CC: ~10min)** — cli.py — `preflight` command
  - Surfaced by: Architecture folded finding — `docker info` currently fails; `kaggle` CLI absent
  - Files: `kagglecracker/cli.py`
  - Verify: `kagglecracker preflight` exits non-zero with a clear message on this machine today
- [ ] **T10 (P2, human: ~1h / CC: ~5min)** — sandbox/Dockerfile — Harden container flags
  - Surfaced by: Architecture folded finding — `--network none` alone is not the whole posture
  - Files: `sandbox/Dockerfile`, `kagglecracker/executors/local_docker.py`
  - Verify: `run_flags_json` in a node row contains `--cap-drop ALL`, `--security-opt no-new-privileges`, `--network none`
- [ ] **T11 (P2, human: ~4h / CC: ~20min)** — tests — Opt-in live prompt eval
  - Surfaced by: Test Issue 7 — snapshots prove the string, not that the prompt works
  - Files: `tests/test_prompt_eval.py`, `tests/fixtures/tiny_tabular/`
  - Verify: `pytest -m eval` reports contract-violation rate per action; excluded from default run
- [ ] **T12 (P3, human: ~3h / CC: ~15min)** — runtime — smolagents swap test
  - Surfaced by: D1 — smolagents demoted from default to proof-of-swappability
  - Files: `kagglecracker/runtime/smolagents_runtime.py`
  - Verify: one feeder runs on `CodeAgent` with zero changes under `kagglecracker/engine/`
- [ ] **T13 (P1, human: ~1h / CC: ~10min)** — cli.py — Kaggle auth + rules preflight, and the Day-0 manual boundary walk
  - Surfaced by: Outside voice #1 — no dependency on credentials, login state, or competition permissions
  - Files: `kagglecracker/cli.py`, `README.md`
  - Verify: `kagglecracker preflight` fails clearly when rules are unaccepted; manual download→submit→score round trip completes on Day 0
- [ ] **T14 (P1, human: ~1h / CC: ~10min)** — db + web — Submit idempotency
  - Surfaced by: Outside voice #2 — "a double-click can burn submission quota"
  - Files: `kagglecracker/db/schema.sql`, `kagglecracker/web/routes.py`
  - Verify: `pytest tests/test_submissions.py::test_double_submit_is_rejected`
- [ ] **T15 (P1, human: ~2h / CC: ~10min)** — engine/loop.py — Circuit breaker on consecutive failures
  - Surfaced by: Outside voice #4 — budget caps do not stop grinding through one failure mode
  - Files: `kagglecracker/engine/loop.py`
  - Verify: `pytest tests/test_loop_integration.py::test_stops_after_consecutive_infra_errors`
- [ ] **T16 (P2, human: ~1h / CC: ~10min)** — db/store.py — Stale worker-lock TTL and steal
  - Surfaced by: Outside voice #3 — "a dead worker can brick the run"
  - Files: `kagglecracker/db/store.py`
  - Verify: `pytest tests/test_worker_lock.py::test_stale_lock_is_stealable`
- [ ] **T17 (P2, human: ~2h / CC: ~10min)** — engine/prompts.py — Context size budget
  - Surfaced by: Outside voice #5 — "no plan for context-size limits, caching, or refresh cadence"
  - Files: `kagglecracker/engine/prompts.py`
  - Verify: `pytest tests/test_prompts.py::test_context_truncates_oldest_first`; `prompt_chars` present on every node row
- [ ] **T18 (P1, human: ~3h / CC: ~20min)** — runtime — Two provider adapters + normalized `GenerateResult`
  - Surfaced by: your requirement — OpenAI, DeepSeek, Anthropic support with exact per-provider usage
  - Files: `kagglecracker/runtime/base.py`, `openai_compat.py`, `anthropic_rt.py`, `kagglecracker/config.py`
  - Verify: `pytest tests/test_runtime.py` — both adapters normalize usage identically against recorded responses; one live call per provider prints raw usage
- [ ] **T18b (P1, human: ~1h / CC: ~10min)** — runtime/pricing.py — Price table + startup assert
  - Surfaced by: provider review — an unpriced model computes $0.00/node and `max_usd` never arms
  - Files: `kagglecracker/runtime/pricing.py`, `kagglecracker/config.py`
  - Verify: `pytest tests/test_pricing.py::test_unknown_model_fails_at_startup`
- [ ] **T18c (P1, human: ~1h / CC: ~10min)** — runtime — `truncated` as a distinct outcome
  - Surfaced by: provider review — Anthropic requires `max_tokens`; a low ceiling looks like malformed code and the debug action chases a config bug forever
  - Files: `kagglecracker/runtime/base.py`, `openai_compat.py`, `anthropic_rt.py`, `kagglecracker/engine/loop.py`
  - Verify: `pytest tests/test_runtime.py::test_length_finish_reason_maps_to_truncated_not_contract_violation`
- [ ] **T19 (P1, human: ~30min / CC: ~5min)** — engine/prompts.py — `tree_summary` spans the protocol, not the run
  - Surfaced by: memory review — cross-run amnesia makes the agent redraft models it already tried
  - Files: `kagglecracker/engine/prompts.py`, `kagglecracker/db/store.py`
  - Verify: `pytest tests/test_prompts.py::test_tree_summary_spans_runs_on_same_protocol`
- [ ] **T20 (P2, human: ~1h / CC: ~10min)** — engine/prompts.py — `known_failures` digest in `tree_summary`
  - Surfaced by: memory review — repeated identical failures teach the tree nothing
  - Files: `kagglecracker/engine/prompts.py`
  - Verify: `pytest tests/test_prompts.py::test_known_failures_groups_by_status_and_signature`
- [ ] **T21 (P3, human: ~30min / CC: ~5min)** — config.py — Per-action model override
  - Surfaced by: provider review — run the bulk of the search on a cheap model, reserve a stronger one for debug
  - Files: `kagglecracker/config.py`, `kagglecracker/runtime/openai_compat.py`
  - Verify: `pytest tests/test_config.py::test_per_action_model_defaults_to_single_model`

## Parallelization

| Step | Modules touched | Depends on |
|---|---|---|
| Sandbox + executor | `executors/`, `sandbox/` | — |
| Contract + validator | `engine/contract.py` | — |
| DB + schema | `db/` | — |
| Runtime + prompts | `runtime/`, `engine/prompts.py` | contract |
| Policy + loop + ranking | `engine/policy.py`, `loop.py`, `ranking.py` | db, runtime, executors |
| Feeders | `feeders/`, `kaggle/` | runtime, prompts |
| Web | `web/` | db, ranking |
| Submitter | `kaggle/`, `web/` | contract, web |

```
Lane A: sandbox + executor          (independent)
Lane B: contract + validator        (independent)
Lane C: db + schema                 (independent)
        └──► Lane D: runtime + prompts (needs B)
                     └──► Lane E: policy + loop + ranking (needs A, C, D)  ← critical path
                                  ├──► Lane F: feeders
                                  └──► Lane G: web ──► Lane H: submitter
```

Launch A + B + C in parallel worktrees, merge, then D, then E. F and G are parallel after E.
**Conflict flag:** Lanes G and H both touch `web/` — run sequentially, H after G.
Realistically this is a solo week, so the value here is knowing that Days 1-2 contain three
genuinely independent pieces you can hand to parallel agents.

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 17 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |
| Outside Voice | `/codex exec` | Independent plan challenge | 1 | issues_found | 9 findings: 5 adopted, 2 rejected, 2 partial |

**CROSS-MODEL:** Codex challenged the web UI as scope creep in a 1-week PoC. Rejected — it had not
read the design doc, where the live tree is the stated reason Approach C was chosen over the safer
Approach A. Its five concrete gaps (Kaggle auth/rules preflight, submit idempotency, stale-lock
recovery, circuit breaker, context size budget) were all real and all adopted.

**FOLLOW-UP PASS** (memory, pipeline choice, LLM providers): 6 tasks added (T18, T18b, T18c, T19-T21).
Key verified finding — `aideml` executes generated code with `exec()` on the host, no container, so it
cannot be the pipeline under this project's sandbox constraint. Providers settled at OpenAI +
DeepSeek + Anthropic behind two adapters.

**VERDICT:** ENG CLEARED — ready to implement. 23 tasks (13×P1, 8×P2, 2×P3), 0 critical gaps.

**OPEN RISK (a Day-2 measurement, not a decision):** token accounting must be confirmed live per
provider before `max_usd` is trusted. Until then `max_nodes` and `max_wall_clock_s` are the hard stops.

NO UNRESOLVED DECISIONS
