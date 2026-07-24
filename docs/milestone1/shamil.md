# Milestone 1 — Shamil Arslanov — Knowledge slice

**Graded commit:** [`70399e8`](https://github.com/homomorfism/KaggleCracker/commit/70399e899247a3f5babc70f7ea6fc26b15688d79) — the knowledge-mining slice of our shared agent, targeting the live Kaggle competition *Playground Series S6E7*.

## 1. What I built

Two self-designed tools, both doing something real (live Kaggle API, persisted JSONL state), each with a *when-not-to-call* line in its description:

- **`search_kaggle_discussions(competition_slug, query, sort, max_results)`** — searches the public Kaggle work for a competition. `sort` is a constrained enum (`hot | recent | votes`) because only those three orders answer a question the agent actually has; `max_results` is capped at 50 *in the schema*, so an oversized ask is rejected by string comparison before any network traffic.
- **`save_technique_note(competition_slug, technique, evidence_url, confidence, est_effort)`** — appends one evidenced technique to `workspace/knowledge/{slug}.jsonl`. `confidence` is a constrained enum (`confirmed_cv_gain | claimed | speculative`) because the difference between a measured gain and a comment-section claim decides which experiments run first — it cannot be free text. Idempotent on duplicates.

## 2. The loop

My slice observes (search results), reasons (which techniques are worth keeping), acts (saves notes / asks to fetch data), and verifies through the shared envelope — every tool returns `{ok, data|error}` and the loop branches on it. Beyond the shared step cap, my **stopping condition is stall detection**: the search tool writes a durable `search_log.jsonl`, and `stall_detector()` stops the run when the same query is issued twice or two consecutive searches surface no new source. It fired on a real run: `runs/knowledge_e2e_run.txt`, after 7 distinct searches and 18 saved notes.

## 3. The gate

**`fetch_external_dataset(dataset_ref, merge_into_training)`** is the slice's irreversible action: merged external data changes every downstream experiment, carries competition-rules obligations (must be public and disclosed — recorded in `external_manifest.jsonl`), and can silently leak the target. The preview tells the human exactly that. Only `yes`/`y` approve; a closed stdin is a denial — absence of a human is never consent. On denial the tests assert the *consequence*: no directory created, no manifest line written. The other two tools stay ungated on purpose — the contrast is the point. A precheck rejects malformed or already-fetched refs *before* the gate, so a human is only ever asked the one question a machine cannot answer.

## 4. The error branch

*"The API returned zero results"* is **valid data** (early in a Playground month there may genuinely be nothing yet); *"the API returned 429"* is an **error** (`rate_limited`, retryable). The naive version collapses both into an empty list — and the model then concludes, with total confidence, that nobody has discussed the competition. `break_it_on_purpose_knowledge.py` runs the *same* reactive model against both versions: with the branch it answers *"corpus state UNKNOWN, retrying later"*; without it, *"no prior public work exists, starting from scratch"*. The two transcripts sit side by side in `runs/knowledge_with_branch.txt` and `runs/knowledge_without_branch.txt`.

## 5. How I tested it

50 tests (suite 131 green), all driven through the real `dispatch()` order — validate → precheck → gate → body — using the shared `FakeModel` and a monkeypatched HTTP layer: schema rejections proven to run before the body, one injected failure per error kind (429, 404, timeout, unreachable network, unparseable JSON), gate approve *and* deny asserting on the filesystem, stall detection unit-tested plus a `run_agent` integration test proving the loop stops without consulting the model. Then one end-to-end run against the real model (`claude-sonnet-5`) over the live Kaggle API.

**What surprised me:** two things, both from the real-model run. First, my stall detector fired *before* the model had saved a single note — it batched all its searches upfront, went dry twice, and the loop cut it off mid-plan; the fix was task cadence (save after each search), not code. Second, the live model once omitted the required `confidence` parameter; the validator rejected it as `bad_input` before the tool body ran, the error reached the loop as its own branch, and the model corrected itself on the next step — the exact machinery of requirements 1 and 4, firing unscripted.
