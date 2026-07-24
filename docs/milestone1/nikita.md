# Milestone 1 — Nikita — Recon slice

**Graded commit:** [`<fill in after committing>`](https://github.com/homomorfism/KaggleCracker/commit/<hash>) — the dataset-reconnaissance slice of our shared agent, targeting the live Kaggle competition *Playground Series S6E7*.

## 1. What I built

Two self-designed tools (plus the gated variant of the second), all stdlib-only, each with a *when-not-to-call* line in its description:

- **`profile_dataset(path, checks, target, compare_path, sample_rows)`** — computes per-column statistics for a CSV under `workspace/data`. `checks` is a constrained *and required* list-enum (`missingness | dtypes | cardinality | target_balance | numeric_summary | correlation_with_target | train_test_drift`): the model must say which questions it is asking, so every plan is traceable to the checks it was grounded on. `sample_rows` is capped at 200 000 in the schema, so an oversized ask is rejected by string comparison before a single row is read. `train_test_drift` compares train against test: standardized mean difference for numeric columns, total variation distance for categorical ones — plus the categories that appear in test but were never seen in train, the case that silently breaks anything fitted on train categories.
- **`write_preprocessing_plan(dataset, plan_markdown)`** — persists the plan to `workspace/plans/<dataset>.plan.md`. `dataset` is constrained to a filename-safe charset because it doubles as the filename; the precheck refuses if a plan already exists and names the tool that may replace it.

## 2. The loop

My slice observes (profile reports), reasons (the model — not a tool — aggregates the reports into recommendations), acts (writes the plan), and verifies through the shared envelope. I deliberately did **not** build an "aggregate findings" tool: synthesis is the reason step, and it belongs to the model. Beyond the shared step cap, my **stopping condition is `plan_written(dataset)`**: like the exec slice's plateau detector it reads the durable artifact (does the plan file exist?), not the transcript. The integration test proves the property that matters: after the plan lands, `run_agent` stops *without consulting the model again*.

## 3. The gate

**`overwrite_preprocessing_plan`** is the slice's irreversible action: destroying the current plan orphans every experiment derived from it. Writing a *new* plan stays ungated — delete the file and nothing happened. The interesting part is that whether "write this plan" is irreversible **depends on world state**, and the frozen core reads irreversibility as a static `ToolSpec` flag; the resolution is two specs sharing one body, each precheck rejecting the world state that belongs to its sibling (a write onto an existing plan is redirected, an overwrite of nothing is `not_found` — a human is never asked to approve a no-op). The preview shows the doomed plan's first lines, its size, and that recovery is impossible. Only `yes`/`y` approve; denial tests assert the consequence: old contents *and* mtime unchanged.

## 4. The error branch

**Dirty values are data; broken structure is an error.** A column mixing `N/A`, `""` and `3.7` comes back as an *ok* finding (`mixed: true`) — that is exactly what a preprocessing plan needs to know. A file whose line 5 has two fields under a three-field header comes back as `bad_input` naming line 5, because no statistic computed from a non-rectangle can be trusted. `break_it_on_purpose_recon.py` runs the *same* reactive model against both versions: with the branch it answers *"NOT writing a plan for a file I could not actually read"*; without it, a naive profiler force-fits the rows and the agent files a confident plan telling us to impute a "missing" status value that is actually a padding artifact of the broken row. Transcripts: `runs/recon_with_branch.txt` / `recon_without_branch.txt`.

## 5. How I tested it

51 tests (suite 182 green), all through the real `dispatch()` order — validate → precheck → gate → body: schema rejections proven to run before the body (the table reader is monkeypatched to explode if touched), one test per error kind (`bad_input`, `not_found`, `denied_by_human`), every happy-path number hand-computed from 4–10-row fixtures (r = 1.0 for y = 2x, SMD = 100/√1.25 for the shifted column), gate approve *and* deny asserting on the filesystem, and both stopping tests (condition fires without a further model call; cap raises `StepLimitReached`).

**What surprised me:** the naive profiler's output is *indistinguishable on its face* from a real report — same shape, same plausible numbers; the force-fit leaves no trace in the output. No downstream check could have caught it, because the lie is born before any statistic is computed. That is why the structural check must be an error branch at the source, not a sanity check later. Smaller surprise: standardized mean difference is undefined for a constant column (std = 0), so my drift check falls through to the categorical comparison for it — found by hand-computing the fixture, not by the first version of the code.
