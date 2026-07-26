# KaggleCracker — Milestone 1 Plan

**Target competition:** [Playground Series S6E7 — Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7)
**Team:** Nikita, Shamil, Diganta
**Milestone:** Session 3 agent assignment, applied to a real competition
**Duration:** 7 working days

---

## 0. Read this first — S6E7 is not a finished competition

The class project description says *"pick 2–3 finished competitions and see where our submissions would have landed on the private leaderboard."* S6E7 is the **current / upcoming** episode of Season 6. That changes three things, mostly for the better:

**Better:**

- **Zero solution leakage.** Nobody has published a 1st-place writeup, because there isn't one yet. Any score the agent gets is earned, not retrieved. This is the cleanest possible evaluation and it removes the biggest methodological hole in the project.
- **A real private leaderboard at the end.** If our timeline lines up with the competition close, we get an actual private-LB placement instead of a simulated late submission. That is a much stronger result to report.
- **The submission gate becomes real.** Live competitions cap submissions per day (confirm the exact number on the rules page). An agent that burns the daily quota on a broken CSV has done real, unrecoverable damage. The approval gate in requirement 3 stops being a classroom exercise.

**Worse:**

- Fewer discussions and notebooks exist early in the month, so Shamil's mining tools have a thinner corpus in week 1. It fills up fast.
- We cannot self-score offline. Ground truth is unavailable until close, so **local CV is the only feedback signal the agent gets** during development. That is realistic, and it is the right thing to build anyway.

**Decision:** use S6E7 as the *live* target, and pick **one closed episode (S6E4 or S6E5) as the regression fixture** — a competition where we know the answer, used for testing the pipeline offline without spending submissions. Do not skip this. Everything in `tests/` runs against the closed episode.

---

## 1. Day 0 checklist — 30 minutes, do it together, before writing code

Kaggle pages are JS-rendered and change without notice, so confirm these by hand and write the answers into this file:

- [x] **Metric** — **Balanced Accuracy Score, higher is better** (confirmed 2026-07-24 via Kaggle API `evaluationMetric`). Write the formula into `core/metrics.py` on day 1.
- [x] **Target** — `health_condition`, 3-class multiclass: `at-risk` 592,561 (85.9%), `unhealthy` 57,724 (8.4%), `fit` 39,803 (5.8%) (confirmed 2026-07-26 from `train.csv`). Heavy imbalance — this is why the metric is Balanced Accuracy; per-class recall matters, plain accuracy would score 0.859 for predicting the majority class.
- [x] **Dates** — enabled 2026-07-01, entry/merger deadline and final submission deadline both **2026-07-31 23:59 UTC** (confirmed 2026-07-24 via Kaggle API).
- [x] **Submission quota** — **10 submissions per day** (confirmed 2026-07-24 via Kaggle API `maxDailySubmissions`). This number goes into the gate's warning message.
- [x] **Data size** — `train.csv` 62.7 MB, 690,088 rows × 15 cols; `test.csv` 24.6 MB, 295,753 rows × 14 cols; `sample_submission.csv` 4.4 MB (confirmed 2026-07-26, downloaded to `workspace/data/`).
- [x] **External data rule** — public external data allowed, standard Playground rules (confirmed 2026-07-26: top-voted public notebooks openly merge the original source dataset; competition rules page is JS-rendered so also eyeball it once by hand before we disclose external data in a submission).
- [x] **Original source dataset** — [`ziya07/college-student-health-behavior-dataset`](https://www.kaggle.com/datasets/ziya07/college-student-health-behavior-dataset) (confirmed 2026-07-26). 50,000 rows, identical feature columns + `health_condition` target; extra columns `student_id`, `timestamp` (no `id`). Training on synthetic + original combined is the first knowledge-slice target.
- [x] **Late submission** — verified 2026-07-26 on `playground-series-s6e5` (closed): `sample_submission.csv` submitted late, scored `COMPLETE`, public/private 0.50000, submission id 55007125. Closed-episode late submission works, so S6E5 is usable as the offline regression fixture. Note: the API 403s until the episode's rules are accepted on the website **while logged in as the token's account** — that cost us two false starts, worth a line in a one-pager.
- [ ] **Kaggle API token** — all three of us have `~/.kaggle/kaggle.json` working. Test with `kaggle competitions list`. *Shamil (`hashshes`) verified 2026-07-26; Nikita and Diganta still need to confirm theirs.*

---

## 2. What we are graded on, and what follows from it

Three separate commits, three separate one-pagers, one shared repo. Commits by someone else do not count for you.

The safe reading of the task is that **all four requirements must be visible in each person's own commit**:

1. Two tools you designed yourself, each with an action-shaped name, a description saying when *and when not* to call it, and at least one constrained parameter.
2. An observe → reason → act → verify loop with an explicit stopping condition.
3. One human approval gate on an irreversible action; reversible actions ungated.
4. One tool error that reaches the loop as its own branch, not as valid data.

The alternative reading — "the group's agent has these properties collectively" — leaves whoever didn't write the gate with nothing to put under the "approval gate" heading of their page. Don't take that risk.

**Therefore: one shared core loop, three vertical slices.** Each slice = 2 tools + 1 gated irreversible action + 1 error branch + 1 distinctive stopping condition.

---

## 3. Repository layout

```
core/                    SHARED — frozen after Day 1
  loop.py                the ~15-line agent loop
  contracts.py           tool result envelope
  registry.py            tool registration + irreversible flag
  gate.py                human approval gate
  errors.py              error kinds
  metrics.py             S6E7 metric, implemented once
tools/
  recon/                 NIKITA
  knowledge/             SHAMIL
  exec/                  DIGANTA
workspace/               agent-writable scratch (gitignored except .gitkeep)
  data/                  downloaded competition data
  plans/                 preprocessing plans
  knowledge/             mined notes, jsonl
  experiments/           run artifacts
  submissions/           generated CSVs
runs/                    saved transcripts — evidence for the writeups
tests/
  core/
  recon/  knowledge/  exec/
  fixtures/              broken CSVs, canned API responses, closed-episode sample
```

One owner per directory under `tools/` and `tests/`. Near-zero merge conflicts by construction.

---

## 4. Day 1 — mob session, ~2 hours, nobody splits off

Produces a co-authored commit that counts for nobody. That's fine; it's the foundation.

### 4a. Write the loop by hand

**Not with the coding assistant.** All three of us should be able to redraw this on a whiteboard afterwards. That is the entire point of the exercise.

```python
def run_agent(messages, tools, max_steps=12):
    for step in range(max_steps):
        reply = model(messages, tools)
        messages.append(reply)

        if not reply.tool_calls:
            return reply.text                     # done-signal stop

        for call in reply.tool_calls:
            result = dispatch(call)               # gate + error branch live here
            messages.append(tool_message(result))

    raise StepLimitReached(messages)              # explicit stop, not a silent return
```

### 4b. Freeze the contract

Every tool returns this envelope. No bare strings, no raw exceptions escaping.

```python
# core/contracts.py
{"ok": True,  "data": {...}}
{"ok": False, "error": {"kind": ErrorKind, "msg": str, "retryable": bool}}

ErrorKind = "not_found" | "bad_input" | "timeout" | "rate_limited" \
          | "exec_failed" | "denied_by_human" | "quota_exhausted"
```

### 4c. Registry and gate

Tools register with an `irreversible: bool` flag. The gate reads the flag — **nobody hardcodes tool names into the gate**.

Gate rules, agreed once, applied by all three:

- Prints *what will happen*, *to what*, and *what will be lost or spent*.
- Accepts only `yes` / `y`, case-insensitive. Everything else — including empty input, `ok`, `sure`, `go ahead` — is a **denial**.
- On denial, returns `{"ok": False, "error": {"kind": "denied_by_human", "retryable": False}}` and feeds it back into the loop as a normal observation, so the model can revise its plan instead of crashing.
- **Test the denial path.** Most people only test approval.

### 4d. Error handling rules

The loop must branch on `ok == False`, visibly and differently from the success path. Minimum behaviour:

- Append a distinctly-formatted tool message so the model can tell failure from data.
- Increment a per-tool failure counter.
- After 2 consecutive failures of the same tool, stop offering that tool for the rest of the run.

That last rule is cheap and gives everyone something concrete to write under "how I tested it".

### 4e. Also on Day 1

- `core/metrics.py` with the S6E7 metric, plus a unit test against a hand-computed example.
- A `KaggleClient` wrapper with a **dry-run mode on by default**, so no one accidentally spends a submission during development.

---

## 5. The three slices

| | **Nikita — Recon** | **Shamil — Knowledge** | **Diganta — Execute & Submit** |
|---|---|---|---|
| **Tool 1** | `profile_dataset` | `search_kaggle_discussions` | `run_experiment` |
| **Tool 2** | `write_preprocessing_plan` | `save_technique_note` | `record_experiment_result` |
| **Gated action** | overwriting an existing plan | downloading external data into the training set | `submit_to_kaggle` |
| **Error branch** | malformed / missing CSV | rate limit ≠ zero results | crash / timeout / no score printed |
| **Stop condition** | done-signal + step cap | stall detection (repeated query, no new sources) | plateau (no CV gain in N runs) |

### 5.1 Nikita — dataset recon

```python
profile_dataset(
    path: str,                      # required
    checks: list[Literal[           # constrained enum
        "missingness", "cardinality", "leakage",
        "target_balance", "dtypes", "train_test_drift"
    ]],
    sample_rows: int = 100_000      # narrow type, capped in schema
)
```

Description must state when *not* to call it: e.g. *"Do not call on files over 500 MB — call `sample_dataset` first. Do not call to inspect a single column; use it for whole-file profiling only."*

`train_test_drift` is the one worth building carefully: Playground synthetic data sometimes has distribution shift between train and test, and catching it is a genuine competitive edge, not a box-ticking check.

```python
write_preprocessing_plan(plan: dict, overwrite: bool = False)
```

Writing a *new* plan is reversible → ungated. **Overwriting an existing one is the gated action.** Nice property to write about: the same tool is gated or not depending on world state, which is more interesting than a blanket "this tool is dangerous" flag.

**Error branch:** feed it `tests/fixtures/ragged.csv` — a file with an inconsistent column count and a column mixing `"N/A"`, `""`, `"null"`, `3.7`. The tool returns `bad_input` with a message the model can act on. Without this, pandas raises, the traceback is stringified into the transcript, and the model happily "profiles" a file it never read.

**Stopping condition:** done-signal (model returns a plan and stops calling tools) plus the step cap.

### 5.2 Shamil — knowledge mining

Note the adjustment for a Playground competition: **academic literature is close to useless here.** The data is synthetic and the winning methods are gradient boosting and ensembling, not novel architectures. Point the tools at Kaggle discussions, public notebooks, and — most importantly — the original real-world dataset the synthetic data was generated from.

```python
search_kaggle_discussions(
    competition_slug: str,                              # required
    query: str,                                         # required
    sort: Literal["hot", "recent", "votes"] = "votes",  # enum
    max_results: int = 10                               # capped at 50 in schema
)
```

```python
save_technique_note(
    competition_slug: str,
    technique: str,
    evidence_url: str,
    confidence: Literal["confirmed_cv_gain", "claimed", "speculative"],
    est_effort: Literal["low", "medium", "high"]
)
```

Appends to `workspace/knowledge/{slug}.jsonl`. Appending is reversible → ungated.

**Gated action:** `fetch_external_dataset(dataset_ref, merge_into_training: bool)`. Downloading third-party data and merging it into the training set is irreversible in the sense that matters — it changes every downstream experiment, it has competition-rules implications (external data must be public and disclosed), and it can silently leak the target. This is the strongest safety argument of the three slices; say so in your one-pager.

**Error branch — Shamil's best material:** *"the API returned zero results"* is **valid data**. *"the API returned 429"* is an **error**. Naive implementations collapse both into an empty list, and then the model concludes with total confidence that nobody has discussed the competition. Make them separate branches, monkeypatch the HTTP client to produce each, and show the two different loop behaviours side by side.

**Stopping condition:** stall detection — if the same query is issued twice, or two consecutive searches return no URL not already in the knowledge file, stop and summarise.

### 5.3 Diganta — execution and submission

```python
run_experiment(
    code: str,                    # required
    dataset_ref: str,             # required
    timeout_s: int = 600,         # narrow type, hard cap 1800
    cv_folds: int = 5
)
```

Subprocess, `cwd` confined to `workspace/`, stdout/stderr captured, wall-clock timeout enforced. The script is required to print a parseable `CV_SCORE: <float>` line — parsing that is where failure mode 3 comes from.

```python
record_experiment_result(
    experiment_id: str,
    cv_score: float,
    fold_scores: list[float],
    notes: str
)
```

Appends to `workspace/experiments/leaderboard.jsonl`. Ungated.

**Gated action:** `submit_to_kaggle(submission_path, message)`. Irreversible for real reasons — a limited daily quota on a live competition, and a public record. The gate message must show: the file, its row count, the first three rows, the CV score of the model that produced it, and **how many submissions remain today**. If the row count doesn't match `sample_submission.csv`, refuse before even reaching the gate.

**Error branch — three distinct kinds, one branch each:**

| Fixture | Expected `kind` |
|---|---|
| script raising `ValueError` | `exec_failed` |
| `while True: pass` | `timeout` |
| script that runs clean but prints no `CV_SCORE:` line | `bad_input` |

A traceback must never be fed back as if it were a result. That is the whole of requirement 4.

**Stopping condition:** plateau detection over `leaderboard.jsonl` — stop when the best CV score has not improved by more than `min_delta` in the last N experiments. This is also the first working piece of KaggleCracker's improvement loop, so it is not throwaway work.

---

## 6. Testing — the part that's graded hardest

> *"Anyone can produce code now; what counts is knowing how it works, and being able to show how you found that out."*

### 6.1 The FakeModel harness — build this on Day 1, everyone uses it

A model stub that replays a pre-written sequence of tool calls. No API key, no cost, fully deterministic, and it reaches every branch of the loop:

```python
FakeModel([
    tool_call("profile_dataset", {"path": "ragged.csv", "checks": ["dtypes"]}),  # -> error branch
    tool_call("profile_dataset", {"path": "train.csv",  "checks": ["dtypes"]}),  # -> recovery
    final_answer("plan written"),
])
```

### 6.2 Minimum test set, per person

1. **Schema rejection** — an out-of-enum value or a missing required field is rejected by the validator *before* the tool body runs.
2. **Happy path** on real S6E7 data.
3. **One injected failure per error kind** in your slice, asserting the loop takes a different path.
4. **Gate approve and gate deny** — both asserting on the filesystem. After a denial, the target file's mtime *and* contents are unchanged, and no submission was spent.
5. **Stopping condition fires** — a FakeModel that never stops (hits the cap), and one that repeats itself (hits stall/plateau).
6. **One end-to-end run against the real model**, transcript committed to `runs/`.

### 6.3 The thing that will separate our writeups from everyone else's

**Break it on purpose and record what happened.**

Delete your error branch. Rerun the failing fixture. Save that transcript next to the fixed one. For Nikita it's the model confidently profiling a file it never read; for Shamil it's the model asserting no prior discussion exists because of a rate limit; for Diganta it's a traceback being parsed as a CV score.

Two transcripts and a diff is a paragraph nobody can dismiss as assistant-generated, because it is about *our* system's specific failure. Keep the failing logs. *"Here is the transcript where it did the wrong thing, here is the commit that fixed it"* beats *"I wrote 14 tests"* every time.

---

## 7. Git workflow — don't lose your commit

- Branch per person: `nikita/recon-tools`, `shamil/knowledge-tools`, `diganta/exec-tools`.
- **No squash-merge.** GitHub's squash rewrites authorship into a single commit and your submitted link then points at something that isn't yours. Use `--no-ff` merges or rebase-and-merge, both of which preserve individual authorship.
- Develop messily, then present **one clean, self-contained, well-described commit** as your graded commit. `fix typo` is not a submission.
- **Verify before submitting:** open your commit link in a private browser window. Check that the author is you, and that the diff visibly contains all four required elements.
- Everyone reviews the other two PRs. Assume you may be asked how the whole system works, not just your slice.

---

## 8. Timeline

| Day | Everyone | Nikita | Shamil | Diganta |
|---|---|---|---|---|
| 0 | Day-0 checklist (§1), repo created, Kaggle API working | | | |
| 1 | Mob: loop by hand, contracts, registry, gate skeleton, FakeModel, `metrics.py` | | | |
| 2 | | tool schemas + bodies | tool schemas + bodies | tool schemas + bodies |
| 3 | | unit tests, fixtures | unit tests, canned API responses | unit tests, 3 failure fixtures |
| 4 | | wire gate + error branch + stop condition into the loop | ditto | ditto |
| 5 | First full end-to-end run on S6E7. **First real submission** (gated, human present). | | | |
| 6 | Break-it-on-purpose runs, save transcripts, cross-review PRs | | | |
| 7 | Merge `--no-ff`, verify commit links, write one-pagers | | | |

Day 7 is a buffer as much as a work day. Something will not work on Day 6.

---

## 9. One-page writeup template

Four headers matching the four requirements, plus evidence. **One page maximum.**

1. **What I built** — my two tools, their schemas, the *when not to call* line in each description, and one sentence on why I constrained that particular parameter.
2. **The loop** — where my slice observes / reasons / acts / verifies, my stopping condition, and the run where it fired.
3. **The gate** — which action, why it is irreversible, what the denial path does. Include both transcripts.
4. **The error branch** — the failure I catch, why the naive version is dangerous, and the with/without comparison.
5. **How I tested it** — the FakeModel harness, the injected failures, and **the one thing I found that surprised me.**

The surprising finding is worth more than a complete feature list. Cut features from the page before you cut that.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| We burn the daily submission quota on a broken CSV | Dry-run mode default; row-count and column-name validation *before* the gate; gate message shows remaining quota |
| Everyone edits `core/` and we spend Day 5 on merge conflicts | `core/` frozen after Day 1; changes announced in the group chat before pushing |
| The agent overfits public CV and we learn nothing | Fixed CV folds, seeded, defined on Day 1 and never changed; log every experiment to `leaderboard.jsonl` |
| A slice isn't done by Day 6 | Each slice is independently submittable. A missing slice costs that person, not the group. Say so out loud on Day 1. |
| Someone's graded commit is missing one of the four requirements | The Day 7 checklist is: open your own commit diff and point at all four. Do it for each other. |
| Squash-merge destroys authorship | Repo setting: disable squash-merge on Day 0. |

---

## 11. Milestone 1 is done when

- [ ] Three commits exist, one per person, each containing two self-designed tools, a gate, an error branch, and a stopping condition.
- [ ] `pytest` is green, and every test in §6.2 exists for every slice.
- [ ] `runs/` contains at least: one successful end-to-end run, one gate-denial run, and one broken-on-purpose run per person.
- [ ] At least one real submission has been made to S6E7 through the gated tool, with the human approval visible in the transcript.
- [ ] Three one-pagers written, each with a surprising finding in the testing section.
