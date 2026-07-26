# Remaining work — closing out Milestone 1

Status as of 2026-07-26. All three slices (recon, knowledge, exec) are merged,
`python -m pytest -q` is green (182 passed), and break-it-on-purpose transcripts
exist for all three slices. What follows is the gap between the repo and the
"Milestone 1 is done when" checklist in `docs/PLAN.md` §11.

**Deadline pressure: S6E7 closes 2026-07-31 23:59 UTC — 5 days from now.**
The real submission (item 4 below) must happen before then or the strongest
evidence in the whole project disappears.

## What is NOT implemented / missing

| # | Gap | Where it's required |
|---|---|---|
| 1 | `core/metrics.py` (Balanced Accuracy + unit test vs hand-computed example) does not exist | PLAN §4e, Day 1 item |
| 2 | `docs/milestone1/diganta.md` one-pager missing (shamil.md and nikita.md exist) | PLAN §11 |
| 3 | No gate-denial run transcript in `runs/` | PLAN §11: "one gate-denial run" |
| 4 | No real submission to S6E7 through the gated tool; `workspace/submissions/` is empty | PLAN §11 |
| 5 | End-to-end runs against the real model exist only for the knowledge slice (`runs/knowledge_e2e_run.txt`); recon and exec slices have none | PLAN §6.2 item 6 |
| 6 | Day-0 checklist items still unchecked: target column details, data size, external-data rule, original source dataset, late-submission verification on the closed episode | PLAN §1 |
| 7 | `workspace/data/` is empty — S6E7 competition data never downloaded, so no full pipeline run has happened | PLAN §8 Day 5 |

## Work split

### Shamil

- [x] Day-0 checklist: external-data rule confirmed (public notebooks openly
  merge the source dataset; standard Playground rules), original source
  dataset identified and schema-verified:
  [`ziya07/college-student-health-behavior-dataset`](https://www.kaggle.com/datasets/ziya07/college-student-health-behavior-dataset),
  50k rows, same features + target. Recorded in `docs/PLAN.md` §1.
- [x] Target confirmed: `health_condition`, 3-class (`at-risk` 85.9% /
  `unhealthy` 8.4% / `fit` 5.8%). Recorded in `docs/PLAN.md` §1.
- [x] `core/metrics.py` — balanced accuracy, stdlib only, agreed core-freeze
  exception. Tests in `tests/core/test_metrics.py` include a hand-computed
  example; implementation fuzz-checked against sklearn (200 random trials,
  exact match).
- [x] Late-submission check — verified 2026-07-26: `sample_submission.csv`
  submitted late to closed `playground-series-s6e5`, scored `COMPLETE`
  (public/private 0.50000). S6E5 confirmed usable as the offline regression
  fixture. Gotcha for the writeup: rules must be accepted on the website while
  logged in as the API token's account, or every data/submit call 403s.

### Nikita

- [x] ~~Download S6E7 data into `workspace/data/`~~ — done 2026-07-26 as a side
  effect of Shamil's target-column check; sizes recorded in the Day-0 checklist.
- [ ] **Recon end-to-end run** against the real model; transcript committed to
  `runs/recon_e2e_run.txt` (mirror `e2e_knowledge_run.py`).
- [ ] **Gate-denial run transcript**: run the plan-overwrite gate, type
  something other than `yes`, save the transcript to `runs/` showing the
  denial fed back into the loop and the plan file unchanged.

### Diganta

- [ ] `docs/milestone1/diganta.md` one-pager — four requirement headers plus
  the surprising finding (template in PLAN §9). Material already exists:
  five failure fixtures, plateau stop, `with/without_branch.txt`.
- [ ] **Exec end-to-end run** against the real model; transcript to
  `runs/exec_e2e_run.txt`.
- [ ] **The real submission** — one gated `submit_to_kaggle` run on S6E7 with a
  human at the keyboard, approval visible in the transcript, committed to
  `runs/`. Depends on Nikita's data download and a completed experiment run;
  schedule for no later than 2026-07-29 to leave buffer before close.
- [ ] A second gate transcript for submit: one **denial** (quota preserved,
  nothing uploaded) alongside the approval, per PLAN §4c "test the denial path".

## Order of operations

1. Nikita downloads data (unblocks everything downstream).
2. Shamil confirms rules/target and lands `metrics.py` (after announcing the
   `core/` change).
3. Recon and exec e2e runs in parallel.
4. Diganta's real submission with human approval — hard deadline 2026-07-29.
5. Diganta's one-pager last; it can cite the submission transcript.

## Definition of done

Every checkbox in `docs/PLAN.md` §11 verifiable from the repo alone:
three authored commits, green pytest, `runs/` containing e2e + denial +
break-it transcripts for all three slices, one real gated submission, three
one-pagers.
