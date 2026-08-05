# KaggleCracker

Autonomous Kaggle tabular competition agent: AIDE-style tree search over sandboxed ML solutions.
See `PLAN.md` for the full 7-day build plan and the reasoning behind each decision.

## Commands

```bash
uv sync                          # install (Python 3.12 pinned)
uv run kagglecracker preflight   # verify every external dependency
uv run kagglecracker init        # create the DB, seed the CV protocol
uv run kagglecracker download    # fetch + unzip competition data

# search
caffeinate -i uv run kagglecracker worker --max-nodes 20   # the loop
uv run kagglecracker worker --resume <run_id>              # continue a dead run
uv run kagglecracker node --action draft                   # exactly one node, by hand
uv run kagglecracker node --action draft --dry-run         # print the prompt only
uv run kagglecracker exec path/to/train.py                 # run a local file in the sandbox

uv run pytest                    # 126 tests, ~0.6s, no Docker and no API key
uv run pytest -m live            # real containers
uv run pytest -m eval            # live LLM calls — costs tokens
uv run ruff check kagglecracker/ tests/
```

## Architecture

Two processes, no scheduler. `worker` is the sole writer and runs one experiment at a time by virtue
of being one process (enforced by an advisory lock). `web` reads. SQLite in WAL mode so reads never
block the writer.

All ML runs inside the Docker sandbox — never on the host. sklearn/lightgbm/etc. are deliberately
NOT host dependencies; the host only orchestrates.

## Non-obvious things

- **kaggle-cli 2.2.4 flags differ from the online docs.** `submit` has no `--wait`/`--poll-interval`
  (poll `competitions submissions --csv` instead). `download` has no `--unzip`. Check `--help`
  against the installed binary before trusting a docs page.
- **Competition rules must be accepted in a browser** before the API will serve data. No API for it.
  `kaggle competitions files <comp>` is the cheapest check.
- **CV must be group-aware.** `PassengerId` is `gggg_pp`; Kaggle split train/test by party, and zero
  test groups appear in train. A plain KFold leaks groupmate labels. See the `protocols` row.
- **Every configured model needs a price entry** in `runtime/pricing.py` or `max_usd` silently
  computes $0.00 and never trips. Startup asserts this.
- **Set `reasoning_effort` explicitly.** Reasoning models default it high, and `max_tokens` covers
  reasoning as well as output. Measured: the default consumed all 32,000 tokens on reasoning and
  returned no code, at 11x the cost and time of `low`.
- **`engine/ranking.py` is the only place nodes are ranked.** It carries two filters that are easy to
  forget and invisible when forgotten: current protocol, and `status='ok'`. An ad-hoc
  `ORDER BY cv_score` elsewhere is a bug — it will happily rank nodes scored under a fold scheme you
  replaced for leaking.
- **The search policy is frozen for week 1.** Tuning it is the project's named rabbit hole. Both its
  fallbacks land on `draft` because draft needs no parent, which is what makes the policy total.
