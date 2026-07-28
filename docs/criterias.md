# Session 8 assignment — criteria check against current architecture

Verdict: 3 of 6 criteria solid, 3 missing or weak.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Real channel | ❌ **Missing** | Only surface is the self-hosted FastAPI + React UI on localhost. No Telegram/Slack/Discord/Email. Not a "channel other people can write into", and there is no bot registration. The assignment does not list a local web UI as an equivalent surface. |
| 2 | Disposable identity | ❌ **Missing** | The agent runs on the personal `ANTHROPIC_API_KEY` and the personal `~/.kaggle/kaggle.json`. `submit_to_kaggle` would submit as the account owner, not as a bot. No fresh account or fresh key anywhere. |
| 3 | 2 non-message triggers | ⚠️ **Half** | Event trigger ✓: the data download reaching `state: done` fires the observer + EDA spawn (`ui/run_setup.py`), documented in `docs/milestone2/shamil.md` and tested. Schedule/heartbeat ✗: `monitor/clock.py` ticks on an interval but drives the offline judge only — it grades transcripts and never triggers an agent that answers anywhere. The observer is never re-fired automatically after setup (known gap). The assignment wants a webhook/event/threshold **AND** a schedule — the schedule side is effectively absent. |
| 4 | Silence branch | ✅ **Present** | An observer refresh that finds nothing new makes no LLM calls and produces no output; the journal line `batch_selected {count: 0, skipped_known: N}` is the written record that silence was a decision. Second instance: the EDA poll answers `{unchanged: true}`. Tested: "pulled exactly once across two refreshes". |
| 5 | Queue | ✅ **Present** | Reject-while-busy: mkdir lock + HTTP 409, the message is NOT stored, and a `spawn_pending` marker closes the spawn gap. The write-up states the choice and its cost (human must resend; the UI disables the input; stale-intent rationale for rejecting over queuing). Tested: 409 under a held lock with the consequence checked (nothing queued). Minor holes outside the criterion: recon runs have no lock, EDA turns have no pid file so they cannot be stopped. |
| 6 | Admin subagent | ⚠️ **Weak** | A written boundary exists (`run_bash` vs the sandboxed script runners; the stop endpoint kills a turn by pid while the agent cannot). But neither is a **subagent** — `run_bash` is a tool available to the same agents, and the stop endpoint is server code, not an agent. The critic/executor pair is HW2 coordination, not a privileged helper. The assignment asks for a privileged *agent* with capabilities the ordinary path lacks. |

## Testing criterion (weighted heaviest)

Strong for what exists:

- Trigger fired on purpose: `tests/ui/test_setup.py` fakes the Kaggle client and asserts the success path spawns exactly `[ui.run_observer, ui.run_eda]`.
- Silence branch asserted as a consequence: an already-seen notebook is pulled exactly once across two refreshes.
- Queue asserted as a consequence: 409 under a held lock, and nothing queued behind it.

But the two unbuilt items (channel, schedule) are also untestable until they exist.

## What to build to pass

1. **Telegram (or Discord) bot** — fresh bot account + token, reads chat, answers there. Wire it to the existing turn runners: they are already stateless subprocesses primed from `messages.jsonl`, so the channel client is mostly plumbing.
2. **Bot identity separation** — bot token in env, distinct from personal keys; the agent posts as the bot, never as the account owner.
3. **Schedule trigger** — a cron/heartbeat that fires an observer refresh (or a leaderboard check) periodically. The `monitor/clock.py` pattern already exists; repoint it at an agent instead of the judge. The silence branch is already ready for it: a scheduled refresh that finds nothing new is a recorded silence. Bonus: a threshold trigger is cheap — `plateau_detector` fires → the agent posts to the channel.
4. **Admin subagent** — e.g. admin-only chat commands (accepted from one configured user ID only) backed by a privileged registry: stop running turns, view submission quota, flip `dry_run`. The ordinary path gets none of these. Write the boundary down.

Items 3 and 4 mostly recombine existing pieces. Items 1 and 2 are new plumbing.
