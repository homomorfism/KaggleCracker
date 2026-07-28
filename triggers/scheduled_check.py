"""A SCHEDULE trigger: wake on a clock, look at the leaderboard, speak only on news.

This is the third way a run can start. HW1's runs started because a human typed
something; the offline monitor grades after the fact; this one starts because
time passed. Nobody asked it anything — so the interesting branch is the one
where it decides to stay quiet.

Each tick does the same four things:

    read  the recorded CV scores (workspace/experiments/leaderboard.jsonl)
    compare the current best against the persisted high-water mark
    act   post to the channel on a new best, or stay SILENT
    write down what it decided, so silence leaves a trace

The silence branch journals its reason instead of posting. That reason is
computed HERE and not read back off the agent loop: core.loop.run_agent calls
should_stop and then returns None, so a stopping condition's reason string never
reaches the caller. A trigger that must explain why it said nothing cannot rely
on a value the loop throws away.

Shape is copied from monitor/clock.py on purpose: sleep between ticks, an
explicit max_ticks stopping condition, and every external thing injected, so a
test can pin the tick count with a fake sleep and spend no wall-clock time.

This module observes and posts. It never imports the tool registry, the gate, or
submit_to_kaggle — nothing here can spend a live submission slot.

Run from the repo root:
    python triggers/scheduled_check.py --every 900              # look every 15 min
    python triggers/scheduled_check.py --every 60 --max-ticks 3  # three ticks, then stop
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hw3_paths import hw3_dir as _hw3_dir
from tools.exec import experiments
# The detector's reader, not a copy of it. Both the stopping condition and this
# trigger have to agree on what counts as a recorded score (missing file -> none,
# non-numeric skipped, bool never read as 1/0); a second copy of those rules
# would eventually drift and the two would disagree about the same file.
from tools.exec.stopping import _recorded_scores


def post_to_channel(text, output_fn=print):
    """Stub for the channel bot: append the message to runs/hw3/channel_out.jsonl.

    A file, not a network call, so the whole trigger is testable before the bot
    exists — and so a test can assert on what would have been said rather than on
    a mock's call count. Swapping this for a real post later changes this
    function only.
    """
    path = _hw3_dir() / "channel_out.jsonl"
    _append_json(path, {"text": text})
    output_fn("scheduled_check: posted -> %s" % text)
    return path


def _append_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _read_last_seen():
    """The persisted high-water mark, or None if we have never posted one.

    A missing file is the ordinary first-run case, not an error. A corrupt file
    is a different thing entirely: this module is the only writer, so unreadable
    JSON here means something is actually wrong and json.loads should raise
    rather than have us quietly restart from zero and re-announce an old score.
    """
    path = _hw3_dir() / "last_seen.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("best")


def _write_last_seen(best):
    path = _hw3_dir() / "last_seen.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"best": best}) + "\n")


def _best(scores):
    """Best recorded score, or None if nothing numeric has been recorded.

    Direction is read live from experiments.HIGHER_IS_BETTER, the same as the
    plateau detector, so flipping that one constant flips what "best" means here
    too.
    """
    if not scores:
        return None
    return max(scores) if experiments.HIGHER_IS_BETTER else min(scores)


def _beats(candidate, baseline):
    """Is candidate a genuinely new best against the persisted baseline?

    No baseline means nothing has ever been announced, so the first recorded
    score is news. Ties are NOT news: re-posting an equal score every tick is
    exactly the noise this trigger exists to avoid.
    """
    if baseline is None:
        return True
    return candidate > baseline if experiments.HIGHER_IS_BETTER else candidate < baseline


def check_once(tick, post=post_to_channel, output_fn=print):
    """One tick: post on a new best, otherwise stay silent and journal why.

    Returns the decision record {"tick", "decision", "reason", "best_seen"};
    "decision" is "posted" or "silent". The silent branch is the one that writes
    to runs/hw3/scheduled_journal.jsonl — a post already leaves its own trace in
    channel_out.jsonl, but silence leaves none by definition, so the journal is
    the only thing that makes a non-event auditable afterwards.
    """
    best_now = _best(_recorded_scores())
    last_seen = _read_last_seen()

    if best_now is not None and _beats(best_now, last_seen):
        if last_seen is None:
            text = "new best CV %g (first recorded score)" % best_now
        else:
            text = "new best CV %g (previous best %g)" % (best_now, last_seen)
        post(text, output_fn=output_fn)
        # Only advance the high-water mark after the post: if posting raises, the
        # next tick still sees the old mark and will try to announce it again,
        # which is the failure we want (a repeat) rather than a lost result.
        _write_last_seen(best_now)
        return {"tick": tick, "decision": "posted", "reason": text, "best_seen": best_now}

    # SILENT. Two ways to get here, and they are not the same fact: nothing has
    # been recorded at all, or something has and it did not beat the mark.
    if best_now is None:
        reason = "no CV scores recorded yet"
        best_seen = last_seen
    else:
        best_seen = last_seen
        reason = "no CV improvement over best %g" % best_seen
    record = {"tick": tick, "decision": "silent", "reason": reason, "best_seen": best_seen}
    _append_json(_hw3_dir() / "scheduled_journal.jsonl", record)
    output_fn("scheduled_check: silent (%s)" % reason)
    return record


def run_on_schedule(every_seconds, max_ticks=None, tick=check_once,
                    sleep=time.sleep, output_fn=print):
    """Tick every every_seconds; return the number of ticks done.

    Sleeps BETWEEN ticks, not before the first, so a fresh start looks at the
    leaderboard immediately. max_ticks is the explicit stopping condition — None
    is the real background-job mode where the return is unreachable, and a test
    pins a small number and passes a fake sleep so no wall-clock time is spent.
    """
    ticks = 0
    while True:
        ticks += 1
        tick(ticks, output_fn=output_fn)
        if max_ticks is not None and ticks >= max_ticks:
            return ticks
        output_fn("scheduled_check: sleeping %ds until the next look" % every_seconds)
        sleep(every_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--every", type=int, default=900,
                        help="seconds between looks at the leaderboard (default 900)")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="stop after this many ticks (default: run until killed)")
    args = parser.parse_args(argv)
    run_on_schedule(args.every, max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
