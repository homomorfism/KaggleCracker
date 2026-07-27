"""The monitor's OWN clock: run the offline judge on a schedule, forever-ish.

judge.py grades whatever transcripts exist when it runs; this module is the
thing that makes it a background job rather than a manual step. It is still
completely outside the request/response loop — it imports the judge and a
sleep function, never the agents, the tools, or the model.

The interval loop has an explicit stopping condition (max_runs), the same
posture as every loop in this repo: an unbounded default is fine for a real
background job, but a test — or a demo — can pin the number of ticks and prove
the loop actually stops. sleep is injectable for the same reason: a test of the
schedule must not spend wall-clock time.

Run from the repo root:
    python monitor/clock.py --every 900            # grade every 15 minutes
    python monitor/clock.py --every 60 --max-runs 3  # three ticks, then stop
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor import judge


def run_on_clock(every_seconds, max_runs=None, run_once=judge.main,
                 sleep=time.sleep, output_fn=print):
    """Tick the judge every every_seconds; return the number of runs done.

    Sleeps BETWEEN runs, not before the first one, so the first grading pass
    happens immediately on start. With max_runs=None it runs until killed —
    that is the real background-job mode, and the return value is unreachable.
    """
    runs = 0
    while True:
        run_once()
        runs += 1
        if max_runs is not None and runs >= max_runs:
            return runs
        output_fn("monitor: sleeping %ds until the next grading pass" % every_seconds)
        sleep(every_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--every", type=int, default=900,
                        help="seconds between grading passes (default 900)")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="stop after this many passes (default: run until killed)")
    args = parser.parse_args(argv)
    run_on_clock(args.every, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
