"""A stopping condition for run_agent: recon is done when the plan is on disk.

plan_written builds a should_stop(state, messages) callable that run_agent
polls at the top of each step. Like the exec slice's plateau detector, it
ignores the loop state and the transcript and reads the durable artifact
instead — workspace/plans/<dataset>.plan.md — because "is recon finished" is
a question about whether the plan actually exists, not about what the model
last said it did.
"""

from tools.recon.paths import plan_path


def plan_written(dataset):
    """Return a should_stop callable that fires once the preprocessing plan
    for ``dataset`` exists. run_agent checks it before consulting the model,
    so no step is spent after the plan has landed."""

    def should_stop(state, messages):
        # state/messages unused on purpose: the signal is the file. The two
        # parameters exist only to match how run_agent invokes its stopping
        # condition.
        path = plan_path(dataset)
        if not path.exists():
            return None
        return "plan written: %s exists (%d bytes); recon is done" % (
            path, path.stat().st_size,
        )

    return should_stop
