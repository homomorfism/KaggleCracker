"""A stopping condition for run_agent: stop once the leaderboard has plateaued.

plateau_detector builds a should_stop(state, messages) callable that run_agent
polls at the top of each step. It ignores the loop state and the transcript and
reads the durable record instead — workspace/experiments/leaderboard.jsonl —
because "have we stopped making progress" is a question about the CV scores we
have actually recorded, not about what the model just said.

Direction is read live from experiments.HIGHER_IS_BETTER, so this file never
assumes higher-is-better on its own; flipping that one constant flips the whole
notion of "improvement" here too.
"""

import json

from tools.exec import experiments


def plateau_detector(window=3, min_delta=0.0005):
    """Return a should_stop callable that fires when the best recorded CV score
    has not improved by more than min_delta across the last ``window`` experiments.

    window is a patience: how many of the most recent experiments may show no
    improvement before we give up. min_delta is how much better counts as real
    improvement — a change smaller than this is noise, not progress.
    """

    def should_stop(state, messages):
        # run_agent calls this as should_stop(state, messages); both are unused
        # here — the signal is the recorded leaderboard, not the loop state or
        # the transcript. The two parameters exist only to match how run_agent
        # invokes its stopping condition.
        scores = _recorded_scores()

        # Insufficient history: we need at least one experiment *before* the last
        # window to have a baseline to measure the window's progress against.
        if len(scores) <= window:
            return None

        higher_is_better = experiments.HIGHER_IS_BETTER
        prior = scores[:-window]
        recent = scores[-window:]
        best_prior = max(prior) if higher_is_better else min(prior)

        # Did any of the last ``window`` experiments beat the prior best by more
        # than min_delta? Direction decides which way "beat" points.
        if higher_is_better:
            improved = any((s - best_prior) > min_delta for s in recent)
        else:
            improved = any((best_prior - s) > min_delta for s in recent)

        if improved:
            return None

        best_overall = max(scores) if higher_is_better else min(scores)
        return (
            "plateau: best CV score has not improved by more than %g over the "
            "last %d experiments (best so far %g)" % (min_delta, window, best_overall)
        )

    return should_stop


def _recorded_scores():
    """The cv_score of each recorded experiment, in recorded order.

    A missing leaderboard means no experiments have been recorded yet -> empty
    list. Only numeric scores are kept; a record without one is skipped rather
    than guessed at. The lines are trusted to be valid JSON because they are
    written by record_experiment_result — a corrupt line is a real problem to
    surface, not to silently swallow.
    """
    leaderboard = experiments._workspace_root() / "experiments" / "leaderboard.jsonl"
    if not leaderboard.exists():
        return []
    scores = []
    with leaderboard.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = rec.get("cv_score")
            # bool is a subclass of int; exclude it so a stray True/False in the
            # file is never treated as a 1/0 score.
            if isinstance(s, (int, float)) and not isinstance(s, bool):
                scores.append(s)
    return scores
