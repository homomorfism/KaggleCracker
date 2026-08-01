"""The only place nodes are ranked.

Every consumer — the search policy, the leaderboard, the approve screen — comes
through here. That is deliberate: ranking carries two filters that are easy to
forget and invisible when forgotten.

    nodes ──► WHERE protocol_id = <current>   ← retired protocols drop out
              AND   status = 'ok'             ← failures are not "best"
          ──► ORDER BY cv_score <direction>   ← lower_is_better exists too

Forget the protocol filter and the tree happily ranks nodes scored under a fold
scheme you already replaced because it was leaking. Nothing errors; the
leaderboard just quietly lies. An ad-hoc `ORDER BY cv_score` anywhere else in
the codebase is a bug, not a shortcut.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from kagglecracker.db.store import Node, Protocol


@dataclass(frozen=True, slots=True)
class RankedNode:
    """A scored node plus the fold statistics the human needs at approve time."""

    node: Node
    fold_mean: float
    fold_std: float
    fold_min: float

    @classmethod
    def of(cls, node: Node) -> RankedNode:
        folds = node.fold_scores or []
        if not folds:
            return cls(node, node.cv_score or 0.0, 0.0, node.cv_score or 0.0)
        mean = sum(folds) / len(folds)
        if len(folds) > 1:
            var = sum((f - mean) ** 2 for f in folds) / (len(folds) - 1)
            std = var**0.5
        else:
            std = 0.0
        return cls(node, mean, std, min(folds))


def _order(protocol: Protocol) -> str:
    return "DESC" if protocol.higher_is_better else "ASC"


def leaderboard(
    conn: sqlite3.Connection,
    protocol: Protocol,
    *,
    limit: int | None = None,
) -> list[RankedNode]:
    """Scored nodes under the current protocol, best first.

    Fold spread is surfaced alongside the score because the search takes many
    draws against one fixed fold split, so the top node is partly the luckiest
    draw. A human approving a submission should see the variance, not just the
    mean — the plan's mitigation for CV overfitting is exactly this visibility.
    """
    sql = (
        "SELECT * FROM nodes "
        "WHERE protocol_id = ? AND status = 'ok' AND cv_score IS NOT NULL "
        f"ORDER BY cv_score {_order(protocol)}, id ASC"
    )
    params: tuple = (protocol.id,)
    if limit is not None:
        sql += " LIMIT ?"
        params = (protocol.id, limit)
    return [RankedNode.of(Node.from_row(r)) for r in conn.execute(sql, params)]


def best_node(
    conn: sqlite3.Connection,
    protocol: Protocol,
    *,
    max_depth: int | None = None,
) -> Node | None:
    """The node to improve.

    `max_depth` exists for the policy's depth cap: at the cap we still want the
    best *improvable* node, not the best node overall. Without it the policy
    would keep selecting a leaf it is not allowed to extend and stall.
    """
    sql = (
        "SELECT * FROM nodes "
        "WHERE protocol_id = ? AND status = 'ok' AND cv_score IS NOT NULL"
    )
    params: list = [protocol.id]
    if max_depth is not None:
        sql += " AND depth < ?"
        params.append(max_depth)
    # `id ASC` as the tie-break keeps selection deterministic: with two equal
    # scores the older node wins every time, so a replayed run picks the same
    # parent and the search stays reproducible.
    sql += f" ORDER BY cv_score {_order(protocol)}, id ASC LIMIT 1"

    row = conn.execute(sql, tuple(params)).fetchone()
    return Node.from_row(row) if row else None


def most_recent_failure(
    conn: sqlite3.Connection,
    protocol: Protocol,
    *,
    max_depth: int | None = None,
) -> Node | None:
    """The newest node the debug action could plausibly repair.

    `infra_error` and `truncated` are excluded on purpose. Neither is a defect
    in the generated code — one is the environment failing, the other is our own
    max_tokens setting — so handing either to the debug action burns a node
    asking the model to fix something it did not do and cannot reach.
    """
    sql = (
        "SELECT * FROM nodes "
        "WHERE protocol_id = ? AND status IN ('error', 'timeout', 'contract_violation')"
    )
    params: list = [protocol.id]
    if max_depth is not None:
        sql += " AND depth < ?"
        params.append(max_depth)
    sql += " ORDER BY id DESC LIMIT 1"

    row = conn.execute(sql, tuple(params)).fetchone()
    return Node.from_row(row) if row else None
