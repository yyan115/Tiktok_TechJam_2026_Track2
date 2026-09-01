from __future__ import annotations

import collections
import math
from collections.abc import Iterable

import numpy as np


class MetricError(ValueError):
    pass


def _as_arrays(
    user_ids: Iterable[int], labels: Iterable[int], scores: Iterable[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users = np.asarray(user_ids)
    y = np.asarray(labels)
    pred = np.asarray(scores)
    if users.ndim != 1 or y.ndim != 1 or pred.ndim != 1:
        raise MetricError("user_ids, labels and scores must be one-dimensional")
    if not (len(users) == len(y) == len(pred)):
        raise MetricError("user_ids, labels and scores must have equal lengths")
    if len(y) == 0:
        raise MetricError("evaluation input cannot be empty")
    if not np.all((y == 0) | (y == 1)):
        raise MetricError("long_view labels must be binary")
    if not np.issubdtype(pred.dtype, np.number):
        raise MetricError("scores must be numeric")
    pred = pred.astype(np.float64, copy=False)
    if not np.all(np.isfinite(pred)):
        raise MetricError("scores contain NaN or infinity")
    return users, y.astype(np.int8, copy=False), pred


def _auc_exact(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    npos = int(sorted_labels.sum())
    nneg = len(sorted_labels) - npos
    if npos == 0 or nneg == 0:
        return 0.5

    changes = np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1
    bounds = np.concatenate(([0], changes, [len(sorted_scores)]))
    positive_rank_sum = 0.0
    for left, right in zip(bounds[:-1], bounds[1:]):
        average_rank = (left + right + 1.0) / 2.0
        positive_rank_sum += average_rank * int(sorted_labels[left:right].sum())
    return (positive_rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)


def evaluate(
    user_ids: Iterable[int], labels: Iterable[int], scores: Iterable[float], k: int = 5
) -> dict[str, float | int]:
    """Exact organizer metric semantics with a NumPy implementation.

    GAUC excludes all-negative/all-positive users and weights remaining user AUCs
    by positive count. nDCG includes every user; all-negative users contribute zero.
    Stable descending score order preserves the organizer's tie behaviour.
    """

    users, y, pred = _as_arrays(user_ids, labels, scores)
    if k <= 0:
        raise MetricError("k must be positive")

    group_order = np.argsort(users, kind="stable")
    grouped_users = users[group_order]
    boundaries = np.concatenate(
        ([0], np.flatnonzero(grouped_users[1:] != grouped_users[:-1]) + 1, [len(users)])
    )
    discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    gauc_numerator = 0.0
    gauc_denominator = 0
    ndcg_sum = 0.0

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = group_order[left:right]
        group_y = y[idx]
        group_scores = pred[idx]
        positives = int(group_y.sum())
        if 0 < positives < len(group_y):
            gauc_numerator += positives * _auc_exact(group_y, group_scores)
            gauc_denominator += positives

        if positives:
            ranked = np.argsort(-group_scores, kind="stable")[:k]
            dcg = float((group_y[ranked] * discounts[: len(ranked)]).sum())
            idcg = float(discounts[: min(positives, k)].sum())
            ndcg_sum += dcg / idcg

    users_count = len(boundaries) - 1
    gauc = gauc_numerator / gauc_denominator if gauc_denominator else 0.5
    ndcg = ndcg_sum / users_count
    return {
        "GAUC": gauc,
        f"nDCG@{k}": ndcg,
        "primary": (gauc + ndcg) / 2.0,
        "users": users_count,
        "rows": len(y),
    }


# Kept only as a regression oracle. This is the supplied organizer implementation.
def evaluate_reference(
    user_ids: Iterable[int], labels: Iterable[int], scores: Iterable[float], k: int = 5
) -> dict[str, float | int]:
    reference_users = list(user_ids)
    reference_labels_all = list(labels)
    reference_scores_all = list(scores)

    def auc(reference_labels: list[int], reference_scores: list[float]) -> float:
        pairs = sorted(zip(reference_scores, reference_labels))
        ranks = [0.0] * len(pairs)
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for position in range(i, j + 1):
                ranks[position] = average
            i = j + 1
        npos = sum(label for _, label in pairs)
        nneg = len(pairs) - npos
        if npos == 0 or nneg == 0:
            return 0.5
        rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
        return (rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)

    def ndcg_at_k(reference_labels: list[int]) -> float:
        discounts = [math.log2(i + 2) for i in range(k)]
        dcg = sum(
            ((2**target) - 1) / discounts[i]
            for i, target in enumerate(reference_labels[:k])
        )
        ideal = sorted(reference_labels, reverse=True)[:k]
        idcg = sum(((2**target) - 1) / discounts[i] for i, target in enumerate(ideal))
        return 0.0 if idcg == 0 else dcg / idcg

    grouped: dict[int, list[tuple[float, int]]] = collections.defaultdict(list)
    for user, target, score in zip(reference_users, reference_labels_all, reference_scores_all):
        grouped[user].append((float(score), int(target)))
    gauc_numerator = 0.0
    gauc_denominator = 0.0
    ndcg_values = []
    for rows in grouped.values():
        rows.sort(key=lambda pair: -pair[0])
        group_labels = [target for _, target in rows]
        positives = sum(group_labels)
        if 0 < positives < len(group_labels):
            gauc_numerator += positives * auc(group_labels, [score for score, _ in rows])
            gauc_denominator += positives
        ndcg_values.append(ndcg_at_k(group_labels))
    gauc = gauc_numerator / gauc_denominator if gauc_denominator else 0.5
    ndcg = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0
    return {
        "GAUC": gauc,
        f"nDCG@{k}": ndcg,
        "primary": (gauc + ndcg) / 2.0,
        "users": len(grouped),
        "rows": len(reference_labels_all),
    }
