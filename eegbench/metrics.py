"""Metrics, implemented directly so the benchmark does not depend on sklearn's versions.

``cohen_kappa`` is reported alongside accuracy throughout. On a 2-class benchmark that may
look redundant, but it is not: pooled cohorts have unequal class counts per subject, and a
degenerate model that predicts one class scores well above chance in accuracy while kappa
correctly reports ~0. It is the cheapest available guard against reading a collapsed
model as a working one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["accuracy", "confusion_matrix", "cohen_kappa", "macro_f1", "chance_band",
           "balanced_accuracy", "roc_auc"]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if y_true.size else float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int | None = None) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    k = n_classes or int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        # Every prediction and every label in one class: kappa is undefined, and
        # returning 0.0 would misreport a degenerate case as merely chance-level.
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


def chance_band(n_trials: int, n_classes: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided binomial band around chance, by the normal approximation.

    A result inside this band is not evidence of decoding, and on the small per-subject
    test sets a cross-subject fold produces the band is wider than people expect -- at 200
    trials and 2 classes it reaches ~57%, which is above several published "above chance"
    per-subject figures.
    """
    from math import sqrt
    p = 1.0 / n_classes
    if n_trials <= 0:
        return (float("nan"), float("nan"))
    z = {0.05: 1.959964, 0.01: 2.575829, 0.10: 1.644854}.get(alpha, 1.959964)
    half = z * sqrt(p * (1 - p) / n_trials)
    return (max(0.0, p - half), min(1.0, p + half))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall. The right headline score for an imbalanced paradigm.

    P300 is roughly 1:5 target:non-target, so a model that predicts "NonTarget" for every
    epoch scores ~83% plain accuracy while decoding nothing. Balanced accuracy puts that
    model at 50%, which is what it is.
    """
    cm = confusion_matrix(y_true, y_pred)
    recalls = []
    for c in range(cm.shape[0]):
        n = cm[c, :].sum()
        if n:
            recalls.append(cm[c, c] / n)
    return float(np.mean(recalls)) if recalls else float("nan")


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Binary ROC AUC from decision scores, computed by rank (no sklearn dependency).

    Reported alongside balanced accuracy for P300 because the operating point matters:
    a speller integrates evidence over repetitions, so ranking quality is closer to the
    quantity of interest than any single-threshold decision is.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = (y_true == 1), (y_true == 0)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties, or AUC is biased when many scores coincide.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
