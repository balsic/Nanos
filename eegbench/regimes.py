"""Evaluation regimes: exactly how much target-subject data an arm is allowed to use.

This is the axis the phrase "zero-shot" hides, and it is worth naming precisely because
almost every published cross-subject number sits at one specific point on it without
saying so.

Ordered strictest to loosest:

``inductive``
    The target subject contributes **nothing**. Every statistic -- alignment reference,
    normalization mean and scale -- comes from the training fold. This is genuine trial
    #1: a model shipped to a new person who has not yet recorded anything. It is the only
    regime in which "zero-shot" is literally true.

``calib:N``
    The target contributes ``N`` **unlabelled** trials, and every per-subject statistic is
    computed from those ``N`` and no others. This is the deployable middle: a short
    eyes-open recording before the session starts, no labels, no supervision. Sweeping
    ``N`` produces the curve that decides whether calibration-free operation is a real
    capability or an artefact of having the whole test set in hand.

``transductive``
    The target's **entire** unlabelled test set defines its statistics. This is what
    Euclidean Alignment normally means in the literature and it is a legitimate, widely
    used protocol -- but it assumes every trial is available before any prediction is
    made, which no online system can do. Reported numbers should say so.

``supervised:N``
    ``N`` *labelled* target trials. Included for completeness and as an upper reference;
    it is no longer calibration-free and must never be compared against the others as
    though it were.

The budget applies to every per-subject statistic, not just alignment
---------------------------------------------------------------------
This is the trap the design exists to close. It is easy to budget the alignment reference
to ``N`` trials and then z-score the target using its full test set -- at which point the
run is transductive again, through the normalization rather than through the alignment,
and it still reports "calibration = 5 trials". :func:`resolve` therefore returns one
``fit_index`` per subject that *both* operations must use, so the budget cannot be honoured
in one place and leaked in another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eegbench.splits import Split
from eegbench.store import Cohort

__all__ = ["Regime", "parse_regime", "resolve", "REGIME_HELP"]

REGIME_HELP = (
    "inductive | calib:N | transductive | supervised:N  "
    "(strictest to loosest; see eegbench.regimes)"
)


@dataclass(frozen=True)
class Regime:
    kind: str            # inductive | calib | transductive | supervised
    n_calib: int | None = None

    def __str__(self) -> str:
        return self.kind if self.n_calib is None else f"{self.kind}:{self.n_calib}"

    @property
    def uses_target_data(self) -> bool:
        return self.kind != "inductive"

    @property
    def uses_target_labels(self) -> bool:
        return self.kind == "supervised"


def parse_regime(text: str) -> Regime:
    text = text.strip().lower()
    if text in ("inductive", "transductive"):
        return Regime(text)
    for kind in ("calib", "supervised"):
        if text.startswith(f"{kind}:"):
            n = text.split(":", 1)[1]
            if not n.isdigit() or int(n) < 1:
                raise ValueError(f"{kind} needs a positive trial count, got {text!r}")
            return Regime(kind, int(n))
    raise ValueError(f"unknown regime {text!r}. Expected: {REGIME_HELP}")


def resolve(cohort: Cohort, split: Split, regime: Regime, *,
            seed: int = 0) -> tuple[np.ndarray, dict]:
    """Which trials may be used to estimate each subject's own statistics.

    Returns ``(fit_index, info)`` where ``fit_index`` is a boolean mask over all trials.
    A subject-wise statistic (alignment reference, z-score mean/std) must be estimated
    from ``trials of that subject AND fit_index`` and from nothing else.

    Training subjects always contribute all their trials -- the budget is a statement
    about the *target*, not a restriction on training data.
    """
    n = cohort.X.shape[0]
    fit = np.zeros(n, dtype=bool)
    fit[split.train] = True
    fit[split.val] = True

    test_subjects = np.unique(cohort.subject[split.test])
    info: dict = {"regime": str(regime), "n_test_subjects": int(test_subjects.size)}

    if regime.kind == "inductive":
        # Nothing from the target. Downstream code must therefore fall back to a
        # source-derived statistic for these subjects; it may not quietly use their data.
        info["target_trials_used"] = 0
        return fit, info

    if regime.kind == "transductive":
        fit[split.test] = True
        info["target_trials_used"] = int(split.test.size)
        return fit, info

    # calib:N and supervised:N -- N trials per target subject, chosen without labels.
    rng = np.random.default_rng(seed)
    used = 0
    per_subject: dict[str, int] = {}
    for s in test_subjects:
        idx = split.test[cohort.subject[split.test] == s]
        take = min(regime.n_calib, idx.size)
        if take < regime.n_calib:
            # Silently taking fewer would make the x-axis of a calibration curve a
            # fiction for the subjects that could not supply the budget.
            raise ValueError(
                f"subject {cohort.subject_index[int(s)]} has only {idx.size} test trials, "
                f"fewer than the requested calibration budget of {regime.n_calib}. Lower "
                "the budget, or exclude the subject explicitly."
            )
        pick = idx[rng.permutation(idx.size)[:take]]
        fit[pick] = True
        used += take
        ds, native = cohort.subject_index[int(s)]
        per_subject[f"{ds}/s{native}"] = int(take)

    info["target_trials_used"] = int(used)
    info["target_trials_per_subject"] = per_subject
    # Recorded because a calibration curve is only interpretable against the size of the
    # set the statistic would otherwise have used.
    info["target_trials_available"] = int(split.test.size)
    return fit, info
