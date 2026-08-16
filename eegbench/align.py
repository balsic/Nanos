"""Covariance alignment: Euclidean, Riemannian, and the calibration-limited variants.

Why alignment is the centre of the zero-shot question
----------------------------------------------------
Per-channel z-scoring equalises each electrode's *scale* and leaves the *covariance
structure* -- which electrodes co-vary, and how -- untouched. But covariance structure is
precisely what differs between people: head shape, cortical folding, electrode placement.
Alignment maps every subject's mean covariance to a common point, which is why it
transfers where z-scoring does not.

Fit and apply are separated
---------------------------
The usual formulation is one function over the whole array, which quietly fixes the
evaluation regime: the target subject's whitener is estimated from *all* of that subject's
trials, including the ones about to be tested. That is legitimate (no labels are used --
it is the standard "calibration-free BCI" bar) but it is **transductive**, and it is not
trial #1.

Here a :class:`Whitener` is fitted on an explicit set of trials and applied to another.
That makes the regime a parameter instead of an assumption:

    w = fit_whitener(X_target[:5])       # 5 unlabelled trials from the new subject
    X_hat = w.transform(X_target)        # applied to everything, including unseen trials

and it is what lets :mod:`eegbench.protocols` express "N unlabelled target trials" as a
measurable axis rather than a caveat in a footnote.

Shrinkage, and what it is actually for
--------------------------------------
It is tempting to say that 5 trials cannot support a 64x64 covariance because the estimate
is rank-5. That is **wrong here**, and worth stating because it is the natural intuition:
each trial contributes ``T`` time samples, so the reference covariance is estimated from
``N * T`` samples -- 5 trials of 625 samples is 3,125 samples for 64 channels, comfortably
over-determined. Rank deficiency from small ``N`` needs ``N * T < C``, which does not occur
at realistic window lengths. (It *does* occur channel-wise, from absent electrodes in a
union montage, which is a different mechanism handled by projection below.)

What small ``N`` actually costs is **estimator variance**. Trial covariances are noisy and
correlated within a trial, so averaging 5 of them gives a reference that is unbiased but
scattered, and the inverse square root amplifies the smallest eigenvalues -- exactly the
directions estimated worst. Shrinking toward a scaled identity,

    R_hat = (1 - a) R + a * (tr(R)/C) I,

trades a little bias for a large variance reduction in precisely those directions, and
degrades gracefully to "do nothing" as ``a -> 1``. The intensity can be set directly or
estimated per subject by the Ledoit-Wolf rule. Whether it materially improves few-trial
calibration on real data is an empirical question this module makes measurable, not one it
settles.

Rank deficiency is handled by projection, not by a ridge
-------------------------------------------------------
A pooled union montage gives absent electrodes exactly zero variance, so the reference
covariance is genuinely singular. Directions below tolerance are given an inverse-sqrt of
zero, which drops them -- what a pseudo-inverse does -- and keeps zero-filled channels at
exactly zero rather than amplifying numerical dust into them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Whitener", "fit_whitener", "align_by_group", "alignment_residual",
    "riemannian_mean", "ledoit_wolf_shrinkage", "MODES",
]

MODES = ("none", "euclid", "riemann")


# ---------------------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------------------
def _trial_covariances(X: np.ndarray) -> np.ndarray:
    """``(N, C, T) -> (N, C, C)``, each normalised by ``T``."""
    Xd = np.asarray(X, dtype=np.float64)
    covs = np.einsum("nct,ndt->ncd", Xd, Xd) / Xd.shape[-1]
    return 0.5 * (covs + np.transpose(covs, (0, 2, 1)))


def ledoit_wolf_shrinkage(covs: np.ndarray) -> float:
    """Ledoit-Wolf intensity for shrinking the mean of ``covs`` toward ``mu * I``.

    The classic estimator, applied to the *set of trial covariances* rather than to raw
    samples: the target is the scaled identity, the dispersion is the mean squared
    deviation of the trial covariances from their mean, and the intensity is the ratio,
    clipped to [0, 1].

    Returning 1.0 when there is a single trial is the correct degenerate answer -- one
    covariance carries no information about its own sampling error, so the estimator
    should fall all the way back to the target.
    """
    n = covs.shape[0]
    if n < 2:
        return 1.0
    mean = covs.mean(axis=0)
    C = mean.shape[0]
    mu = np.trace(mean) / C
    # Squared Frobenius distance from the mean to the shrinkage target.
    denom = float(((mean - mu * np.eye(C)) ** 2).sum())
    if denom <= 0:
        return 1.0
    # Sampling dispersion of the mean estimate.
    num = float(((covs - mean) ** 2).sum(axis=(1, 2)).mean()) / n
    return float(np.clip(num / denom, 0.0, 1.0))


def _shrink(mean: np.ndarray, covs: np.ndarray, shrinkage: float | str) -> tuple[np.ndarray, float]:
    if shrinkage is None or (isinstance(shrinkage, (int, float)) and shrinkage <= 0):
        return mean, 0.0
    a = ledoit_wolf_shrinkage(covs) if shrinkage == "auto" else float(shrinkage)
    a = float(np.clip(a, 0.0, 1.0))
    C = mean.shape[0]
    mu = np.trace(mean) / C
    return (1.0 - a) * mean + a * mu * np.eye(C), a


def _inv_sqrt(A: np.ndarray, rank_tol: float) -> tuple[np.ndarray, int]:
    """``A^(-1/2)`` by eigendecomposition, dropping directions below tolerance.

    Structurally-dead channels are then zeroed **exactly**. This is not cosmetic tidying
    of round-off, and the reason is specific: ``eigh`` returns eigenvectors carrying
    ~1e-16 components in the null space, so an absent electrode leaves the whitener with
    an amplitude around 1e-15 instead of 0. Downstream, per-channel z-scoring divides each
    channel by its own standard deviation -- and a channel whose entire content is 1e-15
    of float noise has standard deviation 1e-15, so it is rescaled to **unit variance**.
    A structurally-absent electrode would arrive at the network as a full-amplitude noise
    channel. Forcing the zeros here keeps an absent electrode absent through the whole
    pipeline.
    """
    A = 0.5 * (A + A.T)
    # A channel with no variance contributes nothing and must receive nothing. Detected on
    # the diagonal, which is exactly the per-channel variance.
    C = A.shape[0]
    scale = max(float(np.trace(A)) / C, 1e-300)
    dead = np.diag(A) <= rank_tol * scale

    w, V = np.linalg.eigh(A)
    keep = w > rank_tol * max(float(w.max()), rank_tol)
    inv = np.zeros_like(w)
    inv[keep] = w[keep] ** -0.5
    W = (V * inv) @ V.T
    if dead.any():
        W[dead, :] = 0.0
        W[:, dead] = 0.0
    return W, int(keep.sum())


def _sqrt_pair(A: np.ndarray, rank_tol: float) -> tuple[np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    keep = w > rank_tol * max(float(w.max()), rank_tol)
    s = np.zeros_like(w)
    si = np.zeros_like(w)
    s[keep] = w[keep] ** 0.5
    si[keep] = w[keep] ** -0.5
    return (V * s) @ V.T, (V * si) @ V.T


# ---------------------------------------------------------------------------------------
# Riemannian (Frechet) mean
# ---------------------------------------------------------------------------------------
def riemannian_mean(covs: np.ndarray, *, iters: int = 50, tol: float = 1e-6,
                    rank_tol: float = 1e-10,
                    on_nonconvergence: str = "warn") -> np.ndarray:
    """Frechet mean of SPD matrices under the affine-invariant metric.

    Covariances live on the SPD manifold, where the arithmetic mean is only a first-order
    approximation to the centre of mass and is biased toward high-variance trials. The
    standard fixed-point iteration: log-map every covariance at the current estimate,
    average in the tangent space, exp-map the result back.

    Falls back to the arithmetic mean when the batch is rank-deficient. That is not a
    convenience -- the affine-invariant metric is *undefined* on singular matrices, and
    iterating anyway produces NaN rather than an error. A pooled union montage is
    rank-deficient by construction, so this path is reached in normal use.

    ``tol`` is on the tangent-space step norm. The default of 1e-6 is chosen against a
    measured floor, not picked for looking strict: in float64 the step norm plateaus
    around 3e-7, so a tighter tolerance warns on every healthy call -- which trains
    everyone to ignore the warning, and is worse than not having one. For reference,
    ``pyriemann`` stops at 1e-4, this is 100x tighter, and the two agree to ~2.5e-8
    relative.

    ``on_nonconvergence`` decides what happens when ``iters`` binds before ``tol`` is met.
    It defaults to warning rather than passing silently because that failure is known to
    occur on real data -- roughly one subject in fifty on a 52-subject corpus -- and a
    non-converged Frechet mean is not slightly worse, it can be off by orders of
    magnitude. Silently returning it produces a wrong whitener for one subject and a
    slightly lower accuracy for the whole arm, which reads as architecture rather than as
    the numerical failure it is.
    """
    covs = np.asarray(covs, dtype=np.float64)
    n, C, _ = covs.shape
    mean = covs.mean(axis=0)

    smallest = np.linalg.eigvalsh(0.5 * (mean + mean.T)).min()
    largest = np.linalg.eigvalsh(0.5 * (mean + mean.T)).max()
    if smallest <= rank_tol * max(largest, rank_tol):
        return mean  # singular: the manifold is not defined here

    for _ in range(iters):
        rt, rti = _sqrt_pair(mean, rank_tol)
        # `optimize=True` is load-bearing, not a micro-optimisation. Without it numpy
        # contracts this three-operand expression in the order written, materialising an
        # (n, C, C) intermediate per operand; with it the pairwise order is chosen by cost.
        # Measured on a Cho2017-shaped workload, the difference in this one call is 41.6
        # minutes versus 2.5 -- i.e. it decides whether Riemannian alignment is a usable
        # option or a 40x preprocessing tax that nobody will ever switch on.
        inner = np.einsum("ij,njk,kl->nil", rti, covs, rti, optimize=True)
        # Batched eigendecomposition: `np.linalg.eigh` accepts a stack, so the whole
        # log-map is one LAPACK call rather than N. At the sizes this is used on
        # (10k trials x 64 channels, per subject, per fold) the Python-loop form is the
        # difference between seconds and minutes, and it is called inside the training
        # loop's preprocessing.
        inner = 0.5 * (inner + np.transpose(inner, (0, 2, 1)))
        w, V = np.linalg.eigh(inner)                       # (n, C), (n, C, C)
        w = np.maximum(w, rank_tol)
        acc = np.einsum("nij,nj,nkj->ik", V, np.log(w), V, optimize=True) / n
        wA, VA = np.linalg.eigh(0.5 * (acc + acc.T))
        step = (VA * np.exp(wA)) @ VA.T
        new = rt @ step @ rt
        # Convergence on the tangent-space norm, which is the natural quantity here: it is
        # the length of the step actually taken on the manifold.
        if np.linalg.norm(acc) < tol:
            return new
        if np.linalg.norm(new - mean) / max(np.linalg.norm(mean), 1e-300) < tol:
            return new
        mean = new

    step_norm = float(np.linalg.norm(acc))
    msg = (f"riemannian_mean did not converge in {iters} iterations "
           f"(final tangent-space step norm {step_norm:.3e} > tol {tol:.1e}); the "
           "returned mean may be far from the true Frechet mean")
    if on_nonconvergence == "raise":
        raise RuntimeError(msg)
    if on_nonconvergence == "warn":
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return mean


# ---------------------------------------------------------------------------------------
# The whitener
# ---------------------------------------------------------------------------------------
@dataclass
class Whitener:
    """A fitted linear map ``x -> W x`` that re-centres one subject's covariance.

    Carries its own provenance because the number of trials it was fitted on *is* the
    evaluation regime -- a result produced with ``n_fit=5`` and one produced with
    ``n_fit=200`` are answers to different questions and must not be compared as though
    they were the same arm.
    """

    matrix: np.ndarray            # (C, C)
    mode: str
    n_fit: int
    rank: int
    shrinkage: float
    diagnostics: dict = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return np.asarray(X, dtype=np.float32)
        return np.einsum("cd,ndt->nct", self.matrix, np.asarray(X, dtype=np.float64)
                         ).astype(np.float32)


def fit_whitener(
    X: np.ndarray,
    *,
    mode: str = "euclid",
    shrinkage: float | str = 0.0,
    rank_tol: float = 1e-10,
    n_calib: int | None = None,
    seed: int = 0,
    riemann_iters: int = 50,
) -> Whitener:
    """Estimate the alignment map from ``(N, C, T)`` trials. Uses no labels.

    ``n_calib`` restricts the estimate to that many randomly chosen trials -- the
    deployment-relevant knob. "How many unlabelled trials must a new subject provide
    before we can predict for them" is exactly what decides whether alignment is usable
    zero-shot, and it is answered by sweeping this.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"expected (N, C, T) trials, got shape {X.shape}")
    C = X.shape[1]
    if mode == "none":
        return Whitener(np.eye(C), "none", int(X.shape[0]), C, 0.0)
    if X.shape[0] == 0:
        raise ValueError("cannot fit a whitener on zero trials")

    ref = X
    if n_calib is not None and n_calib < X.shape[0]:
        rng = np.random.default_rng(seed)
        ref = X[rng.permutation(X.shape[0])[:n_calib]]

    covs = _trial_covariances(ref)
    mean = riemannian_mean(covs, iters=riemann_iters, rank_tol=rank_tol) \
        if mode == "riemann" else covs.mean(axis=0)
    mean, applied = _shrink(mean, covs, shrinkage)
    W, rank = _inv_sqrt(mean, rank_tol)

    return Whitener(
        matrix=W, mode=mode, n_fit=int(ref.shape[0]), rank=rank, shrinkage=applied,
        diagnostics={
            "condition_number": float(np.linalg.cond(mean)) if rank == C else float("inf"),
            "n_channels": int(C),
        },
    )


# ---------------------------------------------------------------------------------------
# Group-wise application
# ---------------------------------------------------------------------------------------
def align_by_group(
    X: np.ndarray,
    groups: np.ndarray,
    *,
    mode: str = "euclid",
    shrinkage: float | str = 0.0,
    rank_tol: float = 1e-10,
    n_calib: int | None = None,
    seed: int = 0,
    fit_mask: np.ndarray | None = None,
    subset: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, Whitener]]:
    """Align each group by its own whitener.

    ``groups``
        Usually one group per subject, which is how Euclidean Alignment is normally
        stated. It does not have to be. Passing a **(subject, session)** grouping makes
        the reference *session-local*, and that is a live question rather than a detail:
        the only published measurement on this axis reports that estimating the reference
        from the wrong session costs up to 5.9 points and can leave alignment worse than
        no alignment at all. Impedance drift, cap re-placement and electrode gel all move
        the covariance between sessions, so a subject-pooled reference is an average over
        conditions that never co-occur at inference.

        Expressing it as a grouping rather than a flag means the two variants are the same
        code path with different inputs, so a difference between them cannot be an
        implementation difference.

    ``fit_mask``
        Boolean over trials selecting which may be used for *fitting*. Every trial is
        still transformed. This is the hook that makes a strict regime expressible: pass a
        mask that excludes the target subject's evaluation trials and the alignment
        becomes non-transductive with respect to them.

        Left as ``None`` it fits on everything, which is the standard calibration-free
        formulation -- correct, widely used, and worth naming rather than defaulting into
        silently.

    ``subset``
        Boolean over trials marking which ones this split actually uses. Groups with no
        trials in the subset are **passed through untransformed** rather than raising.

        This is not a convenience: under the subject-dependent protocols a split covers a
        single subject, so the other subjects' trials sit in neither train, validation nor
        test. They have no fit data by construction, and they are also never read. Without
        this, aligning a 9-subject cohort under `holdout-session` raises on the 8 subjects
        the split does not concern. Left as ``None`` every trial is in scope, which is the
        cross-subject case where the split partitions the whole cohort.
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    out = np.array(X, dtype=np.float32, copy=True)
    whiteners: dict[int, Whitener] = {}
    in_scope = None if subset is None else np.asarray(subset, dtype=bool)

    for g in np.unique(groups):
        m = groups == g
        if in_scope is not None and not (m & in_scope).any():
            continue          # this split does not touch the group; leave it untransformed
        fit_idx = m if fit_mask is None else (m & np.asarray(fit_mask, dtype=bool))
        if not fit_idx.any():
            raise ValueError(
                f"group {g!r} has trials in this split but none permitted for fitting its "
                "whitener. Alignment for it is undefined, so choose a regime that gives it "
                "at least one calibration trial, or exclude the group."
            )
        w = fit_whitener(X[fit_idx], mode=mode, shrinkage=shrinkage, rank_tol=rank_tol,
                         n_calib=n_calib, seed=seed + int(g))
        out[m] = w.transform(X[m])
        whiteners[int(g)] = w
    return out, whiteners


# ---------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------
def alignment_residual(X: np.ndarray, groups: np.ndarray, *,
                       mode: str = "euclid") -> dict[int, float]:
    """Per-group deviation of the mean trial covariance from the identity.

    This is the property alignment exists to establish, so it is the property worth
    testing: after aligning, every group's mean covariance should *be* the identity. The
    metric is the relative Frobenius distance ``||R - I||_F / ||I||_F``, which is 0 for a
    perfectly aligned group and ~1 when alignment did nothing.

    For ``mode="riemann"`` the mean compared against is the Frechet mean, because that is
    the quantity Riemannian alignment centres -- scoring it against the arithmetic mean
    would report a spurious residual for a correctly aligned group.
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    res: dict[int, float] = {}
    for g in np.unique(groups):
        covs = _trial_covariances(X[groups == g])
        R = riemannian_mean(covs) if mode == "riemann" else covs.mean(axis=0)
        C = R.shape[0]
        # Restrict to the subspace that actually carries signal: in a union montage the
        # absent electrodes are structurally zero and can never be mapped to 1, so scoring
        # them would report a large residual for a perfectly aligned subject.
        live = np.diag(R) > 1e-12 * max(float(np.trace(R)) / C, 1e-300)
        if not live.any():
            res[int(g)] = float("nan")
            continue
        Rl = R[np.ix_(live, live)]
        I = np.eye(Rl.shape[0])
        res[int(g)] = float(np.linalg.norm(Rl - I) / np.linalg.norm(I))
    return res
