"""Evaluation protocols, constructed so that folds pair across arms.

The invariant this module exists to guarantee
---------------------------------------------
Every arm of a benchmark must see **literally identical** index arrays. Comparisons
between architectures are made with a paired test over folds -- that is where the
statistical power is at n=10 -- and a paired test is meaningless if arm A's "fold 3" holds
different subjects than arm B's.

Guaranteeing that is harder than seeding a shuffle, because a benchmark runner dispatches
folds to different processes and may assemble a cohort in a different dataset order. So
folds are **not** derived from array positions or from a global RNG stream. They are
derived from a *stable subject key* -- ``(dataset, native_subject_id)`` -- put in canonical
order and permuted by a seeded RNG. Two processes that assemble the same subjects in
different orders therefore produce the same folds, and a fold computed alone is identical
to the same fold computed as part of a sweep.

Protocols
---------
``cross_subject``    leave-N-subjects-out. The regime that tests transfer, and the one the
                     zero-shot question lives in.
``within_session``   stratified k-fold inside each session. Easiest cell; useful as an
                     upper bound and as a pipeline gate against published numbers.
``holdout_session``  session 1 trains, session 2 tests. The BCI-Competition
                     subject-dependent protocol most published 2a numbers use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from eegbench.store import Cohort

__all__ = [
    "Split", "cross_subject", "cross_dataset", "within_session", "holdout_session",
    "build_splits", "assert_integrity", "PROTOCOLS",
]

PROTOCOLS = ("cross-subject", "cross-dataset", "within-session", "holdout-session")


@dataclass
class Split:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"name": self.name, "n_train": int(self.train.size),
                "n_val": int(self.val.size), "n_test": int(self.test.size), **self.meta}


# ---------------------------------------------------------------------------------------
# Deterministic subject ordering
# ---------------------------------------------------------------------------------------
def _canonical_subject_order(cohort: Cohort, seed: int) -> np.ndarray:
    """Global subject ids, permuted deterministically from their stable keys.

    The sort is on ``(dataset_key, native_subject_id)``, which is a property of the person
    rather than of this particular cohort assembly. Permuting *that* order means the fold
    a subject lands in depends only on the seed and on who else is in the cohort -- not on
    the order datasets were listed on the command line.
    """
    gids = np.unique(cohort.subject)
    keyed = sorted(gids, key=lambda g: (cohort.subject_index[int(g)][0],
                                        cohort.subject_index[int(g)][1]))
    order = np.asarray(keyed, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return order[rng.permutation(order.size)]


def _stratified_carve(y: np.ndarray, idx: np.ndarray, frac: float,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``idx`` into ``(keep, held)`` holding out ``frac``, stratified by class.

    ``frac <= 0`` holds nothing out, and that is a supported mode rather than a degenerate
    one: with a fixed epoch budget and final-model selection there is nothing to select
    on, so surrendering trials to a validation set only shrinks the training set.
    """
    idx = np.asarray(idx)
    if frac <= 0 or idx.size == 0:
        return idx, np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    held: list[np.ndarray] = []
    for cls in np.unique(y[idx]):
        pool = idx[y[idx] == cls]
        pool = pool[rng.permutation(pool.size)]
        held.append(pool[: max(1, int(round(frac * pool.size)))])
    held_arr = np.sort(np.concatenate(held)) if held else np.array([], dtype=np.int64)
    return np.setdiff1d(idx, held_arr), held_arr


# ---------------------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------------------
def cross_subject(cohort: Cohort, *, n_folds: int = 10, val_subjects: int = 5,
                  seed: int = 0, test_datasets: list[str] | None = None) -> list[Split]:
    """Leave-N-subjects-out over the cohort.

    Grouped by subject: every trial from a held-out subject is in test and none is in
    train. Validation subjects are drawn from the training pool, never from test, so a
    model-selection decision can never see a test subject.

    ``test_datasets``
        Restricts the *evaluation population* to subjects from these corpora. Subjects
        from every other corpus are always in training and never tested.

        This exists because a scaling curve is otherwise uninterpretable, and the failure
        is subtle enough that it has been misread here already. Growing a cohort changes
        the training set **and the test set together**, so the reported mean absorbs
        whatever difficulty the newly-added corpora bring: measured on this cohort, adding
        liu2024 (54.0) and wairagkar2018 (44.9) to a cho2017 baseline drops the pooled
        mean from 66.6 to 57.1 while cho2017's own subjects stay at 66.5 -- flat. The
        pooled number says "more data made it worse"; the per-corpus breakdown says
        nothing got worse at all.

        Pinning the test population to one corpus makes the comparison mean what it is
        supposed to mean: same people evaluated every time, only the training set grows,
        so a difference is transfer rather than composition.
    """
    if test_datasets:
        unknown = set(test_datasets) - set(cohort.dataset_names)
        if unknown:
            raise ValueError(
                f"test_datasets {sorted(unknown)} are not in this cohort "
                f"{cohort.dataset_names}"
            )
    order = _canonical_subject_order(cohort, seed)
    if test_datasets:
        eligible = {g for g in order
                    if cohort.subject_index[int(g)][0] in set(test_datasets)}
        held_in = np.array([g for g in order if g not in eligible], dtype=np.int64)
        order = np.array([g for g in order if g in eligible], dtype=np.int64)
        if order.size < n_folds:
            raise ValueError(
                f"test_datasets {test_datasets} supply {order.size} subjects, too few for "
                f"{n_folds} folds"
            )
    else:
        held_in = np.array([], dtype=np.int64)
    if order.size < n_folds:
        raise ValueError(
            f"cohort has {order.size} subjects, cannot make {n_folds} leave-subjects-out "
            "folds; each fold needs at least one test subject"
        )
    folds = np.array_split(order, n_folds)

    splits: list[Split] = []
    for k, test_subj in enumerate(folds):
        pool = np.setdiff1d(order, test_subj)
        n_val = min(val_subjects, max(1, pool.size // 5)) if val_subjects > 0 else 0
        # Take validation subjects from the END of the (already permuted) pool. Taking
        # them from the front would make fold k's validation set a prefix of fold k+1's,
        # correlating the folds a paired test assumes are independent draws.
        val_subj = pool[pool.size - n_val:] if n_val else np.array([], dtype=np.int64)
        # Subjects from non-test corpora are always trained on and never evaluated, which
        # is the whole point of `test_datasets`.
        train_subj = np.union1d(np.setdiff1d(pool, val_subj), held_in)
        splits.append(Split(
            name=f"fold{k}",
            train=np.flatnonzero(np.isin(cohort.subject, train_subj)),
            val=np.flatnonzero(np.isin(cohort.subject, val_subj)),
            test=np.flatnonzero(np.isin(cohort.subject, test_subj)),
            meta={
                "fold": k,
                "n_train_subjects": int(train_subj.size),
                "n_val_subjects": int(val_subj.size),
                "n_test_subjects": int(test_subj.size),
                # Recorded as stable keys, not global ids: a results file must remain
                # interpretable when a later cohort assigns different global ids.
                "test_subjects": [list(cohort.subject_index[int(s)]) for s in test_subj],
            },
        ))
    return splits


def cross_dataset(cohort: Cohort, *, val_subjects: int = 0, seed: int = 0,
                  test_datasets: list[str] | None = None) -> list[Split]:
    """Leave-one-**corpus**-out. The operational definition of a general encoder.

    One fold per corpus: every subject of that corpus is tested, and the model trains
    **exclusively** on the others. This is strictly harder than leave-subjects-out and it
    is the regime a deployed encoder actually meets -- a new lab, different amplifier,
    different cap, different population, different task framing.

    Why this is a separate protocol rather than a flag on
    :func:`cross_subject`: that function's ``test_datasets`` pins the *evaluation
    population* while still training on the held-in subjects of those same corpora, which
    is correct for a scaling curve (same people scored every time) and wrong here. Asking
    "can it generalise to an unseen corpus" while 13 of that corpus's 14 subjects are in
    the training set answers a different question and would answer it flatteringly.

    Validation subjects, when requested, are drawn from the *training* corpora, so no part
    of the held-out corpus is ever seen.
    """
    names = list(cohort.dataset_names)
    targets = names if not test_datasets else list(test_datasets)
    unknown = set(targets) - set(names)
    if unknown:
        raise ValueError(f"test_datasets {sorted(unknown)} not in cohort {names}")
    if len(names) < 2:
        raise ValueError(
            f"cross-dataset needs at least 2 corpora, got {names}. With one corpus there "
            "is nothing to hold out."
        )

    ds_of = {int(g): cohort.subject_index[int(g)][0] for g in np.unique(cohort.subject)}
    rng = np.random.default_rng(seed)
    splits: list[Split] = []
    for k, target in enumerate(sorted(targets)):
        test_subj = np.array([g for g, d in ds_of.items() if d == target], dtype=np.int64)
        train_pool = np.array([g for g, d in ds_of.items() if d != target], dtype=np.int64)
        if test_subj.size == 0 or train_pool.size == 0:
            continue
        train_pool = train_pool[rng.permutation(train_pool.size)]
        n_val = min(val_subjects, max(0, train_pool.size - 1)) if val_subjects > 0 else 0
        val_subj = train_pool[:n_val]
        train_subj = train_pool[n_val:]
        splits.append(Split(
            name=f"holdout-{target}",
            train=np.flatnonzero(np.isin(cohort.subject, train_subj)),
            val=np.flatnonzero(np.isin(cohort.subject, val_subj)),
            test=np.flatnonzero(np.isin(cohort.subject, test_subj)),
            meta={"fold": k, "held_out_dataset": target,
                  "n_train_subjects": int(train_subj.size),
                  "n_val_subjects": int(val_subj.size),
                  "n_test_subjects": int(test_subj.size),
                  "train_datasets": sorted({d for g, d in ds_of.items() if d != target})},
        ))
    if not splits:
        raise ValueError(f"cross-dataset produced no folds for targets {targets}")
    return splits


def within_session(cohort: Cohort, *, n_folds: int = 5, val_frac: float = 0.2,
                   seed: int = 0) -> list[Split]:
    """Stratified k-fold inside each (subject, session), independently."""
    from sklearn.model_selection import StratifiedKFold

    splits: list[Split] = []
    for gid in np.unique(cohort.subject):
        sm = cohort.subject == gid
        for sess in sorted(set(cohort.session[sm].tolist())):
            pool = np.flatnonzero(sm & (cohort.session == sess))
            if pool.size < n_folds * 2:
                continue
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for k, (tr, te) in enumerate(skf.split(pool, cohort.y[pool])):
                trainval, test_idx = pool[tr], pool[te]
                train_idx, val_idx = _stratified_carve(cohort.y, trainval, val_frac, seed + k)
                ds, native = cohort.subject_index[int(gid)]
                splits.append(Split(
                    name=f"{ds}-s{native}/{sess}/fold{k}",
                    train=train_idx, val=val_idx, test=test_idx,
                    meta={"subject": int(gid), "dataset": ds, "native_subject": native,
                          "session": str(sess), "fold": k},
                ))
    if not splits:
        raise ValueError("within-session produced no splits: sessions are too small")
    return splits


#: Corpora label their sessions rather than merely ordering them -- BCI IV 2a uses
#: ``0train``/``1test``, 2b uses ``0train,1train,2train,3test,4test``. Splitting on
#: position alone gets 2a right by luck and 2b badly wrong: it would train on the first
#: session only (96 trials of ~290) and test on the second *training* session, which is
#: neither the published protocol nor a meaningful one.
_TRAIN_PAT = re.compile(r"(train|_T$|^T$)", re.I)
_TEST_PAT = re.compile(r"(test|eval|_E$|^E$)", re.I)


def _split_sessions(sessions: list[str], ds: str, native: int) -> tuple[list[str], list[str]]:
    """Partition a subject's sessions into (train, test) by their labels, not position."""
    tr = [s for s in sessions if _TRAIN_PAT.search(s)]
    te = [s for s in sessions if _TEST_PAT.search(s) and not _TRAIN_PAT.search(s)]
    if tr and te:
        return tr, te
    if len(sessions) >= 2:
        # Unlabelled sessions: fall back to first-vs-rest, and say so loudly. A silent
        # guess here produces a plausible number under the wrong protocol.
        import sys
        print(f"[splits] WARNING: {ds} subject {native} has unlabelled sessions "
              f"{sessions}; falling back to {sessions[0]!r} -> train, {sessions[1:]!r} -> "
              "test. Verify this is the intended protocol.", file=sys.stderr)
        return [sessions[0]], sessions[1:]
    raise ValueError(
        f"{ds} subject {native} has sessions {sessions}; the holdout-session protocol "
        "needs at least 2. Use --protocol within-session for single-session corpora."
    )


def holdout_session(cohort: Cohort, *, val_frac: float = 0.2, seed: int = 0) -> list[Split]:
    """Per subject: the training session(s) train, the evaluation session(s) test.

    Sessions are partitioned by their **labels**, not their order, so a corpus with three
    training sessions and two test sessions is handled correctly. Every subject must have
    at least two sessions; one with fewer raises rather than being silently dropped, since
    a protocol that quietly evaluates a subset of the cohort is how a table ends up
    describing a different experiment than its caption claims.
    """
    splits: list[Split] = []
    for gid in np.unique(cohort.subject):
        sm = cohort.subject == gid
        sessions = sorted(set(cohort.session[sm].tolist()))
        ds, native = cohort.subject_index[int(gid)]
        tr_names, te_names = _split_sessions(sessions, ds, native)
        train_pool = np.flatnonzero(sm & np.isin(cohort.session, tr_names))
        test_idx = np.flatnonzero(sm & np.isin(cohort.session, te_names))
        train_idx, val_idx = _stratified_carve(cohort.y, train_pool, val_frac, seed)
        splits.append(Split(
            name=f"{ds}-s{native}/holdout",
            train=train_idx, val=val_idx, test=test_idx,
            meta={"subject": int(gid), "dataset": ds, "native_subject": native,
                  "train_sessions": [str(s) for s in tr_names],
                  "test_sessions": [str(s) for s in te_names]},
        ))
    return splits


def build_splits(cohort: Cohort, protocol: str, *, n_folds: int = 10,
                 val_frac: float = 0.2, val_subjects: int = 5,
                 seed: int = 0, test_datasets: list[str] | None = None) -> list[Split]:
    if protocol == "cross-subject":
        return cross_subject(cohort, n_folds=n_folds, val_subjects=val_subjects,
                             seed=seed, test_datasets=test_datasets)
    if protocol == "cross-dataset":
        return cross_dataset(cohort, val_subjects=val_subjects, seed=seed,
                             test_datasets=test_datasets)
    if test_datasets:
        raise ValueError(
            f"--test-datasets is only defined for the cross-subject protocol, not "
            f"{protocol!r}: the other protocols evaluate every subject by construction."
        )
    if protocol == "within-session":
        return within_session(cohort, n_folds=n_folds, val_frac=val_frac, seed=seed)
    if protocol == "holdout-session":
        return holdout_session(cohort, val_frac=val_frac, seed=seed)
    raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")


def assert_integrity(split: Split, cohort: Cohort) -> None:
    """Train/val/test disjoint, in range, non-empty where required.

    For a cross-subject split this also re-derives the subject-level disjointness rather
    than trusting that index construction preserved it -- subject leakage is the one error
    in this file that would inflate every number rather than break the run.
    """
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = np.intersect1d(getattr(split, a), getattr(split, b))
        if overlap.size:
            raise AssertionError(
                f"split {split.name!r}: {a} and {b} share {overlap.size} trial indices"
            )
    allidx = np.concatenate([split.train, split.val, split.test])
    n = cohort.X.shape[0]
    if allidx.size and (allidx.min() < 0 or allidx.max() >= n):
        raise AssertionError(f"split {split.name!r}: index outside [0, {n})")
    if split.train.size == 0 or split.test.size == 0:
        raise AssertionError(f"split {split.name!r}: empty train or test")

    if "n_test_subjects" in split.meta:
        tr = set(cohort.subject[split.train].tolist()) | set(cohort.subject[split.val].tolist())
        te = set(cohort.subject[split.test].tolist())
        shared = tr & te
        if shared:
            raise AssertionError(
                f"split {split.name!r}: subjects {sorted(shared)} appear in BOTH the "
                "training/validation pool and the test set. Every cross-subject number "
                "from this split would be inflated."
            )
