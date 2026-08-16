"""Cache discovery and cohort assembly.

A prepared cache is a directory of per-subject ``sub-NNN.npz`` tensors with a
``sub-NNN.json`` sidecar describing exactly how they were made. Caches are located by
**reading those sidecars and matching the contract field by field**, not by recomputing an
opaque hash of the preprocessing parameters.

That choice is deliberate and it is the difference between two failure modes. With a hash,
a mismatch is invisible: the loader looks in a directory that does not exist, reports "no
cache", and the natural next step is to regenerate -- silently producing a second copy of
data that already exists under slightly different settings. With field matching, a
mismatch is *nameable*: the loader can say which field differs and by how much. It also
means a cache produced by different code is usable as long as it describes itself
honestly, which is what makes the corpus portable.

The pooled (multi-dataset) path is where the silent-failure risk concentrates, so three
invariants are enforced rather than assumed:

* **One time axis.** Datasets disagree on their native window. A pooled tensor needs a
  single ``n_times``; any disagreement raises instead of truncating.
* **Union montage matched by canonical electrode name**, never by position. A positional
  match between a 22-channel and a 64-channel montage yields a plausible tensor and a
  meaningless model.
* **Globally re-issued subject IDs.** Two corpora both have a "subject 1" and they are
  different people. Concatenating native IDs would let a leave-subjects-out split place
  one person's trials on both sides while believing it had not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from eegbench import PREPARED_ROOT
from eegbench.registry import DatasetSpec, get as get_spec

__all__ = [
    "montage_frontier",
    "Contract", "CacheEntry", "Cohort",
    "discover", "find_cache", "load_subject", "load_cohort", "canonical_channel",
]


# ---------------------------------------------------------------------------------------
# The preprocessing contract
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Contract:
    """What a cached trial tensor must be, for a run to be allowed to use it.

    Defaults are the shared pooling contract: 4-38 Hz (the band the motor-imagery
    deep-learning literature uses), 250 Hz, and 0.5-3.0 s post-cue -- 625 samples, the
    longest window every left/right corpus in the registry can supply.
    """

    paradigm: str = "leftright"   # leftright | motor_imagery | p300 | ssvep
    fmin: float = 4.0
    fmax: float = 38.0
    sfreq: float = 250.0
    tmin: float = 0.5
    tmax: float = 3.0

    @classmethod
    def for_paradigm(cls, paradigm: str, **over) -> "Contract":
        """Defaults appropriate to the *physics* of each paradigm.

        The motor-imagery contract is not a neutral starting point for the others, and
        using it would quietly destroy the signal rather than fail:

        * **P300** is a phase-locked evoked deflection peaking ~300 ms after the stimulus,
          carried at **0.1-8 Hz**. The MI contract's 4 Hz high-pass removes most of it, and
          its 0.5 s onset begins *after* the peak. Hence 1-24 Hz over 0-0.8 s.
        * **SSVEP** is narrowband entrainment at the tag frequency and its harmonics,
          occipital, and needs a long window for frequency resolution. Hence 7-45 Hz over
          0.5-3.5 s.
        * **Motor imagery** is non-phase-locked band-power modulation in mu/beta over
          sensorimotor cortex: 4-38 Hz, 0.5-3.0 s.

        A cross-paradigm comparison is therefore never like-for-like on preprocessing, and
        results must say so.
        """
        base = {
            "leftright": dict(fmin=4.0, fmax=38.0, tmin=0.5, tmax=3.0),
            "motor_imagery": dict(fmin=4.0, fmax=38.0, tmin=0.5, tmax=3.0),
            "p300": dict(fmin=1.0, fmax=24.0, tmin=0.0, tmax=0.8),
            "ssvep": dict(fmin=7.0, fmax=45.0, tmin=0.5, tmax=3.5),
        }
        if paradigm not in base:
            raise ValueError(f"unknown paradigm {paradigm!r}; known: {sorted(base)}")
        return cls(paradigm=paradigm, **{**base[paradigm], **over})

    @property
    def is_imbalanced(self) -> bool:
        """P300 is roughly 1:5 target:non-target, so plain accuracy is not a useful score."""
        return self.paradigm == "p300"

    def n_times(self) -> int:
        return int(round((self.tmax - self.tmin) * self.sfreq))

    def n_classes(self, spec: DatasetSpec) -> int:
        return len(self.class_names(spec))

    def class_names(self, spec: DatasetSpec) -> tuple[str, ...]:
        if self.paradigm == "leftright":
            return ("left_hand", "right_hand")
        if self.paradigm == "p300":
            # Order fixed here, NOT derived from the data: sklearn-style alphabetical
            # sorting would put NonTarget first for some subjects and not others, so class
            # 1 would mean different things in different rows of a results table.
            return ("NonTarget", "Target")
        return spec.events

    def matches(self, cfg: dict) -> tuple[bool, str]:
        """Field-by-field comparison against a sidecar's ``config``.

        Returns ``(ok, reason)``; ``reason`` names the first field that differs, which is
        what makes a cache miss diagnosable instead of merely absent.
        """
        checks = (
            ("paradigm", self.paradigm, cfg.get("paradigm")),
            ("fmin", self.fmin, cfg.get("fmin")),
            ("fmax", self.fmax, cfg.get("fmax")),
            ("sfreq", self.sfreq, cfg.get("sfreq")),
            ("tmin", self.tmin, cfg.get("tmin")),
            ("tmax", self.tmax, cfg.get("tmax")),
        )
        for name, want, got in checks:
            if isinstance(want, float):
                if got is None or abs(float(got) - want) > 1e-9:
                    return False, f"{name}: cache has {got!r}, contract wants {want!r}"
            elif got != want:
                return False, f"{name}: cache has {got!r}, contract wants {want!r}"
        return True, ""


# ---------------------------------------------------------------------------------------
# Channel-name canonicalization
# ---------------------------------------------------------------------------------------
# MNE/MOABB return 10-20 names with inconsistent casing across corpora ("FP1"/"Fp1",
# "FZ"/"Fz", "Oz"/"OZ"), and some carry a reference suffix ("C3-A1") or whitespace. Two
# spellings of one electrode would open two columns in the union montage, each half
# empty -- a wrong tensor rather than an error. Canonicalization is therefore part of the
# correctness argument, not a cosmetic step.
_CANON_RE = re.compile(r"[^A-Za-z0-9]")


def canonical_channel(name: str) -> str:
    """Canonical form of a 10-20 electrode label: ``'fp1' -> 'Fp1'``, ``'FZ' -> 'Fz'``.

    Strips any reference suffix (``C3-A1``), removes punctuation and whitespace, then
    applies the standard capitalization: letters upper-case, except that the trailing
    ``z`` of a midline electrode is lower-case (``Fz``, ``Cz``, ``POz``) and ``Fp`` keeps
    its lower-case ``p``.
    """
    base = name.split("-")[0]
    base = _CANON_RE.sub("", base)
    if not base:
        return name.strip()
    up = base.upper()
    # Midline electrodes end in Z; the convention writes that Z lower-case.
    if up.endswith("Z") and len(up) > 1:
        up = up[:-1] + "z"
    # Fp1/Fp2/Fpz are the one prefix written with a lower-case second letter.
    if up.startswith("FP"):
        up = "Fp" + up[2:]
    return up


# ---------------------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheEntry:
    """One prepared cache directory that satisfies a contract."""

    dataset: str
    path: Path
    config: dict
    subjects: tuple[int, ...]
    channels: tuple[str, ...]
    n_times: int
    n_trials: int

    @property
    def spec(self) -> DatasetSpec:
        return get_spec(self.dataset)


def _read_sidecars(d: Path) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    for js in sorted(d.glob("sub-*.json")):
        try:
            meta = json.loads(js.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        subj = meta.get("subject")
        if subj is None:
            m = re.search(r"sub-(\d+)", js.stem)
            if not m:
                continue
            subj = int(m.group(1))
        out.append((int(subj), meta))
    return out


def discover(root: Path | None = None) -> list[CacheEntry]:
    """Every prepared cache under ``root``, whatever contract it was built with.

    Used by the CLI to *explain* a cache miss ("this dataset is prepared, but at
    tmax=3.5") rather than only report one.
    """
    root = PREPARED_ROOT if root is None else Path(root)
    entries: list[CacheEntry] = []
    if not root.is_dir():
        return entries
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for para_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            for fp_dir in sorted(p for p in para_dir.iterdir() if p.is_dir()):
                sidecars = _read_sidecars(fp_dir)
                if not sidecars:
                    continue
                _, first = sidecars[0]
                entries.append(CacheEntry(
                    dataset=ds_dir.name,
                    path=fp_dir,
                    config=first.get("config", {}),
                    subjects=tuple(sorted(s for s, _ in sidecars)),
                    channels=tuple(first.get("channels", ())),
                    n_times=int(first.get("n_samples", 0)),
                    n_trials=int(sum(m.get("n_trials", 0) for _, m in sidecars)),
                ))
    return entries


def find_cache(dataset: str, contract: Contract,
               root: Path | None = None) -> CacheEntry:
    """The cache for ``dataset`` satisfying ``contract``, or a diagnosable error."""
    candidates = [e for e in discover(root) if e.dataset == dataset]
    if not candidates:
        raise FileNotFoundError(
            f"no prepared cache for {dataset!r} under "
            f"{PREPARED_ROOT if root is None else root}. Run: "
            f"python -m eegbench.prepare --dataset {dataset} "
            f"--paradigm {contract.paradigm} --tmin {contract.tmin} --tmax {contract.tmax}"
        )
    reasons = []
    for e in candidates:
        ok, why = contract.matches(e.config)
        if ok:
            want = contract.n_times()
            if e.n_times != want:
                raise ValueError(
                    f"{dataset}: cache at {e.path} matches the contract but holds "
                    f"{e.n_times} samples where the contract implies {want}. The cache "
                    "is inconsistent with its own sidecar -- do not use it."
                )
            return e
        reasons.append(f"  {e.path.name}: {why}")
    raise FileNotFoundError(
        f"{dataset!r} is prepared, but no cache matches the requested contract.\n"
        + "\n".join(reasons)
        + f"\nPrepare it: python -m eegbench.prepare --dataset {dataset} "
          f"--paradigm {contract.paradigm} --tmin {contract.tmin} --tmax {contract.tmax}"
    )


def load_subject(entry: CacheEntry, subject: int) -> dict:
    """Load one subject's trials from a cache entry."""
    npz = entry.path / f"sub-{subject:03d}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} missing (cache lists subjects {entry.subjects})")
    meta = json.loads((entry.path / f"sub-{subject:03d}.json").read_text())
    with np.load(npz, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    out["channels"] = [canonical_channel(c) for c in meta.get("channels", [])]
    out["meta"] = meta
    return out


# ---------------------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------------------
@dataclass
class Cohort:
    """A pooled, montage-aligned set of trials ready for splitting.

    ``X`` is ``(N, C, T)`` on the union montage; absent electrodes are exactly zero **and**
    marked invalid in ``valid``. Zero-filling alone is not enough -- a convolution with a
    bias emits a constant from a dead channel, which the network can and does read as a
    dataset identity cue.
    """

    X: np.ndarray               # (N, C, T) float32
    y: np.ndarray               # (N,) int64
    subject: np.ndarray         # (N,) int64, globally unique
    dataset: np.ndarray         # (N,) int64, index into `dataset_names`
    session: np.ndarray         # (N,) <U, native session label
    valid: np.ndarray           # (N, C) bool -- per-trial electrode validity
    channels: list[str]         # length C, canonical names
    dataset_names: list[str]
    class_names: tuple[str, ...]
    #: Maps each global subject id back to ``(dataset_key, native_subject_id)``.
    subject_index: dict[int, tuple[str, int]]

    @property
    def n_classes(self) -> int:
        return int(self.y.max()) + 1 if self.y.size else 0

    def summary(self) -> dict:
        per_ds = {
            name: int((self.dataset == i).sum())
            for i, name in enumerate(self.dataset_names)
        }
        return {
            "n_trials": int(self.X.shape[0]),
            "n_subjects": int(np.unique(self.subject).size),
            "n_channels": int(self.X.shape[1]),
            "n_times": int(self.X.shape[2]),
            "n_classes": self.n_classes,
            "datasets": self.dataset_names,
            "trials_per_dataset": per_ds,
            "channel_occupancy": float(self.valid.mean()),
        }


def _assert_montages_are_comparable(keys: list[str],
                                    per_dataset: list[list[str]]) -> None:
    """Refuse to pool a corpus that shares no electrode with any other.

    Not every corpus reports electrode *names*. At least one in the registry labels its
    channels ``EEG1``..``EEG41``, and those are the names MOABB itself returns -- not a
    fallback introduced here. Pooled on a union montage, such a corpus opens 41 private
    columns that match nothing, so its trials are structurally zero everywhere the other
    corpora have signal and vice versa. Nothing raises: the tensor is well-formed, the
    model trains, and what it learns is partly "which dataset is this".

    Zero shared electrodes is the signature, and it is checked against the union of the
    *other* corpora rather than against the whole union -- otherwise a corpus always
    matches itself and the check can never fire.
    """
    # Positional labels (`ch0`, `EEG1`) name no anatomy, so they cannot be matched across
    # corpora even when two datasets happen to use the same strings -- `ch3` in one
    # montage and `ch3` in another are different electrodes. Checked before the overlap
    # test, which would otherwise be *satisfied* by two positional corpora agreeing on
    # meaningless names, exactly inverting the guard.
    # Nothing to reconcile with one corpus. Positional labels are perfectly usable on
    # their own -- the electrodes are consistent within the dataset, which is all a
    # single-dataset cohort needs -- so the checks below are about *pairs* of montages and
    # must not fire here. Refusing a lone corpus would discard real subjects to avoid a
    # problem it does not have.
    if len(keys) < 2:
        return

    def _positional(c: str) -> bool:
        # Bare numerals ('1', '42') as well as prefixed indices ('ch0', 'EEG17'). The
        # numeric form is not hypothetical: one 128-channel corpus in the registry returns
        # its montage as '1'..'128'.
        if c.isdigit():
            return True
        return c.lower().startswith(("ch", "eeg")) and c[-1].isdigit()

    for key, chans in zip(keys, per_dataset):
        if any(_positional(c) for c in chans[:4]):
            raise ValueError(
                f"{key!r} reports positional channel labels ({chans[:4]}) rather than "
                "electrode names, so it cannot be placed on a shared montage: matching "
                "them across corpora would align different electrodes under equal names. "
                "Use it as a single-dataset cohort, or recover its montage from the "
                "source documentation."
            )

    sets = [set(c) for c in per_dataset]
    for i, key in enumerate(keys):
        others: set[str] = set().union(*(s for j, s in enumerate(sets) if j != i))
        shared = sets[i] & others
        if not shared:
            raise ValueError(
                f"{key!r} shares no electrode name with the rest of the cohort "
                f"{[k for k in keys if k != key]}. Its channels are "
                f"{sorted(sets[i])[:6]}... which look positional rather than anatomical. "
                "Pooling it on a union montage would give it private columns matching no "
                "other corpus, and the result would encode dataset identity rather than "
                "neural signal. Use it as a single-dataset cohort, or recover its "
                "electrode names first."
            )


def _union_channels(per_dataset: list[list[str]]) -> list[str]:
    """Union of montages, ordered by how many corpora contain each electrode.

    Ordering by occupancy (most-shared first) is not cosmetic. Every architecture that
    mixes electrodes with a dense ``C -> F`` layer sees this axis, and putting the
    universally-present electrodes -- C3/Cz/C4 and their neighbours, which is where motor
    imagery lives -- in low, stable columns keeps that layer's most useful inputs in the
    same place as the cohort grows. Ties break alphabetically so the order is
    deterministic.
    """
    counts: dict[str, int] = {}
    for chans in per_dataset:
        for c in set(chans):
            counts[c] = counts.get(c, 0) + 1
    return sorted(counts, key=lambda c: (-counts[c], c))


def load_cohort(
    datasets: str | list[str],
    contract: Contract | None = None,
    *,
    subjects: dict[str, list[int]] | None = None,
    trials_per_subject: int | None = None,
    montage: str = "auto",
    channels: list[str] | None = None,
    seed: int = 0,
    root: Path | None = None,
) -> Cohort:
    """Assemble one or more prepared datasets into a single cohort.

    ``montage``
        ``"native"`` requires a single dataset and keeps its own electrode order.

        ``"union"`` takes every electrode any corpus supplies and zero-fills the rest.
        Maximises electrodes at the cost of occupancy: across the eleven poolable corpora
        here it gives 136 electrodes at **32%** occupancy, so roughly two thirds of the
        input tensor is structural zero and any layer that mixes electrodes has to learn
        around a mask that changes per trial.

        ``"shared"`` takes only electrodes present in **every** selected corpus, giving
        **100% occupancy and no structural zeros at all**. This is the montage mode to
        reach for when the question is about subjects rather than electrodes: it removes
        dilution as a confound entirely, so a scaling curve measured on it is a statement
        about population size and not about how much of the tensor is padding.

        The trade is real and worth seeing before choosing -- use
        :func:`montage_frontier`. On this corpus it runs from 376 subjects at 3
        electrodes (C3/C4/Cz) to 185 at 58, with an attractive middle: **354 subjects at
        9 electrodes**, i.e. 94% of the people with none of the padding.

        ``"auto"`` picks native for one dataset and union for several.

    ``channels``
        An explicit electrode list, overriding ``montage``. Corpora missing any of them
        are refused rather than zero-filled, so the resulting tensor is always full.

    ``trials_per_subject``
        Caps each subject's trial count, class-balanced and seeded. Used by the arm that
        separates *subject diversity* from *data volume*: without the class balancing a
        head-slice would take one class disproportionately, because several of these
        corpora order trials by block.
    """
    contract = Contract() if contract is None else contract
    keys = [datasets] if isinstance(datasets, str) else list(datasets)
    if not keys:
        raise ValueError("load_cohort needs at least one dataset")
    if montage == "auto":
        montage = "native" if len(keys) == 1 else "union"
    if montage not in ("native", "union", "shared"):
        raise ValueError(f"montage must be auto|native|union|shared, got {montage!r}")
    if montage == "native" and len(keys) > 1:
        raise ValueError(
            f"montage='native' is only defined for a single dataset, got {keys}. "
            "Pooling requires a union montage -- otherwise electrode k means a different "
            "electrode in different rows of the same tensor."
        )

    entries = [find_cache(k, contract, root) for k in keys]

    # One time axis, checked rather than trusted.
    lengths = {e.dataset: e.n_times for e in entries}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"datasets disagree on trial length: {lengths}. A pooled tensor needs one "
            "n_times; re-prepare the odd one out on the shared contract rather than "
            "resampling here, so the window stays a documented property of the cache."
        )

    per_ds_channels = [list(e.channels) for e in entries]
    if channels is not None:
        union = [canonical_channel(c) for c in channels]
        missing = {k: sorted(set(union) - set(ch))
                   for k, ch in zip(keys, per_ds_channels)
                   if set(union) - set(ch)}
        if missing:
            raise ValueError(
                f"explicit channel list is not satisfied by every corpus: {missing}. "
                "Zero-filling the gap would reintroduce exactly the structural padding "
                "an explicit montage exists to avoid -- drop those corpora, or shorten "
                "the list (see montage_frontier)."
            )
    elif montage == "union":
        _assert_montages_are_comparable(keys, per_ds_channels)
        union = _union_channels(per_ds_channels)
    elif montage == "shared":
        _assert_montages_are_comparable(keys, per_ds_channels)
        shared = set.intersection(*[set(c) for c in per_ds_channels])
        if not shared:
            raise ValueError(
                f"corpora {keys} share no electrode, so there is no common montage. "
                "Use montage='union' and accept the padding, or drop the corpus with the "
                "narrowest montage (see montage_frontier)."
            )
        # Ordered by the same occupancy rule as the union, so that a `shared` montage is a
        # column-wise prefix of the corresponding `union` one wherever both are defined.
        # That keeps the two comparable: electrode k means the same electrode in both.
        union = [c for c in _union_channels(per_dataset=per_ds_channels) if c in shared]
    else:
        union = list(per_ds_channels[0])
    col = {c: i for i, c in enumerate(union)}
    C, T = len(union), entries[0].n_times

    Xs, ys, subs, dss, sess, valids = [], [], [], [], [], []
    subject_index: dict[int, tuple[str, int]] = {}
    next_gid = 0
    rng = np.random.default_rng(seed)

    for ds_i, entry in enumerate(entries):
        wanted = entry.subjects
        if subjects and entry.dataset in subjects:
            req = set(subjects[entry.dataset])
            missing = req - set(wanted)
            if missing:
                raise KeyError(
                    f"{entry.dataset}: requested subjects {sorted(missing)} are not in "
                    f"the cache (has {list(wanted)})"
                )
            wanted = tuple(s for s in wanted if s in req)

        # Which union columns this corpus fills. Computed once: it is a property of the
        # montage, not of the trial.
        src_cols, dst_cols = [], []
        for src, name in enumerate(entry.channels):
            if name in col:
                src_cols.append(src)
                dst_cols.append(col[name])
        if not dst_cols:
            raise ValueError(
                f"{entry.dataset}: none of its electrodes {entry.channels[:5]}... appear "
                f"in the target montage. This is a name-canonicalization failure, not an "
                f"empty dataset."
            )
        src_cols = np.asarray(src_cols)
        dst_cols = np.asarray(dst_cols)

        for native_subj in wanted:
            rec = load_subject(entry, native_subj)
            Xi, yi = rec["X"], rec["y"]
            si = rec.get("session", np.zeros(len(yi), dtype="<U1"))
            if trials_per_subject is not None and yi.size > trials_per_subject:
                keep = _balanced_cap(yi, trials_per_subject, rng)
                Xi, yi, si = Xi[keep], yi[keep], si[keep]
            n = Xi.shape[0]
            if n == 0:
                continue

            xb = np.zeros((n, C, T), dtype=np.float32)
            xb[:, dst_cols, :] = Xi[:, src_cols, :].astype(np.float32, copy=False)

            vb = np.zeros((n, C), dtype=bool)
            # A cache may mark individual electrodes bad; respect that on top of presence.
            cache_valid = rec.get("valid_mask")
            if cache_valid is not None and np.asarray(cache_valid).shape[-1] == len(entry.channels):
                cv = np.asarray(cache_valid).astype(bool)
                cv = cv[src_cols] if cv.ndim == 1 else cv[:, src_cols]
                vb[:, dst_cols] = cv
            else:
                vb[:, dst_cols] = True

            gid = next_gid
            next_gid += 1
            subject_index[gid] = (entry.dataset, int(native_subj))

            Xs.append(xb)
            ys.append(yi.astype(np.int64, copy=False))
            subs.append(np.full(n, gid, dtype=np.int64))
            dss.append(np.full(n, ds_i, dtype=np.int64))
            sess.append(np.asarray(si, dtype="<U16"))
            valids.append(vb)

    if not Xs:
        raise ValueError(f"cohort {keys} assembled zero trials")

    spec0 = entries[0].spec
    return Cohort(
        X=np.concatenate(Xs, 0),
        y=np.concatenate(ys, 0),
        subject=np.concatenate(subs, 0),
        dataset=np.concatenate(dss, 0),
        session=np.concatenate(sess, 0),
        valid=np.concatenate(valids, 0),
        channels=union,
        dataset_names=[e.dataset for e in entries],
        class_names=contract.class_names(spec0),
        subject_index=subject_index,
    )


def montage_frontier(datasets: list[str], contract: Contract | None = None,
                     root: Path | None = None) -> list[dict]:
    """The subjects-vs-electrodes trade available at 100% occupancy.

    Pooling forces a choice that is usually made implicitly and badly: a union montage
    keeps every corpus and pays for it in padding, while a shared montage keeps every
    electrode real and pays for it in corpora. Neither is right in general -- the trade
    depends on how much the narrowest corpus costs, which is a property of *this*
    collection and cannot be reasoned about in the abstract.

    This walks the frontier greedily: at each step drop whichever corpus is limiting the
    shared montage most, and record what that buys in electrodes and costs in subjects.
    Every row describes a cohort with **no structural zeros**.

    Returns rows of ``{n_corpora, n_subjects, n_channels, datasets, channels, dropped}``,
    widest cohort first.
    """
    contract = contract or Contract()
    info = {}
    for k in datasets:
        e = find_cache(k, contract, root)
        info[k] = (set(e.channels), len(e.subjects))

    rows, cur, dropped = [], set(datasets), []
    while cur:
        shared = set.intersection(*[info[n][0] for n in cur])
        rows.append({
            "n_corpora": len(cur),
            "n_subjects": sum(info[n][1] for n in cur),
            "n_channels": len(shared),
            "datasets": sorted(cur),
            "channels": sorted(shared),
            "dropped": list(dropped),
        })
        if len(cur) == 1:
            break
        # Drop the corpus whose removal most widens the shared montage; ties go to the
        # one costing fewer subjects, since electrodes bought with people are the
        # expensive kind.
        best = None
        for n in cur:
            rest = cur - {n}
            gain = len(set.intersection(*[info[m][0] for m in rest]))
            cost = info[n][1]
            if best is None or (gain, -cost) > (best[1], -best[2]):
                best = (n, gain, cost)
        dropped = dropped + [best[0]]
        cur = cur - {best[0]}
    return rows


def _balanced_cap(y: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of at most ``cap`` trials, as class-balanced as the data allows."""
    classes = np.unique(y)
    per = max(1, cap // len(classes))
    keep: list[np.ndarray] = []
    for c in classes:
        pool = np.flatnonzero(y == c)
        pool = pool[rng.permutation(pool.size)]
        keep.append(pool[:per])
    idx = np.sort(np.concatenate(keep))
    return idx[:cap]
