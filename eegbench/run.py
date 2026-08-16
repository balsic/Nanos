"""Single-arm entry point: one arm, one dataset, one seed, one or all folds.

    python -m eegbench.run --arm ours_noattn --dataset cho2017 --protocol cross-subject \
        --align euclid --regime transductive --folds 10 --out results/cho_ea_ours.json

``--fold K`` runs exactly one fold and writes a shard. That is what makes fold-level
parallelism possible, and it is safe *because splits do not depend on which folds are
run*: :func:`eegbench.splits.cross_subject` derives the whole fold structure from a
canonical subject ordering seeded once, so fold 3 computed alone is bit-identical to fold 3
computed as part of a sweep. Shards are merged by :mod:`eegbench.bench.aggregate`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from eegbench import RESULTS_ROOT
from eegbench.align import align_by_group
from eegbench.engine import TrainConfig, run_split
from eegbench.models import ARMS
from eegbench.regimes import REGIME_HELP, parse_regime, resolve
from eegbench.splits import PROTOCOLS, assert_integrity, build_splits
from eegbench.store import Contract, load_cohort

__all__ = ["main", "config_fingerprint", "preprocess_for_split"]


# ---------------------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------------------
#: Fields that define *what experiment this is*. Two results may be compared only if these
#: agree. Everything outside this set (output path, device, log verbosity) may differ.
#:
#: Recorded with defaults resolved, so "key absent" and "key present at its default" are
#: the same string -- otherwise adding a flag silently invalidates every earlier result,
#: and a stale-result guard that fires constantly gets switched off.
CONTROLLED = (
    "arm", "arm_cfg", "datasets", "paradigm", "protocol", "fmin", "fmax", "sfreq",
    "tmin", "tmax", "montage", "channels", "folds", "val_frac", "val_subjects", "regime",
    "test_datasets",
    "align", "align_scope", "align_shrinkage", "norm", "trials_per_subject",
    "epochs", "batch_size", "lr", "weight_decay", "label_smoothing", "grad_clip",
    "scheduler", "select", "patience", "test_time_bn", "seed", "null",
)


def config_fingerprint(cfg: dict) -> str:
    """Stable hash over the controlled fields only."""
    payload = {k: cfg.get(k) for k in CONTROLLED}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def resolve_config(args: argparse.Namespace) -> dict:
    """Canonical config dict from parsed arguments.

    The *single* place raw argument strings become typed values. It exists because there
    were briefly two: this function's logic inline in ``main``, and the scheduler's
    re-parse for resume. They disagreed on one field -- ``--align-shrinkage`` arrives as
    the string ``"0.0"`` and is used as the float ``0.0`` -- so identical runs hashed
    differently and every cached shard was discarded as stale. The fingerprint caught it,
    which is what it is for, but the real fix is that there is now one function and both
    callers use it.
    """
    cfg = dict(vars(args))
    # Resolve the preprocessing contract from the paradigm before anything hashes it, so
    # the fingerprint records the band/window actually used rather than None.
    if "paradigm" in cfg:
        from eegbench.store import Contract
        over = {k: cfg[k] for k in ("fmin", "fmax", "tmin", "tmax")
                if cfg.get(k) is not None}
        c = Contract.for_paradigm(cfg["paradigm"], sfreq=cfg.get("sfreq", 250.0), **over)
        cfg.update(fmin=c.fmin, fmax=c.fmax, tmin=c.tmin, tmax=c.tmax)
    raw = cfg.get("align_shrinkage", 0.0)
    if isinstance(raw, str):
        if raw == "auto":
            cfg["align_shrinkage"] = "auto"
        else:
            try:
                cfg["align_shrinkage"] = float(raw)
            except ValueError:
                raise SystemExit(
                    f"--align-shrinkage must be a number or 'auto', got {raw!r}")
    if isinstance(cfg.get("arm_cfg"), str):
        cfg["arm_cfg"] = json.loads(cfg["arm_cfg"])
    return cfg


# ---------------------------------------------------------------------------------------
# Per-split preprocessing
# ---------------------------------------------------------------------------------------
def preprocess_for_split(cohort, split, args, seed: int) -> tuple[np.ndarray, dict]:
    """Alignment and normalization, applied under the declared regime.

    Runs **inside** the fold loop, not once over the whole cohort. That placement is the
    difference between measuring what the flags say and measuring something looser: done
    outside, every per-subject statistic sees every trial, and the regime becomes
    decorative.
    """
    regime = parse_regime(args.regime)
    fit_mask, info = resolve(cohort, split, regime, seed=seed)
    X = cohort.X
    # Trials this split actually uses. Under cross-subject this is everything; under the
    # subject-dependent protocols it is one subject, and the rest must be left alone.
    in_split = np.zeros(X.shape[0], dtype=bool)
    for idx in (split.train, split.val, split.test):
        if idx.size:
            in_split[idx] = True
    info["n_trials_in_split"] = int(in_split.sum())

    # Grouping for per-subject statistics. Session-local grouping keeps the reference
    # local in time; a stale reference is the one alignment failure with a published
    # effect size attached to it.
    if args.align_scope == "session":
        keys = np.array([f"{s}|{sess}" for s, sess in zip(cohort.subject, cohort.session)])
        _, groups = np.unique(keys, return_inverse=True)
    else:
        groups = cohort.subject

    if args.align != "none":
        if regime.kind == "inductive":
            # No target data at all: one reference estimated from the training pool and
            # applied to everyone, target included. This is what a shipped model does.
            from eegbench.align import fit_whitener
            w = fit_whitener(X[fit_mask & in_split], mode=args.align,
                             shrinkage=args.align_shrinkage, seed=seed)
            X = w.transform(X)
            info["align"] = {"mode": args.align, "scope": "source-pooled",
                             "n_fit": w.n_fit, "rank": w.rank, "shrinkage": w.shrinkage}
        else:
            X, whit = align_by_group(
                X, groups, mode=args.align, shrinkage=args.align_shrinkage,
                fit_mask=fit_mask, subset=in_split, seed=seed)
            info["align"] = {
                "mode": args.align, "scope": args.align_scope,
                "n_groups": len(whit),
                "median_n_fit": float(np.median([w.n_fit for w in whit.values()])),
                "min_rank": int(min(w.rank for w in whit.values())),
            }

    if args.norm == "per-subject":
        if regime.kind == "inductive":
            # The target contributes nothing, so a per-subject statistic is undefined for
            # it. Use one derived from the source pool and applied unchanged -- exactly
            # what the alignment branch above does for this regime, and what a shipped
            # model does. Silently reverting to per-subject here is what made the
            # inductive arm transductive.
            X = _zscore_global(X, fit_mask & in_split)
            info["norm"] = {"kind": "source-pooled",
                            "requested": "per-subject",
                            "reason": "inductive regime admits no target trials"}
        else:
            # Same budget as alignment. Estimating the scale from the full test set after
            # budgeting the alignment reference is the exact leak this shares `fit_mask`
            # to prevent.
            X = _zscore_by_group(X, groups, fit_mask, subset=in_split)
            info["norm"] = {"kind": "per-subject", "scope": args.align_scope,
                            "budgeted_by_regime": True}
    elif args.norm == "train-fold":
        stats_idx = np.zeros(X.shape[0], dtype=bool)
        stats_idx[split.train] = True
        X = _zscore_global(X, stats_idx)
        info["norm"] = {"kind": "train-fold"}

    return X, info


def _zscore_by_group(X, groups, fit_mask, subset=None):
    """Per-group z-score, estimated only from trials the regime permits.

    A group with no permitted trials **raises**. The obvious convenience -- fall back to
    that group's own data -- is precisely the leak this function exists to prevent, and it
    was here: under ``regime=inductive`` the target contributes nothing to ``fit_mask``, so
    the fallback silently normalised the target subject using its own test trials. The run
    was then transductive through the normalization while reporting itself as the inductive
    floor, which is the one number the whole zero-shot axis is measured against.

    Inductive arms must use a source-derived statistic instead; see
    :func:`preprocess_for_split`, which routes them to :func:`_zscore_global`.
    """
    out = np.array(X, dtype=np.float32, copy=True)
    in_scope = None if subset is None else np.asarray(subset, dtype=bool)
    for g in np.unique(groups):
        m = groups == g
        # A group the split does not use has no fit data by construction and is never
        # read; see align_by_group's `subset` for why this is not a convenience.
        if in_scope is not None and not (m & in_scope).any():
            continue
        f = m & fit_mask
        if not f.any():
            raise ValueError(
                f"group {g!r} has no trials permitted for estimating its own "
                "normalization statistics under this regime. Falling back to the group's "
                "own data would make the run transductive through the z-score. Use a "
                "source-derived statistic (--norm train-fold) for fully inductive arms."
            )
        ref = X[f]
        mean = ref.mean(axis=(0, 2), keepdims=True)
        std = np.maximum(ref.std(axis=(0, 2), keepdims=True), 1e-12)
        out[m] = ((X[m] - mean) / std).astype(np.float32)
    return out


def _zscore_global(X, stats_idx):
    ref = X[stats_idx]
    mean = ref.mean(axis=(0, 2), keepdims=True)
    # Floored rather than dropped: in a union montage an absent electrode has std exactly
    # zero, and dividing by it would turn a structural zero into NaN.
    std = np.maximum(ref.std(axis=(0, 2), keepdims=True), 1e-12)
    return ((X - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eegbench.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--datasets", nargs="+", default=["cho2017"])
    d.add_argument("--paradigm", default="leftright",
                   choices=["leftright", "motor_imagery", "p300", "ssvep"])
    # Default None so the band/window can be resolved FROM THE PARADIGM, and an explicit
    # value stays distinguishable from an unset one. Hardcoding the motor-imagery numbers
    # here meant a P300 run asked for a 4-38 Hz cache while the prepared cache was 1-24 Hz
    # -- every stimulus fold failed at load. For leftright/motor_imagery the resolved
    # values are identical to the old defaults, so no existing result changes fingerprint.
    d.add_argument("--fmin", type=float, default=None)
    d.add_argument("--fmax", type=float, default=None)
    d.add_argument("--sfreq", type=float, default=250.0)
    d.add_argument("--tmin", type=float, default=None)
    d.add_argument("--tmax", type=float, default=None)
    d.add_argument("--montage", default="auto",
                   choices=["auto", "native", "union", "shared"],
                   help="'union' zero-fills absent electrodes; 'shared' keeps only "
                        "electrodes every corpus has, giving 100%% occupancy")
    d.add_argument("--channels", nargs="+", default=None,
                   help="explicit electrode list; corpora lacking any of them are "
                        "refused rather than padded. Holding this fixed is what makes a "
                        "subject-scaling curve about subjects and nothing else")
    d.add_argument("--trials-per-subject", type=int, default=None,
                   help="cap per subject, class-balanced; for the subjects-vs-trials arm")

    e = p.add_argument_group("protocol")
    e.add_argument("--protocol", default="cross-subject", choices=list(PROTOCOLS))
    e.add_argument("--folds", type=int, default=10)
    e.add_argument("--fold", type=int, default=None,
                   help="run only this fold and write a shard (enables fold-level parallelism)")
    e.add_argument("--val-frac", type=float, default=0.0)
    e.add_argument("--val-subjects", type=int, default=5)
    e.add_argument("--regime", default="transductive", help=REGIME_HELP)
    e.add_argument("--test-datasets", nargs="+", default=None,
                   help="restrict the EVALUATION population to these corpora; all other "
                        "subjects are always trained on. Required for a scaling curve to "
                        "mean anything -- otherwise growing the cohort changes the test "
                        "set too and the mean absorbs the new corpora's difficulty")

    a = p.add_argument_group("alignment / normalization")
    a.add_argument("--align", default="none", choices=["none", "euclid", "riemann"])
    a.add_argument("--align-scope", default="subject", choices=["subject", "session"])
    a.add_argument("--align-shrinkage", default="0.0",
                   help="0.0-1.0, or 'auto' for the Ledoit-Wolf estimate")
    a.add_argument("--norm", default="per-subject",
                   choices=["none", "per-subject", "train-fold"])

    m = p.add_argument_group("model")
    m.add_argument("--arm", default="ours_noattn", choices=list(ARMS))
    m.add_argument("--arm-cfg", default="{}",
                   help="JSON of encoder kwargs, e.g. '{\"spatial_filter\":88}'")

    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--label-smoothing", type=float, default=0.0)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--scheduler", default="onecycle", choices=["onecycle", "none"])
    t.add_argument("--select", default="final", choices=["final", "best-val"])
    t.add_argument("--patience", type=int, default=0)
    t.add_argument("--test-time-bn", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=25,
                   help="print training progress every N epochs (0 disables). Not part "
                        "of the config fingerprint -- it changes no number")
    t.add_argument("--device", default=None)
    t.add_argument("--null", action="store_true",
                   help="shuffle TRAINING labels only: the wiring control")

    o = p.add_argument_group("output")
    o.add_argument("--out", default=None)
    o.add_argument("--tag", default=None)
    o.add_argument("--checkpoint-dir", default=None,
                   help="persist trained weights here. Each checkpoint carries the full "
                        "arm_cfg, so eegbench.engine.load_checkpoint rebuilds the exact "
                        "architecture rather than a remembered one")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_dict = resolve_config(args)
    args.align_shrinkage = cfg_dict["align_shrinkage"]
    arm_cfg = cfg_dict["arm_cfg"]
    contract = Contract(paradigm=cfg_dict["paradigm"], fmin=cfg_dict["fmin"],
                        fmax=cfg_dict["fmax"], sfreq=cfg_dict["sfreq"],
                        tmin=cfg_dict["tmin"], tmax=cfg_dict["tmax"])

    cohort = load_cohort(args.datasets, contract, montage=args.montage,
                         channels=args.channels,
                         trials_per_subject=args.trials_per_subject, seed=args.seed)
    splits = build_splits(cohort, args.protocol, n_folds=args.folds,
                          val_frac=args.val_frac, val_subjects=args.val_subjects,
                          seed=args.seed, test_datasets=args.test_datasets)
    for s in splits:
        assert_integrity(s, cohort)

    if args.fold is not None:
        if not 0 <= args.fold < len(splits):
            raise SystemExit(f"--fold {args.fold} outside [0, {len(splits)})")
        selected = [splits[args.fold]]
    else:
        selected = splits

    fingerprint = config_fingerprint(cfg_dict)

    tcfg = TrainConfig(**{**vars(args),
                          **({"device": args.device} if args.device else {})})

    records = []
    for split in selected:
        X, info = preprocess_for_split(cohort, split, args, seed=args.seed)
        work = type(cohort)(**{**cohort.__dict__, "X": X})
        if args.null:
            # Destroy training labels only; test labels stay intact, so a correctly wired
            # pipeline must land inside the chance band. Class counts are preserved
            # because this is a permutation.
            rng = np.random.default_rng(args.seed)
            y = work.y.copy()
            y[split.train] = y[split.train][rng.permutation(split.train.size)]
            work.y = y
        if args.checkpoint_dir:
            tcfg.checkpoint_dir = args.checkpoint_dir
            tcfg.checkpoint_tag = f"{args.tag or args.arm}__{split.name.replace('/', '-')}"
        rec = run_split(work, split, args.arm, tcfg, arm_cfg)
        rec["regime_info"] = info
        records.append(rec)
        print(f"[run] {split.name}: acc={rec['accuracy']*100:.2f} "
              f"subj_mean={rec['subject_mean_accuracy']*100:.2f} "
              f"kappa={rec['kappa']:.3f} ({rec['wall_seconds']:.0f}s)",
              file=sys.stderr, flush=True)

    accs = np.array([r["subject_mean_accuracy"] for r in records]) * 100
    payload = {
        "schema": 1,
        "fingerprint": fingerprint,
        "config": cfg_dict,
        "cohort": cohort.summary(),
        "n_splits_total": len(splits),
        "fold": args.fold,
        "folds": [r for r in records],
        "summary": {
            "fold_accuracy": accs.tolist(),
            "mean": float(accs.mean()), "sd": float(accs.std(ddof=1)) if accs.size > 1 else 0.0,
            "kappa_mean": float(np.mean([r["kappa"] for r in records])),
        },
        "env": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    out = Path(args.out) if args.out else (
        RESULTS_ROOT / f"{args.tag or args.arm}"
        f"{'' if args.fold is None else f'.fold{args.fold}'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str))
    print(f"[run] {out}  mean={accs.mean():.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
