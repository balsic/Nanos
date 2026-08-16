"""Merge shards, run paired tests, and print a leaderboard.

What makes this different from sorting a column
-----------------------------------------------
Folds are **paired**: every arm sees the same subjects in the same fold, guaranteed by
:mod:`eegbench.splits`. That pairing is where the statistical power is at n=10, and using
it is not optional -- an unpaired comparison of two arms whose fold-to-fold spread is
6-8 points cannot resolve the 1-3 point effects that separate real architectures.

Three guards, each against a mistake this kind of table invites:

**Fingerprint agreement is checked before anything is compared.** Two shards may only be
paired if their controlled configuration matches on everything except the arm. A stale
shard from an earlier sweep is otherwise indistinguishable from a fresh one.

**Folds are aligned by fold index, and mismatched sets are truncated with a warning
rather than silently zipped.** Comparing arm A's folds 0-9 against arm B's folds 0-6 by
position pairs different subjects.

**Multiple comparisons are corrected.** A sweep with twelve arms against one reference
gets twelve chances to produce p < 0.05; reporting the raw p-values would manufacture a
result roughly half the time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["ArmResult", "collect", "paired_test", "leaderboard", "compare_to_reference"]


@dataclass
class ArmResult:
    arm: str
    dataset: str
    seed: int
    fold_acc: dict[int, float]          # fold index -> subject-mean accuracy (%)
    fingerprint: str
    config: dict
    n_params: int | None = None
    #: How many seeds this record averages. 1 for a record straight out of ``collect``;
    #: >1 only for the merged records ``seed_average`` produces.
    n_seeds: int = 1

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.arm, self.dataset, self.seed)

    @property
    def mean(self) -> float:
        return float(np.mean(list(self.fold_acc.values()))) if self.fold_acc else float("nan")

    @property
    def sd(self) -> float:
        v = list(self.fold_acc.values())
        return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def collect(results_dir: Path) -> dict[tuple[str, str, int], ArmResult]:
    """Merge every shard under ``results_dir/shards`` into per-arm records."""
    out: dict[tuple[str, str, int], ArmResult] = {}
    shards = sorted((Path(results_dir) / "shards").glob("*.json"))
    for p in shards:
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            print(f"[agg] skipping unreadable shard {p.name}")
            continue
        if not d.get("folds"):
            continue
        cfg = d.get("config", {})
        # The tag encodes the identity; parsing it rather than re-deriving keeps the
        # scheduler and the aggregator agreeing on what a job was.
        stem = p.stem
        try:
            arm, dataset, seed_s, fold_s = stem.split("__")
            seed, fold = int(seed_s.lstrip("s")), int(fold_s.lstrip("f"))
        except ValueError:
            print(f"[agg] skipping shard with unparseable name {p.name}")
            continue
        acc = float(d["folds"][0]["subject_mean_accuracy"]) * 100
        k = (arm, dataset, seed)
        if k not in out:
            out[k] = ArmResult(arm=arm, dataset=dataset, seed=seed, fold_acc={},
                               fingerprint=d.get("fingerprint", ""), config=cfg,
                               n_params=(d["folds"][0].get("arch") or {}).get("n_parameters"))
        out[k].fold_acc[fold] = acc
    return out


def seed_average(results: dict) -> dict:
    """Collapse the seed axis by averaging **within** fold.

    ``collect`` keys on ``(arm, dataset, seed)``, so a five-seed results directory yields
    five separate records per arm. That is the right storage shape and the wrong reporting
    shape: :func:`compare_to_reference` takes ``refs[0]`` -- whichever seed the dict
    happened to yield first -- and then filters every other arm to match it, so a
    multi-seed report describes one arbitrary seed and says so nowhere.

    Averaging within fold rather than pooling ``(seed, fold)`` pairs is the pre-registered
    choice (PAPER_PLAN_SECTION9.md): five seeds of fold 3 are five measurements of one
    fold, not five folds. Pooling them would hand the Wilcoxon 50 units where the
    experiment supports 10, and shrink every p-value by a factor nothing earned.

    The merged record carries ``seed=-1`` and ``config["seed"]=None`` so two merged records
    compare equal under :func:`_comparable`, which checks ``seed`` among ``CONTROLLED``.
    With one seed on disk this is the identity apart from that relabelling.
    """
    grouped: dict[tuple[str, str], list[ArmResult]] = {}
    for r in results.values():
        grouped.setdefault((r.arm, r.dataset), []).append(r)

    merged: dict[tuple[str, str, int], ArmResult] = {}
    for (arm, dataset), rs in grouped.items():
        per_fold: dict[int, list[float]] = {}
        for r in rs:
            for f, v in r.fold_acc.items():
                per_fold.setdefault(f, []).append(v)
        counts = {len(v) for v in per_fold.values()}
        if len(counts) > 1:
            print(f"[agg] {arm}/{dataset}: ragged seed coverage {sorted(counts)} across "
                  "folds -- folds are averaged over unequal seed counts, so this arm's "
                  "mean mixes seed depths. Finish the sweep before reporting it.")
        merged[(arm, dataset, -1)] = ArmResult(
            arm=arm, dataset=dataset, seed=-1,
            fold_acc={f: float(np.mean(v)) for f, v in per_fold.items()},
            fingerprint="seed-avg", config={**rs[0].config, "seed": None},
            n_params=rs[0].n_params, n_seeds=len(rs),
        )
    return merged


def _comparable(a: ArmResult, b: ArmResult) -> tuple[bool, str]:
    """Two arms are comparable when everything controlled agrees except the arm."""
    from eegbench.run import CONTROLLED
    for field in CONTROLLED:
        if field in ("arm", "arm_cfg"):
            continue
        if a.config.get(field) != b.config.get(field):
            return False, (f"{field}: {a.arm}={a.config.get(field)!r} vs "
                           f"{b.arm}={b.config.get(field)!r}")
    return True, ""


def paired_test(a: dict[int, float], b: dict[int, float]) -> dict:
    """Wilcoxon signed-rank over folds present in both, plus the plain sign count.

    Wilcoxon rather than a t-test because ten folds give no useful check on normality and
    the fold distribution is visibly skewed by which subjects land in each. The sign count
    is reported alongside because it is what a reader can verify by eye, and because a
    small mean difference carried by 9 of 10 folds means something different from the same
    difference carried by 5.
    """
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return {"n": len(common), "p": float("nan"), "wins": 0,
                "note": "too few shared folds for a paired test"}
    va = np.array([a[f] for f in common])
    vb = np.array([b[f] for f in common])
    d = va - vb
    wins = int((d > 0).sum())
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(va, vb).pvalue) if np.any(d != 0) else 1.0
    except ImportError:
        p = float("nan")
    return {"n": len(common), "delta": float(d.mean()), "wins": wins,
            "losses": int((d < 0).sum()), "p": p}


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values. Controls family-wise error, no independence
    assumption -- which matters here because arms share folds and are correlated."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def compare_to_reference(results: dict, reference: str, *, dataset: str | None = None,
                         seed: int | None = None) -> list[dict]:
    """Paired comparison of every arm against ``reference``, Holm-corrected."""
    refs = [r for r in results.values() if r.arm == reference
            and (dataset is None or r.dataset == dataset)
            and (seed is None or r.seed == seed)]
    if not refs:
        raise KeyError(f"reference arm {reference!r} not found in results")
    ref = refs[0]

    rows, pvals = [], []
    for r in sorted(results.values(), key=lambda x: -x.mean):
        if r.arm == reference and r.dataset == ref.dataset and r.seed == ref.seed:
            continue
        if r.dataset != ref.dataset or r.seed != ref.seed:
            continue
        ok, why = _comparable(r, ref)
        if not ok:
            rows.append({"arm": r.arm, "mean": r.mean, "sd": r.sd,
                         "incomparable": why})
            continue
        t = paired_test(r.fold_acc, ref.fold_acc)
        missing = sorted(set(ref.fold_acc) ^ set(r.fold_acc))
        row = {"arm": r.arm, "mean": r.mean, "sd": r.sd, "n_params": r.n_params, **t}
        if missing:
            row["fold_mismatch"] = missing
        rows.append(row)
        pvals.append(t["p"] if np.isfinite(t["p"]) else 1.0)

    testable = [r for r in rows if "p" in r]
    for row, adj in zip(testable, _holm(pvals)):
        row["p_holm"] = adj
    return rows


def leaderboard(results: dict, *, reference: str | None = None,
                dataset: str | None = None) -> str:
    lines = []
    by_ds: dict[str, list[ArmResult]] = {}
    for r in results.values():
        if dataset and r.dataset != dataset:
            continue
        by_ds.setdefault(r.dataset, []).append(r)

    for ds, arms in sorted(by_ds.items()):
        lines.append(f"\n=== {ds} ===")
        arms.sort(key=lambda r: -r.mean)
        nfold = {len(r.fold_acc) for r in arms}
        lines.append(f"{'arm':<20}{'seed':>5}{'acc':>8}{'sd':>7}{'folds':>7}{'params':>11}")
        lines.append("-" * 58)
        for r in arms:
            # A merged record has no single seed; show the depth it averages instead, so a
            # seed-averaged table is never mistaken for seed 0's.
            seed = f"avg{r.n_seeds}" if r.seed < 0 else str(r.seed)
            lines.append(f"{r.arm:<20}{seed:>5}{r.mean:>8.2f}{r.sd:>7.2f}"
                         f"{len(r.fold_acc):>7}"
                         f"{(f'{r.n_params:,}' if r.n_params else '-'):>11}")
        if len(nfold) > 1:
            lines.append(f"  ! arms have different fold counts {sorted(nfold)}: the means "
                         "above are NOT comparable until every arm has the same folds")
        # The same hazard one axis over. Mid-sweep, an arm whose seeds have all landed is
        # an average of 5 and its rival an average of 1 -- same estimand, very different
        # variance, and the avg5 arm's sd is smaller for a reason that has nothing to do
        # with the architecture. The paired test below will still run, so say it plainly.
        nseed = {r.n_seeds for r in arms}
        if len(nseed) > 1:
            lines.append(f"  ! arms average different seed counts {sorted(nseed)} (see the "
                         "seed column): the deltas below compare estimates of unequal "
                         "precision. Not reportable until the sweep finishes.")

        if reference:
            try:
                rows = compare_to_reference(results, reference, dataset=ds)
            except KeyError:
                continue
            lines.append(f"\n  paired vs {reference}:")
            lines.append(f"  {'arm':<20}{'delta':>8}{'wins':>10}{'p':>10}{'p_holm':>10}")
            lines.append("  " + "-" * 56)
            for row in rows:
                if "incomparable" in row:
                    lines.append(f"  {row['arm']:<20}  INCOMPARABLE: {row['incomparable']}")
                    continue
                w = f"{row['wins']}/{row['n']}"
                lines.append(f"  {row['arm']:<20}{row['delta']:>+8.2f}{w:>10}"
                             f"{row['p']:>10.4f}{row.get('p_holm', float('nan')):>10.4f}")
                if "fold_mismatch" in row:
                    lines.append(f"    ! folds not shared with reference: "
                                 f"{row['fold_mismatch']}")
    return "\n".join(lines)
