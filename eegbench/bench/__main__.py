"""CLI for the benchmark runner.

    python -m eegbench.bench run    <sweep.json> --results results/xs [--dry-run]
    python -m eegbench.bench report <results-dir> [--reference ARM] [--json]
    python -m eegbench.bench gpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eegbench.bench.aggregate import (collect, compare_to_reference, leaderboard,
                                      seed_average)
from eegbench.bench.scheduler import GpuPool, Scheduler, gpu_status
from eegbench.bench.spec import load_sweep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eegbench.bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="execute a sweep")
    r.add_argument("sweep")
    r.add_argument("--results", required=True)
    r.add_argument("--gpus", nargs="*", type=int, default=None,
                   help="restrict to these cards; default is every card with room")
    r.add_argument("--job-mb", type=int, default=4000,
                   help="memory a single job is assumed to need, for admission")
    r.add_argument("--max-per-gpu", type=int, default=2)
    r.add_argument("--max-util", type=int, default=101,
                   help="skip cards busier than this percent; 101 disables the check")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--no-resume", action="store_true")

    p = sub.add_parser("report", help="merge shards and print the leaderboard")
    p.add_argument("results")
    p.add_argument("--reference", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--per-seed", action="store_true",
                   help="report each seed as its own row instead of averaging within "
                        "fold. Diagnostic only -- it shows how much of the spread is "
                        "seed noise, and its paired tests compare one seed against one "
                        "seed, which is not the pre-registered unit.")

    sub.add_parser("gpus", help="show measured GPU capacity")

    args = ap.parse_args(argv)

    if args.cmd == "gpus":
        states = gpu_status()
        if not states:
            print("no GPUs visible (nvidia-smi unavailable) -- jobs will run on CPU")
            return 0
        print(f"{'gpu':>4}{'free MiB':>11}{'util %':>9}")
        for g in states:
            print(f"{g.index:>4}{g.free_mb:>11}{g.util:>9}")
        return 0

    if args.cmd == "run":
        sweep = load_sweep(args.sweep)
        print(f"[bench] {json.dumps(sweep.summary())}", file=sys.stderr)
        pool = GpuPool(visible=args.gpus, job_mb=args.job_mb,
                       max_per_gpu=args.max_per_gpu, max_util=args.max_util)
        sched = Scheduler(sweep, Path(args.results), pool=pool,
                          dry_run=args.dry_run, resume=not args.no_resume)
        summary = sched.run()
        print(json.dumps(summary, indent=1))
        return 1 if summary.get("failed") else 0

    results = collect(Path(args.results))
    if not results:
        print(f"no usable shards under {args.results}/shards", file=sys.stderr)
        return 1
    # Seeds are averaged within fold by default. Reporting them raw would let
    # compare_to_reference pick an arbitrary seed and present it as the result.
    n_seeds = len({r.seed for r in results.values()})
    if not args.per_seed:
        results = seed_average(results)
    elif n_seeds > 1:
        print(f"[agg] --per-seed: {n_seeds} seeds shown separately; the paired tests below "
              "compare one seed against one seed and are diagnostic, not the result.",
              file=sys.stderr)
    if args.json:
        payload = {
            "arms": [{"arm": r.arm, "dataset": r.dataset, "seed": r.seed,
                      "n_seeds": r.n_seeds,
                      "mean": r.mean, "sd": r.sd, "n_folds": len(r.fold_acc),
                      "fold_accuracy": r.fold_acc, "n_params": r.n_params}
                     for r in results.values()],
        }
        if args.reference:
            payload["paired"] = compare_to_reference(
                results, args.reference, dataset=args.dataset)
        print(json.dumps(payload, indent=1))
        return 0
    print(leaderboard(results, reference=args.reference, dataset=args.dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
