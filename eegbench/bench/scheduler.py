"""GPU-aware job scheduler: dispatch fold-level units, resume, never poll by grep.

Three properties, each replacing a specific failure of the shell scripts this supersedes.

**Completion is observed by process handle, never by pattern matching.** A watcher that
polls ``ps | grep -- --run-tag foo`` has ``--run-tag foo`` in its own command line and so
matches itself, waiting forever for a job that already finished. That deadlock is on
record. ``Popen.poll()`` cannot express the bug.

**Admission is by measured free memory, not by a GPU count.** This machine routinely
carries foreign work -- at the time of writing, all eight cards were 65-100% busy with
other jobs, with free memory ranging from 7 GB to 16 GB. A scheduler handed ``--gpus
0..7`` would launch eight jobs onto cards that cannot hold them and take eight
out-of-memory failures. Each card is admitted only when it has ``reserve_mb`` free, and
its budget is decremented by the job's own estimate on dispatch so two jobs are not
admitted against the same free memory.

**Resume is keyed on the config fingerprint, not on the filename.** A shard whose
fingerprint matches is skipped; a shard whose fingerprint differs is *reported and
re-run*, because a stale result silently compared against fresh ones is how a sweep ends
up describing an experiment nobody ran.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from eegbench.bench.spec import Job, Sweep

__all__ = ["GpuPool", "Scheduler", "shard_path", "gpu_status"]


# ---------------------------------------------------------------------------------------
# GPU capacity
# ---------------------------------------------------------------------------------------
@dataclass
class GpuState:
    index: int
    free_mb: int
    util: int
    #: Memory this scheduler has already promised to jobs it launched but that have not
    #: yet allocated. Without it, several jobs are admitted against the same free memory
    #: in the seconds before any of them touches the card.
    reserved_mb: int = 0

    @property
    def available_mb(self) -> int:
        return max(0, self.free_mb - self.reserved_mb)


def gpu_status() -> list[GpuState]:
    """Query real free memory and utilization. Empty list if there is no GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    states = []
    for line in out.strip().splitlines():
        try:
            idx, used, total, util = [int(x.strip()) for x in line.split(",")]
        except ValueError:
            continue
        states.append(GpuState(index=idx, free_mb=total - used, util=util))
    return states


class GpuPool:
    """Admits jobs onto cards that can actually hold them."""

    def __init__(self, visible: list[int] | None = None, *, job_mb: int = 4000,
                 max_per_gpu: int = 2, max_util: int = 101, refresh: float = 20.0):
        self.visible = visible
        self.job_mb = job_mb
        self.max_per_gpu = max_per_gpu
        self.max_util = max_util
        self.refresh = refresh
        self._states: dict[int, GpuState] = {}
        self._running: dict[int, int] = {}
        self._last = 0.0

    def _sync(self, force: bool = False) -> None:
        if not force and (time.time() - self._last) < self.refresh:
            return
        self._last = time.time()
        fresh = {g.index: g for g in gpu_status()
                 if self.visible is None or g.index in self.visible}
        for idx, g in fresh.items():
            g.reserved_mb = self._states[idx].reserved_mb if idx in self._states else 0
            self._states[idx] = g

    def acquire(self) -> int | None:
        """Index of a card with room, or ``None``. CPU-only returns ``-1`` once."""
        self._sync()
        if not self._states:
            return -1 if not self._running.get(-1) else None
        best, best_free = None, -1
        for idx, g in self._states.items():
            if self._running.get(idx, 0) >= self.max_per_gpu:
                continue
            if g.util > self.max_util:
                continue
            if g.available_mb < self.job_mb:
                continue
            if g.available_mb > best_free:
                best, best_free = idx, g.available_mb
        if best is None:
            return None
        self._states[best].reserved_mb += self.job_mb
        self._running[best] = self._running.get(best, 0) + 1
        return best

    def release(self, idx: int) -> None:
        self._running[idx] = max(0, self._running.get(idx, 1) - 1)
        if idx in self._states:
            self._states[idx].reserved_mb = max(
                0, self._states[idx].reserved_mb - self.job_mb)
        # The card's true free memory changed; re-read rather than trust the estimate.
        self._last = 0.0


# ---------------------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------------------
def shard_path(results_dir: Path, job: Job) -> Path:
    return results_dir / "shards" / f"{job.tag}.json"


class Scheduler:
    def __init__(self, sweep: Sweep, results_dir: Path, *, pool: GpuPool | None = None,
                 logs_dir: Path | None = None, dry_run: bool = False,
                 resume: bool = True, python: str | None = None):
        self.sweep = sweep
        self.results_dir = Path(results_dir)
        self.logs_dir = Path(logs_dir or self.results_dir / "logs")
        self.pool = pool or GpuPool()
        self.dry_run = dry_run
        self.resume = resume
        self.python = python or sys.executable
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "shards").mkdir(exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # -- resume ---------------------------------------------------------------------
    def _already_done(self, job: Job) -> tuple[bool, str]:
        p = shard_path(self.results_dir, job)
        if not p.exists():
            return False, ""
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return False, "unreadable shard, re-running"
        if not d.get("folds"):
            return False, "shard has no fold record, re-running"
        want = self._fingerprint(job)
        got = d.get("fingerprint")
        if got != want:
            return False, f"fingerprint {got} != {want}, re-running (config changed)"
        return True, ""

    def _fingerprint(self, job: Job) -> str:
        """What ``eegbench.run`` will hash for this job, derived the same way it does.

        Routed through the same parser and the same ``resolve_config`` the run itself
        uses, so the two cannot drift. They did once: the scheduler re-implemented the
        argument canonicalization, disagreed on one field, and every cached shard was
        discarded as stale on every re-run.
        """
        from eegbench.run import build_parser, config_fingerprint, resolve_config
        args = build_parser().parse_args(job.argv(Path("/dev/null"))[2:])
        return config_fingerprint(resolve_config(args))

    # -- run ------------------------------------------------------------------------
    def run(self, poll: float = 5.0) -> dict:
        jobs = self.sweep.jobs()
        pending, skipped, restale = [], 0, []
        for j in jobs:
            if self.resume:
                done, why = self._already_done(j)
                if done:
                    skipped += 1
                    continue
                if why:
                    restale.append((j.tag, why))
            pending.append(j)

        for tag, why in restale:
            print(f"[bench] STALE {tag}: {why}", file=sys.stderr)
        print(f"[bench] {self.sweep.name}: {len(jobs)} jobs, {skipped} cached, "
              f"{len(pending)} to run", file=sys.stderr)

        if self.dry_run:
            for j in pending[:10]:
                print(f"[bench] would run {j.tag}: "
                      f"{' '.join(j.argv(shard_path(self.results_dir, j)))}",
                      file=sys.stderr)
            if len(pending) > 10:
                print(f"[bench] ... and {len(pending) - 10} more", file=sys.stderr)
            return {"planned": len(pending), "cached": skipped, "dry_run": True}

        running: list[tuple[Job, subprocess.Popen, int, object, float]] = []
        failed, completed = [], 0
        t0 = time.time()

        while pending or running:
            while pending:
                gpu = self.pool.acquire()
                if gpu is None:
                    break
                job = pending.pop(0)
                out = shard_path(self.results_dir, job)
                env = {**os.environ}
                if gpu >= 0:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                else:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                # Threads are capped because dozens of concurrent numpy/torch processes
                # each defaulting to all cores is how a scheduler starves itself.
                env.setdefault("OMP_NUM_THREADS", "2")
                env.setdefault("MKL_NUM_THREADS", "2")
                lf = (self.logs_dir / f"{job.tag}.log").open("w")
                proc = subprocess.Popen([self.python, *job.argv(out)],
                                        stdout=lf, stderr=lf, env=env,
                                        cwd=str(Path(__file__).resolve().parents[2]))
                running.append((job, proc, gpu, lf, time.time()))
                print(f"[bench] start {job.tag} gpu{gpu}", file=sys.stderr, flush=True)

            time.sleep(poll)

            for item in list(running):
                job, proc, gpu, lf, started = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                self.pool.release(gpu)
                lf.close()
                dt = time.time() - started
                if proc.returncode == 0:
                    completed += 1
                    print(f"[bench] done  {job.tag} ({dt:.0f}s) "
                          f"[{completed}/{len(jobs) - skipped}]",
                          file=sys.stderr, flush=True)
                else:
                    failed.append((job.tag, proc.returncode))
                    print(f"[bench] FAIL  {job.tag} rc={proc.returncode} -- see "
                          f"{self.logs_dir / (job.tag + '.log')}",
                          file=sys.stderr, flush=True)

        return {
            "sweep": self.sweep.name,
            "total": len(jobs), "cached": skipped,
            "completed": completed, "failed": [t for t, _ in failed],
            "wall_seconds": round(time.time() - t0, 1),
        }
