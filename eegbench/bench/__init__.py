"""Declarative sweeps, fold-level GPU scheduling, and paired statistics.

    python -m eegbench.bench run    sweeps/cross_subject.json --results results/xs
    python -m eegbench.bench report results/xs --reference ours_noattn

Launching a sweep that outlives your shell
------------------------------------------
A real sweep runs for hours, so it must not be a child of the terminal that started it.
``nohup`` alone is **not** sufficient: it suppresses SIGHUP but leaves the process in the
launching shell's process group, so anything that signals that group -- closing a session,
a tooling teardown, ``pkill -f`` on a pattern the parent also matches -- takes the whole
sweep with it, including every training job it had dispatched. That has happened here, and
the symptom is a results directory with zero shards and a log that simply stops.

Use ``setsid`` so the driver gets its own session, which its children then inherit::

    setsid nohup python -m eegbench.bench run sweeps/nopool.json \
        --results results/nopool > results/nopool_driver.log 2>&1 < /dev/null &

Verify it took: ``ps -o pid,pgid,sid -p <driver-pid>`` should show all three equal.

Interrupting one is safe regardless. Every completed ``(arm, dataset, seed, fold)`` shard
is cached and keyed on the config fingerprint, so re-running the same command resumes and
only the unfinished units are dispatched again.

``spec``       a sweep is data: arms x datasets x seeds x folds, with a shared contract
``scheduler``  dispatches (arm, dataset, seed, fold) units onto cards with measured room
``aggregate``  merges shards, pairs folds, corrects for multiple comparisons
"""

from eegbench.bench.spec import Arm, Job, Sweep, load_sweep

__all__ = ["Arm", "Job", "Sweep", "load_sweep"]
