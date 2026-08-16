"""Declarative sweep specification: arms x datasets x seeds x folds.

A sweep is data, not a script. That is the whole point of this module, and the reason is
concrete: the shell scripts this replaces each re-implemented GPU assignment, completion
polling, tag naming and log routing, and each got at least one of them subtly wrong. One
recorded failure is worth repeating because it is so easy to write again -- a watcher that
polls with ``ps | grep <pattern>`` contains ``<pattern>`` in its own command line, so it
matches itself and waits forever.

Design rules encoded here:

**The shared contract is shared by construction.** A sweep has one ``base`` and arms may
only add to it. An arm cannot quietly change the epoch budget or the validation fraction,
because the merge refuses keys the base already fixed unless they are declared overridable.
This is what stops the most common one-sided comparison: re-running the winning arm under
better settings and pairing it against baselines that never got them.

**Every job is a (arm, dataset, seed, fold) unit.** Dispatching whole arms wastes the tail
of every sweep; dispatching folds keeps every slot busy and makes a failed fold cost one
fold rather than one arm.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Arm", "Sweep", "Job", "load_sweep"]


@dataclass(frozen=True)
class Arm:
    """One architecture or configuration under test."""

    name: str
    flags: dict = field(default_factory=dict)
    #: Free-text statement of what this arm is expected to do, written BEFORE the run.
    #: Carried into the results so a surprising outcome can be told apart from a
    #: predicted one after the fact.
    prediction: str = ""

    def resolved(self, base: dict, overridable: frozenset[str]) -> dict:
        clash = {k for k in self.flags if k in base and k not in overridable}
        if clash:
            raise ValueError(
                f"arm {self.name!r} overrides shared contract fields {sorted(clash)}. "
                "Every arm in a sweep must pay the same recipe, or the comparison "
                "measures the recipe. Add them to `overridable` if the difference IS the "
                "experiment -- and then every arm needs its own value, not just this one."
            )
        return {**base, **self.flags}


@dataclass(frozen=True)
class Job:
    """One dispatchable unit of work."""

    arm: str
    dataset_key: str
    seed: int
    fold: int
    flags: dict

    @property
    def tag(self) -> str:
        return f"{self.arm}__{self.dataset_key}__s{self.seed}__f{self.fold}"

    def argv(self, out_path: Path) -> list[str]:
        argv = ["-m", "eegbench.run", "--out", str(out_path), "--tag", self.tag,
                "--fold", str(self.fold), "--seed", str(self.seed)]
        for k, v in sorted(self.flags.items()):
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    argv.append(flag)
            elif isinstance(v, (list, tuple)):
                argv.extend([flag, *[str(x) for x in v]])
            elif isinstance(v, dict):
                argv.extend([flag, json.dumps(v)])
            else:
                argv.extend([flag, str(v)])
        return argv


@dataclass
class Sweep:
    name: str
    base: dict
    arms: list[Arm]
    datasets: list[list[str]]
    seeds: list[int] = field(default_factory=lambda: [0])
    folds: int = 10
    overridable: frozenset[str] = frozenset()
    note: str = ""

    def jobs(self) -> list[Job]:
        out: list[Job] = []
        for arm, ds, seed, fold in itertools.product(
                self.arms, self.datasets, self.seeds, range(self.folds)):
            flags = arm.resolved({**self.base, "datasets": list(ds)}, self.overridable)
            flags.setdefault("folds", self.folds)
            out.append(Job(arm=arm.name, dataset_key="+".join(ds), seed=seed,
                           fold=fold, flags=flags))
        return out

    def summary(self) -> dict:
        return {"name": self.name, "n_arms": len(self.arms),
                "n_datasets": len(self.datasets), "n_seeds": len(self.seeds),
                "folds": self.folds, "n_jobs": len(self.jobs()),
                "arms": [a.name for a in self.arms]}


def load_sweep(path: str | Path) -> Sweep:
    """Load a sweep from JSON.

    Kept to JSON rather than YAML so the harness has no dependency a results file might
    outlive, and so a sweep can be emitted by a script as easily as written by hand.
    """
    d = json.loads(Path(path).read_text())
    return Sweep(
        name=d["name"],
        base=d.get("base", {}),
        arms=[Arm(a["name"], a.get("flags", {}), a.get("prediction", ""))
              for a in d["arms"]],
        datasets=[x if isinstance(x, list) else [x] for x in d["datasets"]],
        seeds=d.get("seeds", [0]),
        folds=d.get("folds", 10),
        overridable=frozenset(d.get("overridable", ())),
        note=d.get("note", ""),
    )
