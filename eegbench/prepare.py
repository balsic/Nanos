"""MOABB -> per-subject trial tensors, cached under ``data/prepared``.

    python -m eegbench.prepare --dataset kumar2024 --paradigm leftright
    python -m eegbench.prepare --dataset cho2017 --subjects 1 2 3 --dry-run

Uses MOABB's *paradigm* API rather than ``dataset.get_data()``. The paradigm owns band-pass
filtering, epoching around the cue and event selection, so going around it would mean
re-implementing three things that must match the literature exactly for any comparison to
mean anything.

Two conventions this module pins down, because both have silently corrupted data before:

**Units.** MOABB's paradigm pipeline ends by multiplying by ``dataset.unit_factor``, which
defaults to 1e6 -- so ``get_data`` returns **microvolts** while the raw recordings are in
volts. Everything downstream here is in volts, so the factor is undone and the result is
range-checked. Band-passed scalp EEG sits around 1e-6..1e-5 V; the accepted window is three
orders of magnitude either side of that, wide enough never to false-positive and tight
enough to catch a factor-of-1e6 error.

**Channel counts are validated against the registry.** A spec that disagrees with the data
is a hard failure, not a warning. This is deliberately strict and it has a cost -- five
datasets in the registry were recorded as "unpreparable" when in fact only their channel
literal was wrong -- but the alternative is a plausible tensor built on a wrong assumption
about what the electrodes are.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eegbench import MOABB_ROOT, PREPARED_ROOT
from eegbench.registry import REGISTRY, get as get_spec
from eegbench.store import Contract, canonical_channel

__all__ = ["prepare", "main"]

#: Plausible range for band-passed scalp EEG, in volts.
VOLT_RANGE = (1e-8, 1e-3)


def _pin_download_root() -> None:
    """Force every MOABB/MNE download and lookup into this project's ``data/moabb``.

    Setting ``MNE_DATA`` alone is **not** sufficient, and the way it fails is silent.
    MNE resolves a dataset's location through ``MNE_DATASETS_<SIGN>_PATH`` and only falls
    back to ``MNE_DATA`` when that key is absent -- and those per-dataset keys persist in
    ``~/.mne/mne-python.json``, where any earlier project on the same machine will have
    written its own paths. The result is a project that believes it owns its corpus while
    actually reading from somewhere else entirely: preparation succeeds, the data is real,
    and nothing indicates that deleting the other directory would break this one.

    ``mne.get_config`` consults the environment before the config file, so exporting the
    per-dataset keys overrides them without editing a shared user-level file that other
    work on this machine depends on.

    The sign is not always the class name -- Cho2017 downloads under GIGADB, PhysionetMI
    under EEGBCI, GrosseWentrup2009 under MUNICHMI -- so rather than deriving it, every
    key MNE already knows about is redirected, plus the generic fallback.
    """
    MOABB_ROOT.mkdir(parents=True, exist_ok=True)
    root = str(MOABB_ROOT)
    os.environ["MNE_DATA"] = root
    os.environ["MOABB_RESULTS"] = root
    try:
        import mne
        for key in mne.get_config():
            if key.startswith("MNE_DATASETS_") and key.endswith("_PATH"):
                os.environ[key] = root
    except Exception:
        # Redirecting is best-effort; a missing config file is the normal fresh-machine
        # case and MNE_DATA covers it.
        pass


def _contract_dirname(contract: Contract) -> str:
    """Directory name encoding the contract, readable rather than hashed.

    A hash would make a cache miss invisible -- the loader looks somewhere that does not
    exist and reports "not prepared", and the natural response is to regenerate data that
    already exists under slightly different settings. A readable name makes the mismatch
    obvious from ``ls``. The sidecar JSON remains the authority; this is navigation.
    """
    return (f"f{contract.fmin:g}-{contract.fmax:g}"
            f"_sr{contract.sfreq:g}"
            f"_t{contract.tmin:g}-{contract.tmax:g}")


def _build_paradigm(contract: Contract, spec):
    from moabb.paradigms import LeftRightImagery, MotorImagery

    kw = dict(fmin=contract.fmin, fmax=contract.fmax, resample=contract.sfreq,
              tmin=contract.tmin, tmax=contract.tmax)
    if contract.paradigm == "leftright":
        return LeftRightImagery(**kw)
    if contract.paradigm == "motor_imagery":
        return MotorImagery(n_classes=len(spec.events), events=list(spec.events), **kw)
    if contract.paradigm == "p300":
        from moabb.paradigms import P300
        # No baseline: the window starts at the stimulus, and MOABB's baseline argument
        # expects a pre-stimulus interval that tmin=0.0 does not provide. The 1 Hz
        # high-pass already removes the drift a baseline would correct.
        return P300(**kw)
    if contract.paradigm == "ssvep":
        from moabb.paradigms import SSVEP
        # n_classes=None keeps every stimulation frequency the dataset defines rather
        # than silently truncating to the first few, which would make the class count a
        # property of MOABB's default instead of the corpus.
        return SSVEP(n_classes=None, **kw)
    raise ValueError(f"unknown paradigm {contract.paradigm!r}")


def _build_dataset(spec):
    import moabb.datasets as mds

    cls = getattr(mds, spec.moabb_name, None)
    if cls is None:
        raise ValueError(
            f"moabb has no dataset {spec.moabb_name!r}. Installed version may predate it."
        )
    # Some datasets take arguments that materially change what is returned -- Lee2019_MI's
    # `test_run` defaults to False and silently halves the trials. Pass the full-data
    # choice explicitly where the parameter exists rather than accepting a default whose
    # effect is invisible in the output.
    import inspect
    params = inspect.signature(cls).parameters
    kw = {}
    if "test_run" in params:
        kw["test_run"] = False
    if "sessions" in params:
        pass  # take every session
    return cls(**kw)


def _to_volts(X: np.ndarray, dataset) -> tuple[np.ndarray, float]:
    """Undo MOABB's unit_factor and verify the result is plausibly volts."""
    factor = float(getattr(dataset, "unit_factor", 1e6) or 1.0)
    Xv = (X / factor).astype(np.float32)
    med = float(np.median(np.abs(Xv)))
    lo, hi = VOLT_RANGE
    if not (lo <= med <= hi):
        raise ValueError(
            f"median |X| = {med:.3e} after dividing by unit_factor={factor:g}, which is "
            f"outside the plausible volt range {VOLT_RANGE}. Either the dataset reports a "
            "different unit convention or the scaling is wrong -- both produce a tensor "
            "that trains without complaint and cannot be compared to anything."
        )
    return Xv, med


def _channel_names(paradigm, dataset, subject, n_expected: int) -> list[str]:
    """Electrode labels for the channels the paradigm returned.

    Falls back to positional names only as a last resort, and says so in the sidecar:
    positional names cannot be matched across corpora, so a dataset carrying them may be
    used alone but must never join a union montage (``store`` refuses this).
    """
    try:
        raw = dataset.get_data([subject])
        for _, sessions in raw.items():
            for _, runs in sessions.items():
                for _, r in runs.items():
                    picks = [i for i, t in enumerate(r.get_channel_types())
                             if t == "eeg"]
                    names = [r.info["ch_names"][i] for i in picks]
                    if len(names) == n_expected:
                        return [canonical_channel(n) for n in names]
                    # Length disagreement means the paradigm dropped or reordered
                    # channels; guessing an alignment here is how a montage silently
                    # becomes wrong.
                    break
            break
    except Exception:
        pass
    return [f"ch{i}" for i in range(n_expected)]


def prepare(dataset_key: str, contract: Contract | None = None, *,
            subjects: list[int] | None = None, out_root: Path | None = None,
            overwrite: bool = False, dry_run: bool = False) -> dict:
    """Prepare one dataset. Returns a summary; per-subject failures are recorded, not fatal."""
    contract = contract or Contract()
    spec = get_spec(dataset_key)
    out_root = Path(out_root or PREPARED_ROOT)

    if contract.paradigm == "leftright" and not spec.has_leftright():
        raise ValueError(f"{dataset_key} does not define left_hand/right_hand")
    if not spec.supports(contract.tmin, contract.tmax):
        raise ValueError(
            f"{dataset_key} supports at most {spec.min_post_cue}s post-cue; the contract "
            f"asks for {contract.tmax}s. Pooling needs one window, so widen the corpus "
            "selection rather than shortening this dataset alone."
        )

    _pin_download_root()

    # Applied explicitly, and only for the corpora that need it, so the shim's scope is
    # visible at the call site rather than hidden in an import.
    if spec.moabb_name in ("GrosseWentrup2009",):
        from eegbench._compat import patch_pymatreader_opaque
        if not patch_pymatreader_opaque():
            print(f"[prepare] WARNING: {dataset_key} needs the pymatreader shim and it "
                  "could not be applied; expect a read failure", file=sys.stderr)

    dest = out_root / dataset_key / contract.paradigm / _contract_dirname(contract)
    dest.mkdir(parents=True, exist_ok=True)

    dataset = _build_dataset(spec)
    paradigm = _build_paradigm(contract, spec)
    todo = subjects if subjects is not None else list(dataset.subject_list)
    want_T = contract.n_times()

    if dry_run:
        return {"dataset": dataset_key, "dest": str(dest), "subjects": todo,
                "n_times": want_T, "dry_run": True}

    written, skipped = [], []
    ch_names: list[str] | None = None
    measured_channels: int | None = None

    for subj in todo:
        path = dest / f"sub-{subj:03d}.npz"
        if path.exists() and not overwrite:
            written.append(subj)
            continue
        try:
            X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj])
            Xv, med = _to_volts(np.asarray(X), dataset)

            if spec.n_channels is None:
                # First contact with this corpus: there is no prior expectation to
                # violate, so the measurement becomes the expectation. Printed in
                # paste-ready form because the guard only earns its keep once the
                # registry states a number that a later run can contradict.
                if measured_channels is None:
                    measured_channels = int(Xv.shape[1])
                    print(f"[prepare] {dataset_key}: MEASURED n_channels="
                          f"{measured_channels} (registry had None). Add this to "
                          f"eegbench/registry.py so drift can be detected from now on.",
                          file=sys.stderr, flush=True)
                elif Xv.shape[1] != measured_channels:
                    raise ValueError(
                        f"got {Xv.shape[1]} channels but subject {todo[0]} of this same "
                        f"corpus gave {measured_channels}. Subjects disagree on the "
                        "montage; they cannot share one tensor."
                    )
            elif Xv.shape[1] != spec.n_channels:
                raise ValueError(
                    f"got {Xv.shape[1]} channels, registry says {spec.n_channels}. "
                    "Measure it and correct the registry -- do not relax this check."
                )
            # Resampling can leave one sample of slack; trim, never pad.
            if Xv.shape[2] != want_T:
                if abs(Xv.shape[2] - want_T) > max(2, int(0.01 * want_T)):
                    raise ValueError(
                        f"got {Xv.shape[2]} samples, contract implies {want_T}"
                    )
                Xv = Xv[:, :, :want_T]

            classes = list(contract.class_names(spec))
            code = {c: i for i, c in enumerate(classes)}
            yi = np.array([code[str(v)] for v in y], dtype=np.int64)

            if ch_names is None:
                ch_names = _channel_names(paradigm, dataset, subj, Xv.shape[1])

            # Explicit unicode dtype, not object. MOABB returns metadata as pandas
            # columns, and `np.asarray(series.astype(str))` yields dtype=object -- which
            # `savez` stores as a pickled array, and the loader reads with
            # `allow_pickle=False` (correctly: a cache file should never be able to
            # execute code on load). The result is data that writes fine and cannot be
            # read back.
            def _labels(key):
                if key in meta:
                    return np.asarray(meta[key].astype(str).to_numpy(), dtype="<U16")
                return np.zeros(len(yi), dtype="<U16")

            sess, run = _labels("session"), _labels("run")

            # Written to a temp file and renamed, so an interrupted run leaves no
            # half-written cache entry that a later run would happily load.
            #
            # The handle is opened explicitly rather than passing a path: numpy's savez
            # appends ".npz" to any filename that does not already end in it, so a
            # "sub-001.npz.tmp" path silently becomes "sub-001.npz.tmp.npz" and the rename
            # then fails on a file that was never there.
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, X=Xv, y=yi, session=sess, run=run,
                                    valid_mask=np.ones(Xv.shape[1], dtype=bool))
            tmp.replace(path)

            (dest / f"sub-{subj:03d}.json").write_text(json.dumps({
                "subject": int(subj),
                "config": {"dataset": dataset_key, "paradigm": contract.paradigm,
                           "fmin": contract.fmin, "fmax": contract.fmax,
                           "sfreq": contract.sfreq, "tmin": contract.tmin,
                           "tmax": contract.tmax},
                "channels": ch_names,
                "positional_channel_names": bool(ch_names and ch_names[0].startswith("ch")),
                "class_names": classes,
                "class_counts": [int((yi == i).sum()) for i in range(len(classes))],
                "n_trials": int(Xv.shape[0]), "n_channels": int(Xv.shape[1]),
                "n_samples": int(Xv.shape[2]),
                "sessions": {s: int((sess == s).sum()) for s in sorted(set(sess.tolist()))},
                "unit": "V", "median_abs_volts": med,
                "env": {"moabb": __import__("moabb").__version__},
            }, indent=1, sort_keys=True))

            written.append(subj)
            print(f"[prepare] {dataset_key} sub-{subj:03d}: {Xv.shape} "
                  f"median|X|={med:.3e}V classes={[int((yi==i).sum()) for i in range(len(classes))]}",
                  file=sys.stderr, flush=True)
        except Exception as exc:  # one bad subject must not lose the other 51
            skipped.append((subj, f"{type(exc).__name__}: {exc}"))
            print(f"[prepare] {dataset_key} sub-{subj:03d} SKIPPED: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if os.environ.get("EEGBENCH_TRACE"):
                traceback.print_exc()

    if skipped:
        (dest / "_skipped.json").write_text(json.dumps(
            {"skipped": [[s, m] for s, m in skipped]}, indent=1))
    out = {"dataset": dataset_key, "dest": str(dest),
           "written": written, "skipped": skipped}
    if spec.n_channels is None and measured_channels is not None:
        out["measured_n_channels"] = measured_channels
        out["action_required"] = (
            f"set n_channels={measured_channels} for {dataset_key!r} in "
            "eegbench/registry.py")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eegbench.prepare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--paradigm", default="leftright",
                    choices=["leftright", "motor_imagery", "p300", "ssvep"])
    ap.add_argument("--defaults", action="store_true",
                    help="use the band/window appropriate to the paradigm rather than the "
                         "motor-imagery contract (see Contract.for_paradigm)")
    # Default None, not the motor-imagery numbers, so an explicit value is
    # distinguishable from an unset one. Without that, --defaults could not tell whether
    # tmax=3.0 was the user's choice or argparse's, and silently overrode the user.
    ap.add_argument("--fmin", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--subjects", nargs="*", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    over = {k: v for k, v in (("fmin", a.fmin), ("fmax", a.fmax),
                              ("tmin", a.tmin), ("tmax", a.tmax)) if v is not None}
    if a.defaults:
        contract = Contract.for_paradigm(a.paradigm, sfreq=a.sfreq, **over)
        print(f"[prepare] {a.paradigm} contract: {contract.fmin}-{contract.fmax} Hz, "
              f"{contract.tmin}-{contract.tmax} s -> {contract.n_times()} samples"
              + (f" (overrides: {sorted(over)})" if over else ""), file=sys.stderr)
    else:
        base = dict(fmin=4.0, fmax=38.0, tmin=0.5, tmax=3.0)
        contract = Contract(paradigm=a.paradigm, sfreq=a.sfreq, **{**base, **over})
    summary = prepare(a.dataset, contract, subjects=a.subjects,
                      overwrite=a.overwrite, dry_run=a.dry_run)
    print(json.dumps(summary, indent=1, default=str))
    return 1 if summary.get("skipped") else 0


if __name__ == "__main__":
    raise SystemExit(main())
