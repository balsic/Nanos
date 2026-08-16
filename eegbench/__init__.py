"""Benchmark and evaluation harness for the EEG encoder in ``encoder/model.py``.

Scope
-----
This package owns everything *around* the encoder: dataset preparation, cohort
assembly, split construction, alignment, reference baselines, the training loop, and
the parallel benchmark runner. The encoder itself is treated as the object under test.

Two questions it exists to answer, and they are not the same question:

* **Cross-subject motor imagery.** Does the trunk decode a held-out subject better than
  the reference architectures (EEGNet, MSVTNet, ...) under one identical recipe? This is
  a number, and it is comparable to published work.
* **Zero-shot transfer.** How much of that survives when the target subject contributes
  no labels, few unlabelled trials, or a different montage? This is a *curve*, and it is
  the question that actually governs deployment.

The harness is built so the second is measurable rather than asserted: see
:mod:`eegbench.align` for the calibration axis and :mod:`eegbench.protocols` for the
regime taxonomy.

Layout
------
``registry``    dataset specifications (MOABB name, montage, event vocabulary)
``prepare``     MOABB -> cached per-subject trial tensors
``store``       cache discovery, cohort assembly, union montage
``splits``      protocols; worker-independent so folds pair across arms
``align``       Euclidean / Riemannian alignment and the calibration-limited variants
``models``      arm factory: the encoder under test plus braindecode references
``engine``      one training/evaluation loop, shared by every arm
``run``         single-arm entry point (``python -m eegbench.run``)
``bench``       declarative sweeps, GPU scheduling, paired statistics
"""

from __future__ import annotations

import pathlib

__all__ = ["REPO_ROOT", "DATA_ROOT", "PREPARED_ROOT", "MOABB_ROOT", "RESULTS_ROOT"]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

#: Per-subject trial tensors, written by :mod:`eegbench.prepare`.
PREPARED_ROOT = DATA_ROOT / "prepared"

#: Raw MOABB/MNE downloads. Pointed at by ``MNE_DATA`` when preparing.
MOABB_ROOT = DATA_ROOT / "moabb"

RESULTS_ROOT = REPO_ROOT / "results"
