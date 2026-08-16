"""One training and evaluation loop, shared by every arm.

Two decisions here are worth defending because they cost accuracy and are kept anyway.

**Final-epoch selection is the default, not best-validation.** Selecting the best epoch by
validation accuracy on a small carve-out does not measure the model, it measures the
carve-out: with a few dozen validation trials the epoch-to-epoch noise is larger than most
architectural effects, so "best val" reliably manufactures a gain that does not survive on
test. A fixed epoch budget with final-model selection has no such degree of freedom.

**The validation carve is a real cost and every arm pays it.** With final selection there
is nothing to select on, so ``val_frac > 0`` simply removes trials from training. It is
supported because early stopping and per-epoch diagnostics need it, but ``val_frac=0.0``
is the honest default for a headline number -- and if one arm is re-run at 0.0, every arm
it is compared against must be too.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eegbench.splits import Split
from eegbench.store import Cohort

__all__ = ["TrainConfig", "run_split", "adapt_batchnorm", "set_seed",
           "save_checkpoint", "load_checkpoint"]


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _TrialSet(Dataset):
    def __init__(self, X, y, valid, subject):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        self.valid = torch.from_numpy(np.ascontiguousarray(valid, dtype=np.float32))
        self.subject = torch.from_numpy(np.ascontiguousarray(subject, dtype=np.int64))

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i], "valid": self.valid[i],
                "subject": self.subject[i]}


class TrainConfig:
    """Recipe shared by every arm. Recorded verbatim into every result."""

    def __init__(self, **kw):
        self.epochs = int(kw.get("epochs", 150))
        self.batch_size = int(kw.get("batch_size", 64))
        self.lr = float(kw.get("lr", 1e-3))
        self.weight_decay = float(kw.get("weight_decay", 1e-4))
        self.label_smoothing = float(kw.get("label_smoothing", 0.0))
        self.grad_clip = float(kw.get("grad_clip", 1.0))
        self.scheduler = kw.get("scheduler", "onecycle")
        self.select = kw.get("select", "final")          # final | best-val
        self.patience = int(kw.get("patience", 0))       # 0 = no early stopping
        # `kw.get(key, default)` returns the STORED value when the key is present, even
        # if that value is None -- and callers pass `vars(args)`, where an unset
        # `--device` is present as None. The default therefore never applied, `cfg.device`
        # was None, and `.to(None)` is a silent no-op on both the model and every batch.
        # Result: the whole harness trained on CPU while reporting nothing unusual. It is
        # the characteristic failure of this project -- not a crash, just a number (here a
        # wall-clock) that is quietly wrong by two orders of magnitude.
        requested = kw.get("device") or None
        if requested is None:
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"device={requested!r} was requested but torch.cuda.is_available() is "
                "False. Refusing to fall back to CPU silently: a cross-subject sweep on "
                "CPU is ~100x slower and looks identical to one that is merely queued."
            )
        self.device = str(requested)
        self.test_time_bn = bool(kw.get("test_time_bn", False))
        self.seed = int(kw.get("seed", 0))
        self.log_every = int(kw.get("log_every", 25))
        # Where to persist trained weights. Off by default so a sweep does not silently
        # fill a disk with 594 checkpoints, on whenever a run is meant to produce a model
        # rather than only a number.
        self.checkpoint_dir = kw.get("checkpoint_dir") or None
        self.checkpoint_tag = kw.get("checkpoint_tag") or None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@torch.no_grad()
def adapt_batchnorm(model: nn.Module, loader: DataLoader, device: str) -> int:
    """Re-estimate BatchNorm running statistics on the target subject's unlabelled trials.

    This is "BN-1" in the test-time-adaptation literature, where recomputing normalization
    statistics on the target has been reported to match or beat Euclidean Alignment while
    costing zero parameters and using no labels.

    Two things to hold in mind when reading a result from it. First, the statistics are
    reset and re-accumulated with ``momentum=None``, so they come from the target trials
    *alone* -- typically a couple of hundred, against many thousands for the statistics
    they replace. Second, stacking this on top of an already-aligned input is not the
    setting the large published gains were measured in; those compared BN-1 and alignment
    as *alternatives*. Expect substantially less on top of alignment, and do not read a
    small gain as a failure to reproduce.

    Returns the number of BatchNorm modules adapted -- zero means the architecture has
    none, which makes the arm a no-op and must not be reported as a null result.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()
            m.momentum = None      # cumulative average over what we are about to show it
            n += 1
    if n == 0:
        return 0
    model.train()
    for batch in loader:
        x = batch["x"].to(device)
        v = batch["valid"].to(device)
        _forward(model, x, v)
    model.eval()
    return n


def _log_epoch(cfg: "TrainConfig", split: Split, rec: dict, *, force: bool = False) -> None:
    """Periodic progress to stderr.

    Without it a fold emits nothing for its entire run, so a job training normally is
    indistinguishable from one that has hung. On a 200-job sweep that is the difference
    between catching a stall in minutes and catching it at the end.
    """
    ep = rec["epoch"]
    if not force and not (cfg.log_every and
                          (ep % cfg.log_every == 0 or ep == cfg.epochs - 1)):
        return
    import sys as _sys
    msg = (f"[engine] {split.name} epoch {ep + 1}/{cfg.epochs} "
           f"loss={rec['train_loss']:.4f}")
    if "val_acc" in rec:
        msg += f" val={rec['val_acc'] * 100:.2f}"
    print(msg, file=_sys.stderr, flush=True)


def _forward(model: nn.Module, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Call the model, passing the validity mask only to arms that accept one."""
    from eegbench.models import EEGTrialClassifier
    if isinstance(model, EEGTrialClassifier):
        return model(x, valid)
    # braindecode reference models take the raw tensor; mask before the call instead, so
    # an absent electrode is zero at the input rather than merely zero-filled upstream.
    return model(x * valid.unsqueeze(-1))


def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits, ys, subs = [], [], []
    with torch.no_grad():
        for b in loader:
            out = _forward(model, b["x"].to(device), b["valid"].to(device))
            logits.append(out.float().cpu())
            ys.append(b["y"])
            subs.append(b["subject"])
    return (torch.cat(logits).numpy(), torch.cat(ys).numpy(), torch.cat(subs).numpy())


def run_split(cohort: Cohort, split: Split, arm: str, cfg: TrainConfig,
              arm_cfg: dict | None = None) -> dict:
    """Train and evaluate one arm on one split. Returns a serialisable record."""
    from eegbench.models import build_arm, _SklearnArm

    set_seed(cfg.seed)
    n_classes = int(cohort.y.max()) + 1
    C, T = cohort.X.shape[1], cohort.X.shape[2]
    t0 = time.time()

    model, desc = build_arm(arm, n_channels=C, n_classes=n_classes, n_times=T,
                            cfg=arm_cfg or {})

    if isinstance(model, _SklearnArm):
        return _run_sklearn(model, cohort, split, desc, t0)

    device = cfg.device
    model = model.to(device)

    def loader(idx, shuffle):
        return DataLoader(
            _TrialSet(cohort.X[idx], cohort.y[idx], cohort.valid[idx], cohort.subject[idx]),
            batch_size=cfg.batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

    train_dl = loader(split.train, True)
    val_dl = loader(split.val, False) if split.val.size else None
    test_dl = loader(split.test, False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = None
    if cfg.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=max(1, cfg.epochs * len(train_dl)))
    lossf = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    history, best = [], {"acc": -1.0, "state": None, "epoch": -1}
    stale = 0
    for ep in range(cfg.epochs):
        model.train()
        tot, seen = 0.0, 0
        for b in train_dl:
            x, y = b["x"].to(device), b["y"].to(device)
            v = b["valid"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(_forward(model, x, v), y)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            tot += float(loss.detach()) * y.numel()
            seen += y.numel()
        rec = {"epoch": ep, "train_loss": tot / max(seen, 1)}
        if val_dl is not None:
            lg, yy, _ = _evaluate(model, val_dl, device)
            rec["val_acc"] = float((lg.argmax(1) == yy).mean())
            if cfg.select == "best-val" and rec["val_acc"] > best["acc"]:
                best = {"acc": rec["val_acc"],
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()},
                        "epoch": ep}
                stale = 0
            else:
                stale += 1
            if cfg.patience and stale >= cfg.patience:
                history.append(rec)
                _log_epoch(cfg, split, rec, force=True)
                break
        history.append(rec)
        _log_epoch(cfg, split, rec)

    if cfg.select == "best-val":
        if best["state"] is None:
            raise ValueError(
                "--select best-val requires a non-empty validation set, but val_frac left "
                "it empty. Selection would silently fall back to the final epoch while the "
                "result recorded 'best-val'."
            )
        model.load_state_dict(best["state"])

    n_adapted = None
    if cfg.test_time_bn:
        # Fitted on the TEST loader deliberately: the statistics are a label-free property
        # of the target subject's signal. No label is read, and the loader is the only
        # source of that subject's data at inference time.
        n_adapted = adapt_batchnorm(model, test_dl, device)

    logits, y_true, subs = _evaluate(model, test_dl, device)
    rec = _score(logits, y_true, subs, cohort, split, desc, cfg, t0,
                 history=history, n_bn_adapted=n_adapted,
                 best_epoch=best["epoch"] if cfg.select == "best-val" else None)

    if cfg.checkpoint_dir:
        rec["checkpoint"] = save_checkpoint(
            model, cfg.checkpoint_dir, cfg.checkpoint_tag or f"{arm}_{split.name}",
            arm=arm, arm_cfg=arm_cfg or {}, n_channels=C, n_classes=n_classes,
            n_times=T, channels=list(cohort.channels), split=split.name,
            accuracy=rec["subject_mean_accuracy"])
    return rec


def save_checkpoint(model, directory, tag, **meta) -> str:
    """Write weights plus **everything needed to rebuild the architecture**.

    A bare ``state_dict`` is not a checkpoint. This project's predecessor stored one and
    rebuilt it by passing only the *published* constructor arguments, so a model trained
    with a different pooling mode or token budget loaded under ``strict=True`` without
    complaint and silently ran a different architecture -- measured at chance agreement,
    then written into a results file. The full ``arm_cfg`` therefore travels with the
    weights, and :func:`load_checkpoint` reconstructs from that rather than from anything
    the caller remembers.
    """
    from pathlib import Path
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{tag}.pt"
    payload = {
        "schema": 1,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "n_parameters": sum(p.numel() for p in model.parameters()),
        **meta,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return str(path)


def load_checkpoint(path, device: str | None = None, strict: bool = True):
    """Rebuild the exact architecture from a checkpoint and load its weights.

    Returns ``(model, meta)``. Loads with ``strict=True`` by default: a key mismatch means
    the rebuilt architecture is not the one that was trained, and tolerating that silently
    is the failure this format exists to prevent.
    """
    from eegbench.models import build_arm
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("schema") != 1:
        raise ValueError(f"{path}: unknown checkpoint schema {ck.get('schema')!r}")
    model, desc = build_arm(ck["arm"], n_channels=ck["n_channels"],
                            n_classes=ck["n_classes"], n_times=ck["n_times"],
                            cfg=ck.get("arm_cfg") or {})
    model.load_state_dict(ck["state_dict"], strict=strict)
    got = sum(p.numel() for p in model.parameters())
    if got != ck["n_parameters"]:
        raise ValueError(
            f"{path}: rebuilt model has {got:,} parameters but the checkpoint recorded "
            f"{ck['n_parameters']:,}. The architecture was not reconstructed correctly."
        )
    if device:
        model = model.to(device)
    model.eval()
    meta = {k: v for k, v in ck.items() if k != "state_dict"}
    meta["arch"] = desc
    return model, meta


def _run_sklearn(model, cohort, split, desc, t0) -> dict:
    pipe = model.pipeline()
    Xtr = cohort.X[split.train].astype(np.float64)
    Xte = cohort.X[split.test].astype(np.float64)
    pipe.fit(Xtr, cohort.y[split.train])
    pred = pipe.predict(Xte)
    try:
        proba = pipe.predict_proba(Xte)
    except (AttributeError, NotImplementedError):
        proba = None
    n_classes = int(cohort.y.max()) + 1
    logits = proba if proba is not None else np.eye(n_classes)[pred.astype(int)]
    return _score(np.asarray(logits), cohort.y[split.test], cohort.subject[split.test],
                  cohort, split, desc, None, t0)


def _score(logits, y_true, subs, cohort, split, desc, cfg, t0, **extra) -> dict:
    from eegbench.metrics import (accuracy, balanced_accuracy, cohen_kappa,
                                  macro_f1, roc_auc)

    pred = logits.argmax(1)
    per_subject = {}
    for s in np.unique(subs):
        m = subs == s
        ds, native = cohort.subject_index[int(s)]
        per_subject[f"{ds}/s{native}"] = {
            "accuracy": float(accuracy(y_true[m], pred[m])),
            "kappa": float(cohen_kappa(y_true[m], pred[m])),
            "n": int(m.sum()),
        }
    out = {
        "split": split.summary(),
        "arch": desc,
        "accuracy": float(accuracy(y_true, pred)),
        "kappa": float(cohen_kappa(y_true, pred)),
        "macro_f1": float(macro_f1(y_true, pred)),
        # Reported always, and the HEADLINE score whenever classes are imbalanced: on a
        # 1:5 paradigm a constant predictor scores ~83% plain accuracy while decoding
        # nothing, and 50% balanced.
        "balanced_accuracy": float(balanced_accuracy(y_true, pred)),
        # The mean over subjects, not over trials. These differ whenever subjects
        # contribute unequal trial counts, and the subject mean is the one a
        # cross-subject claim is about -- a trial mean lets one prolific subject
        # dominate a fold.
        "subject_mean_accuracy": float(np.mean([v["accuracy"] for v in per_subject.values()])),
        "per_subject": per_subject,
        "class_counts": np.bincount(y_true, minlength=int(y_true.max()) + 1).tolist(),
        "wall_seconds": round(time.time() - t0, 2),
    }
    if logits.shape[1] == 2:
        sc = logits[:, 1] - logits[:, 0]
        out["roc_auc"] = float(roc_auc(y_true, sc))
        per_auc = {}
        for s_ in np.unique(subs):
            m = subs == s_
            ds, native = cohort.subject_index[int(s_)]
            per_auc[f"{ds}/s{native}"] = float(roc_auc(y_true[m], sc[m]))
        out["per_subject_auc"] = per_auc
    out.update({k: v for k, v in extra.items() if v is not None})
    if cfg is not None:
        out["train_config"] = cfg.to_dict()
    return out
