"""Index every saved checkpoint, and verify each one still loads.

    python -m eegbench.manifest                     # write checkpoints/MANIFEST.json
    python -m eegbench.manifest --verify            # also rebuild every model
    python -m eegbench.manifest --dir other/place

``--verify`` is the point of this module rather than a nicety. A checkpoint index that
lists files without loading them is exactly the artefact that let a predecessor project
ship weights which rebuilt into a *different architecture* than the one trained. Verifying
rebuilds each model from the config stored beside its weights and checks the parameter
count, so a broken checkpoint is found here rather than downstream in the fMRI pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eegbench import REPO_ROOT

__all__ = ["build_manifest", "main"]


def build_manifest(directory: Path, verify: bool = False) -> dict:
    import torch

    entries, failures = [], []
    for p in sorted(Path(directory).glob("*.pt")):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as exc:
            failures.append({"file": p.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        e = {
            "file": p.name,
            "tag": p.stem.split("__")[0],
            "split": p.stem.split("__")[1] if "__" in p.stem else None,
            "arm": ck.get("arm"),
            "arm_cfg": ck.get("arm_cfg"),
            "n_channels": ck.get("n_channels"),
            "n_classes": ck.get("n_classes"),
            "n_times": ck.get("n_times"),
            "channels": ck.get("channels"),
            "n_parameters": ck.get("n_parameters"),
            "accuracy": ck.get("accuracy"),
            "size_mb": round(p.stat().st_size / 1e6, 2),
        }
        if verify:
            try:
                from eegbench.engine import load_checkpoint
                load_checkpoint(p)
                e["verified"] = True
            except Exception as exc:
                e["verified"] = False
                failures.append({"file": p.name,
                                 "error": f"{type(exc).__name__}: {exc}"})
        entries.append(e)

    by_tag = defaultdict(list)
    for e in entries:
        by_tag[e["tag"]].append(e)
    groups = []
    for tag, es in sorted(by_tag.items()):
        accs = [x["accuracy"] for x in es if x["accuracy"] is not None]
        groups.append({
            "tag": tag,
            "n_checkpoints": len(es),
            "arm": es[0]["arm"],
            "arm_cfg": es[0]["arm_cfg"],
            "n_channels": es[0]["n_channels"],
            "n_classes": es[0]["n_classes"],
            "n_times": es[0]["n_times"],
            "channels": es[0]["channels"],
            "n_parameters": es[0]["n_parameters"],
            "mean_accuracy": round(100 * sum(accs) / len(accs), 2) if accs else None,
        })
    return {"schema": 1, "n_checkpoints": len(entries), "groups": groups,
            "checkpoints": entries, "failures": failures}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(REPO_ROOT / "checkpoints"))
    ap.add_argument("--verify", action="store_true",
                    help="rebuild every model from its stored config; slow but the only "
                         "check that the weights and the architecture still agree")
    a = ap.parse_args(argv)

    d = Path(a.dir)
    if not d.is_dir():
        print(f"no such directory: {d}")
        return 1
    m = build_manifest(d, verify=a.verify)
    (d / "MANIFEST.json").write_text(json.dumps(m, indent=1))

    print(f"{m['n_checkpoints']} checkpoints in {d}")
    for g in m["groups"]:
        acc = f"{g['mean_accuracy']:.2f}%" if g["mean_accuracy"] is not None else "-"
        print(f"  {g['tag']:<26}{g['n_checkpoints']:>3} files  "
              f"{g['n_channels']}ch x {g['n_times']}  {g['n_classes']}-class  "
              f"{(g['n_parameters'] or 0):>9,} params  mean {acc}")
    if m["failures"]:
        print(f"\n{len(m['failures'])} FAILED:")
        for f in m["failures"]:
            print(f"  {f['file']}: {f['error']}")
        return 1
    if a.verify:
        print("\nall checkpoints rebuilt and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
