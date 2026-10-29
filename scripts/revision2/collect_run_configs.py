# -*- coding: utf-8 -*-
"""T5 (R1-4): collect the experimental configuration of every training run from the run's own args.yaml
(written by Ultralytics at training time), so the published configuration is the one actually used.
Absolute local paths are reduced to dataset/model names. Output: results/T5/runs_config.csv + summary json."""
import csv
import glob
import json
import os
from collections import Counter
from pathlib import Path

import yaml

import pathlib as _pl, sys as _sys  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
RUNS = runs_dir(r"D:\runs\weld2")
OUT = Path(__file__).resolve().parent.parent / "results" / "T5"
KEYS = ["model", "data", "epochs", "batch", "imgsz", "lr0", "seed", "workers", "cache", "patience",
        "optimizer", "momentum", "weight_decay", "warmup_epochs", "close_mosaic", "mosaic", "mixup",
        "degrees", "translate", "scale", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "deterministic", "amp"]


def dataset_name(p):
    p = str(p).replace("\\", "/")
    parts = [x for x in p.split("/") if x]
    if "aug_r2" in parts:
        return "dedup495 (exp_aug_dihedral.py)"
    if "diversity" in parts:
        return "dedup495 (exp_diversity.py)"
    for marker in ("final0_clean", "final0"):
        if marker in parts:
            i = parts.index(marker)
            return "/".join(parts[i:i + 1])
    return "/".join(parts[-3:-1]) if len(parts) > 2 else p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(str(RUNS / "*" / "args.yaml")) + glob.glob(str(RUNS / "*" / "*" / "args.yaml"))):
        a = yaml.safe_load(open(f, encoding="utf-8"))
        rid = os.path.relpath(os.path.dirname(f), RUNS).replace(os.sep, "/")
        if "smoke" in rid:
            continue
        r = {"run": rid}
        for k in KEYS:
            v = a.get(k)
            if k == "model":
                v = os.path.basename(str(v))
            elif k == "data":
                v = dataset_name(v)
            r[k] = v
        rows.append(r)
    with open(OUT / "runs_config.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["run"] + KEYS)
        w.writeheader()
        w.writerows(rows)
    summ = {k: dict(Counter(str(r[k]) for r in rows)) for k in KEYS}
    (OUT / "runs_config_summary.json").write_text(json.dumps({"n_runs": len(rows), "values": summ},
                                                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(rows), "runs")
    for k in KEYS:
        d = summ[k]
        print(f"{k:14s}", d if len(d) <= 6 else f"{len(d)} distinct")


if __name__ == "__main__":
    main()
