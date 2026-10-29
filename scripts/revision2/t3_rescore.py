# -*- coding: utf-8 -*-
"""T3 (R1-2): re-score the frozen gate-D models (YOLOv5nu seeds 42-44, best.pt) on test subsets that exclude
images with a same-photo counterpart across splits. Same evaluator as the paper's -0.007 analysis
(04_experiments/scripts/bootstrap_ultra.py: ultralytics m.val, split=test, workers=0, overall = unweighted
mean of the 3-seed per-class means). No model is selected or trained; the sealed test images are only
re-aggregated, as in Section 4.2 "Integrity checks".

Subsets
  full71        sanity: must reproduce 0.576
  excl12_paper  sanity: the paper's 12 residual-duplicate images -> must reproduce about -0.007
  excl31        29 visually confirmed same-scene (U6) + 2 tile-rule images
  excl33        excl31 + 2 UNSURE
  only31        the 31 excluded images alone (are they easier?)
Output: results/T3/rescore.json
"""
import json
import shutil
from pathlib import Path
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir, runs_dir  # noqa: E402,F401  (path defaults; see weldpaths.py)
_T3 = Path(__file__).resolve().parents[2] / "results" / "revision2" / "T3"  # was work/results/T3

import numpy as np

BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
CLEAN = BASE / "04_experiments" / "data" / "final0_clean"
TEST = CLEAN / "test"
T3 = _T3
WORK = runs_dir(r"D:\runs\weld2") / "aug_r2" / "rescore"
RUNS = {"E2_v5n": 42, "E3_v5n_s43": 43, "E3_v5n_s44": 44}
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]


def subsets():
    tally = json.loads((T3 / "tally_claude.json").read_text(encoding="utf-8"))
    tile = json.loads((T3 / "tile_fix.json").read_text(encoding="utf-8"))
    imp = json.loads((_T3.parents[1] / "phash_impact.json").read_text(encoding="utf-8"))
    allids = sorted(p.stem for p in (TEST / "images").glob("*.jpg"))
    e12 = set(imp["affected_clean_files"]["test"]["ids"])
    e31 = set(tally["U6"]["same_scene_test_ids"]) | set(tile["final0_clean"]["test_images_affected"])
    e33 = e31 | set(tally["U6"]["unsure_test_ids"])
    return {"full71": allids,
            "excl12_paper": [i for i in allids if i not in e12],
            "excl31": [i for i in allids if i not in e31],
            "excl33": [i for i in allids if i not in e33],
            "only31": sorted(e31)}, {"e12": sorted(e12), "e31": sorted(e31), "e33": sorted(e33)}


def build(name, ids):
    d = WORK / name
    if d.exists():
        shutil.rmtree(d)
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)
    inst = {c: 0 for c in NAMES}
    for i in ids:
        shutil.copyfile(TEST / "images" / f"{i}.jpg", d / "images" / f"{i}.jpg")
        shutil.copyfile(TEST / "labels" / f"{i}.txt", d / "labels" / f"{i}.txt")
        for ln in (TEST / "labels" / f"{i}.txt").read_text().splitlines():
            if ln.strip():
                inst[NAMES[int(ln.split()[0])]] += 1
    y = d / "data.yaml"
    y.write_text(f"train: {CLEAN / 'train' / 'images'}\nval: {CLEAN / 'valid' / 'images'}\n"
                 f"test: {d / 'images'}\n\nnc: 5\nnames: {NAMES}\n", encoding="utf-8")
    return y, inst


def main(dry=False):
    subs, sets = subsets()
    out = {"evaluator": "ultralytics m.val split=test workers=0 (as bootstrap_ultra.py)",
           "models": {k: str(runs_dir(r"D:\runs\weld2") / k / "weights" / "best.pt") for k in RUNS},
           "excluded_sets": sets, "subsets": {}}
    models = None
    if not dry:
        from ultralytics import YOLO
        models = {rid: YOLO(out["models"][rid]) for rid in RUNS}
    for name, ids in subs.items():
        y, inst = build(name, ids)
        rec = {"n_images": len(ids), "instances": inst}
        if models:
            per = {}
            for rid, m in models.items():
                r = m.val(data=str(y), split="test", batch=16, imgsz=640, device=0, workers=0,
                          verbose=False, plots=False)
                per[rid] = {"mAP50": round(float(r.box.map50), 4),
                            "per_class": {m.names[ci]: round(float(r.box.ap50[k]), 4)
                                          for k, ci in enumerate(r.box.ap_class_index.tolist())}}
            pc = {}
            for c in NAMES:
                v = [per[r]["per_class"][c] for r in RUNS if c in per[r]["per_class"]]
                pc[c] = {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4)} if v else None
            ov = [per[r]["mAP50"] for r in RUNS]
            rec.update(per_run=per, per_class_3seed=pc,
                       overall_3seed={"mean": round(float(np.mean(ov)), 4), "sd": round(float(np.std(ov, ddof=1)), 4)})
        out["subsets"][name] = rec
        print(name, json.dumps({k: v for k, v in rec.items() if k != "per_run"}, ensure_ascii=False))
    if models:
        f = out["subsets"]["full71"]["overall_3seed"]["mean"]
        for k in ("excl12_paper", "excl31", "excl33", "only31"):
            out["subsets"][k]["delta_vs_full"] = round(out["subsets"][k]["overall_3seed"]["mean"] - f, 4)
        (T3 / "rescore.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    import sys
    main(dry="--dry" in sys.argv)
