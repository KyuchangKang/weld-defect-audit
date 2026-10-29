# -*- coding: utf-8 -*-
"""R4 OOD 사후 재집계 (evaluation_plan v1.1, 사용자 승인 2026-08-13).

게이트 D의 동결된 모델 3시드(best.pt 불변)를 test 부분집합으로 재채점:
  - field57: OOD 14장 제외한 현장 사진 57장
  - ood14:   OOD 14장만 (대조)
새 추론이지만 모델·라벨·선택 불변 = 동일 측정의 재집계. GPU는 R1 체인이 사용 중이므로 CPU 실행.
출력: 05_results/upgrade/ood_reanalysis.json
"""
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

from weldpaths import project_dir, runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
CLEAN = BASE / "04_experiments" / "data" / "final0_clean"
LOCAL = runs_dir(r"D:\runs\weld2") / "upgrade"
OUT = BASE / "05_results" / "upgrade" / "ood_reanalysis.json"

NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
OOD_IDS = ["00045", "00046", "00047", "00048", "00049", "00050", "00051", "00053",
           "00054", "00060", "00061", "00062", "00065", "00068"]
RUNS = {"E2_v5n": 42, "E3_v5n_s43": 43, "E3_v5n_s44": 44}


def build_subset(name, keep_pred):
    dst = LOCAL / f"ood_{name}"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir(parents=True)
    inst = Counter()
    n = 0
    for img in sorted((CLEAN / "test" / "images").iterdir()):
        sid = img.stem.split("_")[-1]
        if not keep_pred(sid):
            continue
        n += 1
        shutil.copyfile(img, dst / "images" / img.name)
        lbl = CLEAN / "test" / "labels" / (img.stem + ".txt")
        text = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
        (dst / "labels" / (img.stem + ".txt")).write_text(text, encoding="utf-8")
        for ln in text.strip().splitlines():
            parts = ln.split()
            if len(parts) == 5:
                inst[NAMES[int(parts[0])]] += 1
    yaml = dst / "data.yaml"
    yaml.write_text(
        f"train: {CLEAN / 'train' / 'images'}\nval: {CLEAN / 'valid' / 'images'}\n"
        f"test: {dst / 'images'}\n\nnc: 5\nnames: {NAMES}\n", encoding="utf-8")
    return yaml, n, dict(inst)


def main():
    from ultralytics import YOLO

    subsets = {}
    subsets["field57"] = build_subset("field", lambda s: s not in OOD_IDS)
    subsets["ood14"] = build_subset("ood", lambda s: s in OOD_IDS)
    out = {"ood_ids": OOD_IDS, "subsets": {}}
    for name, (yaml, n, inst) in subsets.items():
        out["subsets"][name] = {"n_images": n, "instances": inst, "per_run": {}}
        print(f"== {name}: {n} images, instances={inst}")

    for rid, seed in RUNS.items():
        w = runs_dir(r"D:\runs\weld2") / rid / "weights" / "best.pt"
        m = YOLO(str(w))
        for name, (yaml, n, inst) in subsets.items():
            r = m.val(data=str(yaml), split="test", batch=4, imgsz=640, device="cpu",
                      workers=2, verbose=False, plots=False)
            ap50 = {m.names[ci]: round(float(r.box.ap50[i]), 4)
                    for i, ci in enumerate(r.box.ap_class_index.tolist())}
            out["subsets"][name]["per_run"][rid] = {
                "seed": seed, "mAP50_present_classes": round(float(r.box.map50), 4),
                "per_class_ap50": ap50}
            print(f"[{rid} {name}] mAP50(present)={float(r.box.map50):.4f} {ap50}")

    for name in out["subsets"]:
        agg = {}
        for cls in NAMES:
            vals = [pr["per_class_ap50"].get(cls)
                    for pr in out["subsets"][name]["per_run"].values()]
            vals = [v for v in vals if v is not None]
            if vals:
                agg[cls] = {"AP50_mean": round(statistics.mean(vals), 4),
                            "AP50_std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                            "n_instances": out["subsets"][name]["instances"].get(cls, 0)}
            else:
                agg[cls] = {"AP50_mean": None, "note": "no instances in subset",
                            "n_instances": out["subsets"][name]["instances"].get(cls, 0)}
        out["subsets"][name]["agg_3seed"] = agg

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
