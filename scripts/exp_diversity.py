# -*- coding: utf-8 -*-
"""M8 대응 — '증강 사본 유지' 효과에서 볼륨과 다양성을 분리 (2026-08-19).

기존 dedup ablation: train 3,567 -> 495 (content당 1파일), 100 epoch, valid mAP50 0.661 -> 0.553.
지적: 이는 볼륨 효과이지 "사전 생성 사본이 유용하다"는 증거가 아니다.

통제: 495 파일에 대해 **최적화 예산(반복 수)을 동일하게** 맞춘다.
      3,567 x 100 epoch = 356,700 image-passes = 495 x 720 epoch
      -> 720 epoch로 재학습 후 valid 비교. 차이가 남으면 그것이 '다양성' 효과.
출력: 05_results/upgrade/diversity_control.json
"""
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
CLEAN = BASE / "04_experiments" / "data" / "final0_clean"
LOCAL = Path(r"D:\runs\weld2\diversity")
OUT = BASE / "05_results" / "upgrade" / "diversity_control.json"
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
EPOCHS = 720          # 495 x 720 = 356,400 ~= 3,567 x 100
# patience=EPOCHS: 예산 일치를 위해 조기 종료 비활성화 (보고값은 best.pt = 최적 epoch)
TRAIN_KW = dict(batch=16, imgsz=640, lr0=0.01, device=0, workers=4, cache=False,
                exist_ok=True, plots=False, verbose=False, patience=EPOCHS)


def build():
    dst = LOCAL / "data"
    yaml = dst / "data.yaml"
    if yaml.exists() and (dst / "train" / "images").exists():
        n = len(list((dst / "train" / "images").iterdir()))
        if n > 400:
            print(f"reuse dataset ({n} train files)", flush=True)
            return yaml, n
    rows = list(csv.DictReader(open(CLEAN / "manifest.csv", encoding="utf-8-sig")))
    for s in ("train", "valid"):
        for sub in ("images", "labels"):
            (dst / s / sub).mkdir(parents=True, exist_ok=True)
    by_content = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by_content[r["content_id"]].append(r)
    n = 0
    for cid, rs in by_content.items():
        r = sorted(rs, key=lambda x: (len(x["orig_filename"]), x["orig_filename"]))[0]
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            src = CLEAN / "train" / sub / (r["new_id"] + ext)
            if src.exists():
                shutil.copyfile(src, dst / "train" / sub / (r["new_id"] + ext))
        n += 1
    for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
        for p in (CLEAN / "valid" / sub).iterdir():
            shutil.copyfile(p, dst / "valid" / sub / p.name)
    yaml.write_text(f"train: {dst / 'train' / 'images'}\nval: {dst / 'valid' / 'images'}\n"
                    f"test: {dst / 'valid' / 'images'}\n\nnc: 5\nnames: {NAMES}\n", encoding="utf-8")
    print(f"built dedup dataset: {n} train contents", flush=True)
    return yaml, n


def main():
    from ultralytics import YOLO
    yaml, n_train = build()
    best = LOCAL / "run720" / "weights" / "best.pt"
    if not best.exists():
        print(f"training {EPOCHS} epochs on {n_train} unique-content files ...", flush=True)
        YOLO("yolov5nu.pt").train(data=str(yaml), project=str(LOCAL), name="run720",
                                  seed=42, epochs=EPOCHS, **TRAIN_KW)
    m = YOLO(str(best))
    r = m.val(data=str(yaml), split="val", batch=16, imgsz=640, device=0, workers=0,
              verbose=False, plots=False)
    idx = r.box.ap_class_index.tolist()
    res = {"n_train_files": n_train, "epochs": EPOCHS,
           "image_passes": n_train * EPOCHS,
           "reference": {"full_3567_files_100ep_valid_mAP50_3seed": 0.6606,
                         "dedup_495_files_100ep_valid_mAP50": 0.5527,
                         "full_image_passes": 3567 * 100},
           "valid_mAP50": round(float(r.box.map50), 4),
           "valid_mAP50_95": round(float(r.box.map), 4),
           "per_class_ap50": {m.names[c]: round(float(r.box.ap50[i]), 4)
                              for i, c in enumerate(idx)}}
    res["interpretation"] = {
        "budget_matched_gap_vs_full": round(0.6606 - res["valid_mAP50"], 4),
        "note": "budget-matched gap = diversity effect; 0.108 - this = volume effect"}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
