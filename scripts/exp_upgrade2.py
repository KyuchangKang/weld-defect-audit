# -*- coding: utf-8 -*-
"""V2·V3·V4 학습 검증 체인 (evaluation_plan v1.2, 사용자 승인 2026-08-14).

V2: 상위 구성 다중 시드 검증 — v5mu·v12m 시드 43/44 (valid 평가만, 재선정 아님)
V3: 조건 (a) 분포화 — raw 학습 v5nu 시드 43/44 (원본 test 평가)
V4: 중복 제거 학습 ablation — 부모당 1콘텐츠 train, v5nu 시드 42 (valid 평가만)
봉인된 clean test는 미접촉. 순차 실행, 단계별 저장(재개 가능).
출력: 05_results/upgrade/upgrade2_results.json
"""
import csv
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
DATA = BASE / "04_experiments" / "data"
CLEAN = DATA / "final0_clean"
CLEAN_YAML = CLEAN / "data.yaml"
RAW_YAML = DATA / "final0" / "data.yaml"
LOCAL = Path(r"D:\runs\weld2\upgrade2")
OUT = BASE / "05_results" / "upgrade" / "upgrade2_results.json"
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
# workers 8→4 (2026-08-14): pinned host memory 고갈로 크래시 2회 — RAM 16GB 제약 강화 대응
TRAIN_KW = dict(epochs=100, batch=16, imgsz=640, lr0=0.01, device=0,
                workers=4, cache=False, exist_ok=True, plots=False, verbose=False)


def ep(p):
    return "\\\\?\\" + str(p)


def build_dedup_dataset():
    """final0_clean과 동일 배정을 유지하되 train을 콘텐츠당 1파일로 축소."""
    dst = LOCAL / "dedup_data"
    if (dst / "data.yaml").exists():
        return dst / "data.yaml"
    for s in ("train", "valid", "test"):
        (dst / s / "images").mkdir(parents=True, exist_ok=True)
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(MANIFEST := CLEAN / "manifest.csv",
                                    encoding="utf-8-sig")))
    # valid/test: final0_clean 그대로 복사
    for s in ("valid", "test"):
        for img in (CLEAN / s / "images").iterdir():
            shutil.copyfile(ep(img), ep(dst / s / "images" / img.name))
        for lbl in (CLEAN / s / "labels").iterdir():
            shutil.copyfile(ep(lbl), ep(dst / s / "labels" / lbl.name))
    # train: content_id당 대표 1개 (짧은 원본명 우선 — build_clean2와 동일 규칙)
    by_content = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by_content[r["content_id"]].append(r)
    n = 0
    for cid, rs in by_content.items():
        r = sorted(rs, key=lambda x: (len(x["orig_filename"]), x["orig_filename"]))[0]
        n += 1
        src_img = CLEAN / "train" / "images" / (r["new_id"] + ".jpg")
        src_lbl = CLEAN / "train" / "labels" / (r["new_id"] + ".txt")
        shutil.copyfile(ep(src_img), ep(dst / "train" / "images" / f"{r['new_id']}.jpg"))
        shutil.copyfile(ep(src_lbl), ep(dst / "train" / "labels" / f"{r['new_id']}.txt"))
    (dst / "data.yaml").write_text(
        f"train: {dst / 'train' / 'images'}\nval: {dst / 'valid' / 'images'}\n"
        f"test: {dst / 'test' / 'images'}\n\nnc: 5\nnames: {NAMES}\n", encoding="utf-8")
    print(f"[V4] dedup train contents: {n}")
    return dst / "data.yaml"


def evaluate(weights, data_yaml, split, tag):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data=str(data_yaml), split=split, batch=16, imgsz=640, device=0,
              workers=4, verbose=False, plots=False)
    ap50 = {m.names[ci]: round(float(r.box.ap50[i]), 4)
            for i, ci in enumerate(r.box.ap_class_index.tolist())}
    res = {"mAP50": round(float(r.box.map50), 4),
           "mAP50_95": round(float(r.box.map), 4), "per_class_ap50": ap50}
    print(f"[{tag}] {split} mAP50={res['mAP50']:.4f}")
    return res


def train(model_name, data_yaml, run_name, seed):
    from ultralytics import YOLO
    m = YOLO(model_name)
    m.train(data=str(data_yaml), project=str(LOCAL), name=run_name, seed=seed,
            **TRAIN_KW)
    return LOCAL / run_name / "weights" / "best.pt"


def save(results):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")


JOBS = [
    # (key, model, data, seed, eval_data, eval_split)
    ("V2_v5m_s43", "yolov5mu.pt", "clean", 43, "clean", "val"),
    ("V2_v5m_s44", "yolov5mu.pt", "clean", 44, "clean", "val"),
    ("V2_v12m_s43", "yolo12m.pt", "clean", 43, "clean", "val"),
    ("V2_v12m_s44", "yolo12m.pt", "clean", 44, "clean", "val"),
    ("V3_raw_s43", "yolov5nu.pt", "raw", 43, "raw", "test"),
    ("V3_raw_s44", "yolov5nu.pt", "raw", 44, "raw", "test"),
    ("V4_dedup_s42", "yolov5nu.pt", "dedup", 42, "dedup", "val"),
]


def main():
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    yamls = {"clean": CLEAN_YAML, "raw": RAW_YAML}
    for key, model, data, seed, edata, esplit in JOBS:
        if key in results:
            continue
        if data == "dedup" or edata == "dedup":
            yamls["dedup"] = build_dedup_dataset()
        w = train(model, yamls[data], key, seed)
        results[key] = {"seed": seed, "model": model,
                        **evaluate(w, yamls[edata], esplit, key)}
        save(results)

    # 요약 (기존 실측과 합산)
    v5m = [0.6791, results["V2_v5m_s43"]["mAP50"], results["V2_v5m_s44"]["mAP50"]]
    v12m = [0.6735, results["V2_v12m_s43"]["mAP50"], results["V2_v12m_s44"]["mAP50"]]
    v5n = [0.6808, 0.6439, 0.6571]
    a3 = [0.8313, results["V3_raw_s43"]["mAP50"], results["V3_raw_s44"]["mAP50"]]

    def ms(v):
        return {"mean": round(statistics.mean(v), 4),
                "std": round(statistics.stdev(v), 4), "values": v}

    results["summary"] = {
        "V2_stage1_3seed_valid": {"v5mu": ms(v5m), "yolo12m": ms(v12m)},
        "V2_stage2_3seed_valid": {"v5nu": ms(v5n), "v5mu": ms(v5m)},
        "V3_a_3seed_raw_test": ms(a3),
        "V4_dedup_valid_mAP50": results["V4_dedup_s42"]["mAP50"],
        "V4_baseline_valid_3seed": ms(v5n),
    }
    save(results)
    print(json.dumps(results["summary"], indent=1))


if __name__ == "__main__":
    main()
