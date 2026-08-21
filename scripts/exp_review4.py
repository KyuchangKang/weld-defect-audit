# -*- coding: utf-8 -*-
"""4차 리뷰(Opus5) 대응 재분석 3종 — 추가 학습 없이 동결 예측 재채점만 수행.

A (M7) leaky test 이분: 조건 (a) 모델 고정, 원 test 310장을 "부모가 train에 존재(seen)" /
        "존재하지 않음(unseen)"으로 나눠 AP50 비교 -> 구성 교란 없이 누수 효과만 남김
B (M6) mAP50-95: clean test 3시드 클래스별 mAP50-95 산출 (Table 3 열 추가용)
C (M9) 경계 라벨 최악 시나리오: 감사가 표시한 경계/약근거 라벨 8건(Geometric 7 + crack 1)을
        전부 '결함 없음'으로 간주(GT 박스 삭제)하고 3시드 재채점

출력: 05_results/upgrade/review4_reanalysis.json
"""
import json
import re
import shutil
import statistics
from pathlib import Path

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
DATA = BASE / "04_experiments" / "data"
RAW = DATA / "final0"
CLEAN = DATA / "final0_clean"
LOCAL = Path(r"D:\runs\weld2\review4")
RUNS = Path(r"D:\runs\weld2")
OUT = BASE / "05_results" / "upgrade" / "review4_reanalysis.json"
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
CLEAN_SEEDS = {"E2_v5n": 42, "E3_v5n_s43": 43, "E3_v5n_s44": 44}
RAW_BEST = RUNS / "E4_v5n_raw" / "weights" / "best.pt"
BORDERLINE = ["00010", "00019", "00023", "00031", "00032", "00044", "00046", "00071"]


def source_stem(name):
    s = re.sub(r"\.rf\.[0-9a-f]{32}.*$", "", name)
    s = re.sub(r"_(jpeg|jpg|png|bmp)(_(jpeg|jpg|png|bmp))*$", "", s, flags=re.I)
    return s.lower()


def content_id(stem):
    p, pats = stem, [r"[-_]+flip$", r"[-_]+(0|90|180|270)[-_]*$", r"[-_]+rot(0|90|180|270)$",
                     r"[-_]+copy(-\d+)?$", r"[-_]+\(\d+\)$", r"[-_]+aug\d*$"]
    changed = True
    while changed:
        changed = False
        for pat in pats:
            new = re.sub(pat, "", p)
            if new != p and len(new) >= 3:
                p, changed = new, True
    return p


def parent_id(name):
    return re.sub(r"_block_\d+_\d+$", "", content_id(source_stem(name)))


def write_yaml(path, test_dir):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"train: {CLEAN / 'train' / 'images'}\nval: {CLEAN / 'valid' / 'images'}\n"
                    f"test: {test_dir}\n\nnc: 5\nnames: {NAMES}\n", encoding="utf-8")
    return path


def evaluate(weights, yaml_path, tag):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data=str(yaml_path), split="test", batch=16, imgsz=640, device=0,
              workers=0, verbose=False, plots=False)
    idx = r.box.ap_class_index.tolist()
    res = {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
           "per_class_ap50": {m.names[c]: round(float(r.box.ap50[i]), 4) for i, c in enumerate(idx)},
           "per_class_map50_95": {m.names[c]: round(float(r.box.maps[c]), 4) for c in idx}}
    print(f"[{tag}] mAP50={res['mAP50']:.4f} mAP50-95={res['mAP50_95']:.4f}", flush=True)
    return res


def build_subset(stems, src_split, dst, drop_labels=()):
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir(parents=True)
    for st in stems:
        shutil.copyfile(src_split / "images" / (st + ".jpg"), dst / "images" / (st + ".jpg"))
        lbl = src_split / "labels" / (st + ".txt")
        if lbl.exists():
            txt = "" if st in drop_labels else lbl.read_text()
            (dst / "labels" / (st + ".txt")).write_text(txt)
    return dst


def ms(v):
    return {"mean": round(statistics.mean(v), 4),
            "std": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0}


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    # ---------- A. leaky test 이분 (M7) ----------
    train_parents = {parent_id(p.name) for p in (RAW / "train" / "images").iterdir()}
    seen, unseen = [], []
    for p in sorted((RAW / "test" / "images").iterdir()):
        (seen if parent_id(p.name) in train_parents else unseen).append(p.stem)
    print(f"[A] leaky test: seen={len(seen)} unseen={len(unseen)}", flush=True)
    a = {"n_seen": len(seen), "n_unseen": len(unseen)}
    for tag, stems in (("seen", seen), ("unseen", unseen)):
        if not stems:
            continue
        d = build_subset(stems, RAW / "test", LOCAL / f"test_{tag}")
        a[tag] = evaluate(RAW_BEST, write_yaml(LOCAL / f"{tag}.yaml", d / "images"),
                          f"A/{tag} n={len(stems)}")
    res["A_leaky_test_bisection"] = a
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- B. clean test mAP50-95 (M6) ----------
    yaml_clean = write_yaml(LOCAL / "clean.yaml", CLEAN / "test" / "images")
    b = {}
    for rid, seed in CLEAN_SEEDS.items():
        b[f"seed{seed}"] = evaluate(RUNS / rid / "weights" / "best.pt", yaml_clean, f"B/seed{seed}")
    b["per_class_map50_95_3seed"] = {
        c: ms([b[f"seed{s}"]["per_class_map50_95"][c] for s in CLEAN_SEEDS.values()])
        for c in NAMES}
    b["overall_map50_95_3seed"] = ms([b[f"seed{s}"]["mAP50_95"] for s in CLEAN_SEEDS.values()])
    res["B_map50_95"] = b
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- C. 경계 라벨 최악 시나리오 (M9) ----------
    test_stems = sorted(p.stem for p in (CLEAN / "test" / "images").iterdir())
    drop = {s for s in test_stems if any(bid in s for bid in BORDERLINE)}
    print(f"[C] borderline images dropped: {len(drop)} -> {sorted(drop)}", flush=True)
    d = build_subset(test_stems, CLEAN / "test", LOCAL / "test_worst", drop_labels=drop)
    y = write_yaml(LOCAL / "worst.yaml", d / "images")
    c = {"dropped_images": sorted(drop)}
    for rid, seed in CLEAN_SEEDS.items():
        c[f"seed{seed}"] = evaluate(RUNS / rid / "weights" / "best.pt", y, f"C/seed{seed}")
    c["overall_3seed"] = ms([c[f"seed{s}"]["mAP50"] for s in CLEAN_SEEDS.values()])
    c["per_class_ap50_3seed"] = {
        cls: ms([c[f"seed{s}"]["per_class_ap50"][cls] for s in CLEAN_SEEDS.values()])
        for cls in NAMES}
    res["C_borderline_worst_case"] = c
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
