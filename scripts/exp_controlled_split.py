# -*- coding: utf-8 -*-
"""통제 분할 실험 (2026-08-18, 사용자 승인) — 리뷰 3종 공통 지적 "누수 효과 vs 데이터 구성 효과" 분리.

문제: 조건 (a) raw(4,027 train / 310 test)와 (c) clean(3,567 / 71)은 누수 유무 외에
      학습량(-11%)과 test 크기(310 vs 71)도 함께 다르다.

통제 조건 (전부 YOLOv5nu, 동일 하이퍼파라미터, 학습 시드 42):
  (a)    기존 실측       raw train 4,027 -> raw test 310            = 0.8313
  (a-v)  학습량 일치     raw train에서 무작위 3,567장 -> raw test 310   [신규 학습 1런]
  (a-t)  test 크기 일치  기존 raw 모델(E4) -> raw test의 무작위 71장 x R회
  (a-vt) 둘 다 일치      (a-v) 모델      -> 동일한 71장 부분집합 x R회
  (c)    기존 실측       clean 3,567 -> clean test 71               = 0.576 +- 0.025

해석: (a-v)와 (a-vt)가 (a)와 유사하면 학습량·test 크기는 a-c 격차의 원인이 아니다.
출력: 05_results/upgrade/controlled_split.json
"""
import json
import random
import shutil
import statistics
from collections import Counter
from pathlib import Path

from weldpaths import project_dir, runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
DATA = BASE / "04_experiments" / "data"
RAW = DATA / "final0"
LOCAL = runs_dir(r"D:\runs\weld2") / "ctrl"
OUT = BASE / "05_results" / "upgrade" / "controlled_split.json"
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
N_TRAIN_MATCH = 3567          # final0_clean train 파일 수
N_TEST_MATCH = 71             # final0_clean test 파일 수
N_SUBSETS = 20                # test 부분집합 반복
SEED = 42
TRAIN_KW = dict(epochs=100, batch=16, imgsz=640, lr0=0.01, device=0,
                workers=4, cache=False, exist_ok=True, plots=False, verbose=False)
RAW_BEST = runs_dir(r"D:\runs\weld2") / "E4_v5n_raw" / "weights" / "best.pt"


def ep(p):
    # 이 PC는 LongPathsEnabled=1 + 최장 경로 ~120자이므로 접두사 불필요
    return str(p)


def copy_pair(stem, src_split, dst_dir):
    shutil.copyfile(ep(src_split / "images" / (stem + ".jpg")),
                    ep(dst_dir / "images" / (stem + ".jpg")))
    lbl = src_split / "labels" / (stem + ".txt")
    if lbl.exists():
        shutil.copyfile(ep(lbl), ep(dst_dir / "labels" / (stem + ".txt")))


def instance_counts(split_dir, stems=None):
    c = Counter()
    for lbl in (split_dir / "labels").iterdir():
        if stems is not None and lbl.stem not in stems:
            continue
        for line in lbl.read_text().splitlines():
            if line.strip():
                c[NAMES[int(line.split()[0])]] += 1
    return dict(c)


def write_yaml(path, train, val, test):
    path.write_text(f"train: {train}\nval: {val}\ntest: {test}\n\nnc: 5\nnames: {NAMES}\n",
                    encoding="utf-8")
    return path


def evaluate(weights, yaml_path, split, tag):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data=str(yaml_path), split=split, batch=16, imgsz=640, device=0,
              workers=0, verbose=False, plots=False)     # 반복 평가는 workers=0 (세션3 교훈)
    ap = {m.names[ci]: round(float(r.box.ap50[i]), 4)
          for i, ci in enumerate(r.box.ap_class_index.tolist())}
    res = {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
           "per_class_ap50": ap}
    print(f"[{tag}] {split} mAP50={res['mAP50']:.4f}", flush=True)
    return res


def main():
    rng = random.Random(SEED)
    LOCAL.mkdir(parents=True, exist_ok=True)
    res = {"design": "controlled split: volume-matched / test-size-matched leaky conditions",
           "seed": SEED, "n_train_match": N_TRAIN_MATCH, "n_test_match": N_TEST_MATCH,
           "n_subsets": N_SUBSETS}
    if OUT.exists():
        res.update(json.loads(OUT.read_text(encoding="utf-8")))

    # ---------- 1) 학습량 일치 데이터셋 (raw train 무작위 3,567장) ----------
    ds = LOCAL / "data_av"
    train_stems = sorted(p.stem for p in (RAW / "train" / "images").iterdir())
    pick = sorted(rng.sample(train_stems, N_TRAIN_MATCH))
    if not (ds / "train" / "images").exists() or \
            len(list((ds / "train" / "images").iterdir())) != N_TRAIN_MATCH:
        for s in ("train", "valid", "test"):
            for sub in ("images", "labels"):
                (ds / s / sub).mkdir(parents=True, exist_ok=True)
        print(f"copying {N_TRAIN_MATCH} train files ...", flush=True)
        for st in pick:
            copy_pair(st, RAW / "train", ds / "train")
        for s in ("valid", "test"):                      # raw valid/test 그대로
            for p in (RAW / s / "images").iterdir():
                copy_pair(p.stem, RAW / s, ds / s)
        print("copy done", flush=True)
    yaml_av = write_yaml(ds / "data.yaml", ds / "train" / "images",
                         ds / "valid" / "images", ds / "test" / "images")
    res["train_instances"] = {
        "raw_full": instance_counts(RAW / "train"),
        "volume_matched": instance_counts(RAW / "train", set(pick)),
    }
    res["n_raw_train_files"] = len(train_stems)

    # ---------- 2) (a-v) 학습 ----------
    best_av = LOCAL / "av_run" / "weights" / "best.pt"
    if not best_av.exists():
        from ultralytics import YOLO
        print("training (a-v) volume-matched leaky ...", flush=True)
        YOLO("yolov5nu.pt").train(data=str(yaml_av), project=str(LOCAL), name="av_run",
                                  seed=SEED, **TRAIN_KW)
    res["a_v_full_raw_test"] = evaluate(best_av, yaml_av, "test", "a-v / raw test 310")

    # ---------- 3) test 크기 일치 부분집합 ----------
    test_stems = sorted(p.stem for p in (RAW / "test" / "images").iterdir())
    res["n_raw_test_files"] = len(test_stems)
    sub_dir = LOCAL / "sub"
    a_t, a_vt = [], []
    for k in range(N_SUBSETS):
        stems = rng.sample(test_stems, N_TEST_MATCH)
        if sub_dir.exists():
            shutil.rmtree(ep(sub_dir), ignore_errors=True)
        for s in ("test",):
            for sub in ("images", "labels"):
                (sub_dir / s / sub).mkdir(parents=True, exist_ok=True)
        for st in stems:
            copy_pair(st, RAW / "test", sub_dir / "test")
        y = write_yaml(sub_dir / "data.yaml", ds / "train" / "images",
                       ds / "valid" / "images", sub_dir / "test" / "images")
        a_t.append(evaluate(RAW_BEST, y, "test", f"a-t {k+1}/{N_SUBSETS}")["mAP50"])
        a_vt.append(evaluate(best_av, y, "test", f"a-vt {k+1}/{N_SUBSETS}")["mAP50"])
        res["a_t_subsets"] = a_t
        res["a_vt_subsets"] = a_vt
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    def ms(v):
        return {"mean": round(statistics.mean(v), 4),
                "std": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0,
                "min": round(min(v), 4), "max": round(max(v), 4), "n": len(v)}

    res["summary"] = {
        "a_reference": 0.8313,
        "a_v_volume_matched": res["a_v_full_raw_test"]["mAP50"],
        "a_t_test_size_matched": ms(a_t),
        "a_vt_both_matched": ms(a_vt),
        "c_clean_protocol": 0.576,
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, flush=True)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
