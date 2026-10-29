# -*- coding: utf-8 -*-
"""V5 재구현: Ultralytics val 파이프라인 자체로 bootstrap CI·민감도·보정 전 재채점.

자체 AP 구현이 소형 클래스에서 게이트 D 실측과 괴리(매칭·전처리 세부 차이)를 보여 폐기하고,
보고 수치와 동일한 평가기(m.val)로 통일:
  (1) bootstrap: test 71장 복원추출 리샘플 데이터셋 300회 × 3시드 → per-class AP50의
      3시드 평균 분포 → 95% percentile CI
  (2) 잔존 근접중복 영향 test 12장 제외(59장) 민감도 — phash_impact.json 기반
  (3) 라벨 보정 전(porosity 12) 재채점
전부 동결 모델 재채점(선택 없음). 출력: 05_results/upgrade/test_uncertainty.json
"""
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from weldpaths import project_dir, runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
TEST = BASE / "04_experiments" / "data" / "final0_clean" / "test"
CLEAN = BASE / "04_experiments" / "data" / "final0_clean"
ORIG_LBL = BASE / "04_experiments" / "data" / "label_fixes_20260813" / "labels_orig"
IMPACT = BASE / "05_results" / "upgrade" / "phash_impact.json"
WORK = runs_dir(r"D:\runs\weld2") / "upgrade" / "boot"
OUT = BASE / "05_results" / "upgrade" / "test_uncertainty.json"
RUNS = {"E2_v5n": 42, "E3_v5n_s43": 43, "E3_v5n_s44": 44}
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
N_BOOT = 1000  # v0.5 리뷰 B-5: 300 -> 1000 증량 (2026-08-14)


def make_yaml(dirpath):
    y = dirpath / "data.yaml"
    y.write_text(
        f"train: {CLEAN / 'train' / 'images'}\nval: {CLEAN / 'valid' / 'images'}\n"
        f"test: {dirpath / 'images'}\n\nnc: 5\nnames: {NAMES}\n", encoding="utf-8")
    return y


def eval_dir(models, yaml_path, tag=""):
    """3시드 각각 per-class AP50 dict 반환.
    workers=0: 반복 val 호출 시 Windows 워커 spawn 교착(2026-08-15 행 발생) 방지."""
    res = {}
    for rid, m in models.items():
        r = m.val(data=str(yaml_path), split="test", batch=16, imgsz=640,
                  device=0, workers=0, verbose=False, plots=False)
        res[rid] = {m.names[ci]: float(r.box.ap50[i])
                    for i, ci in enumerate(r.box.ap_class_index.tolist())}
    return res


def three_seed_mean(per_run):
    agg = {}
    for c in NAMES:
        vals = [pr[c] for pr in per_run.values() if c in pr]
        agg[c] = round(float(np.mean(vals)), 4) if vals else None
    agg["overall_mAP50"] = round(float(np.mean(
        [v for v in agg.values() if v is not None])), 4)
    return agg


def main():
    from ultralytics import YOLO
    models = {rid: YOLO(str(runs_dir(r"D:\runs\weld2") / rid / "weights" / "best.pt"))
              for rid in RUNS}
    imgs = sorted((TEST / "images").glob("*.jpg"))
    n = len(imgs)
    out = {"n_boot": N_BOOT, "evaluator": "ultralytics 8.4.118 m.val (동일 파이프라인)"}

    # (2) 잔존 근접중복 제외 민감도
    imp = json.loads(IMPACT.read_text(encoding="utf-8"))
    bad = set(imp["affected_clean_files"].get("test", {}).get("ids", []))
    sub = WORK / "resid_excl"
    if sub.exists():
        shutil.rmtree(sub)
    (sub / "images").mkdir(parents=True)
    (sub / "labels").mkdir(parents=True)
    for f in imgs:
        if f.stem in bad:
            continue
        shutil.copyfile(f, sub / "images" / f.name)
        lbl = TEST / "labels" / (f.stem + ".txt")
        shutil.copyfile(lbl, sub / "labels" / (f.stem + ".txt"))
    r = eval_dir(models, make_yaml(sub))
    out["residual_dup_excluded"] = {
        "excluded_test_images": sorted(bad), "n_kept": n - len(bad),
        "per_class_3seed_mean": three_seed_mean(r)}
    print("[resid-excl]", out["residual_dup_excluded"]["per_class_3seed_mean"])

    # (3) 보정 전 라벨 재채점
    pre = WORK / "precorr"
    if pre.exists():
        shutil.rmtree(pre)
    (pre / "images").mkdir(parents=True)
    (pre / "labels").mkdir(parents=True)
    for f in imgs:
        shutil.copyfile(f, pre / "images" / f.name)
        olbl = ORIG_LBL / (f.stem + ".txt")
        src = olbl if olbl.exists() else TEST / "labels" / (f.stem + ".txt")
        shutil.copyfile(src, pre / "labels" / (f.stem + ".txt"))
    r = eval_dir(models, make_yaml(pre))
    out["pre_correction_labels"] = {"per_class_3seed_mean": three_seed_mean(r)}
    print("[pre-corr]", out["pre_correction_labels"]["per_class_3seed_mean"])

    # (1) bootstrap — 중간 저장·재개 가능 (2026-08-15 행 사고 후 보강)
    partial_p = OUT.with_suffix(".partial.json")
    boot = defaultdict(list)
    start_it = 0
    if partial_p.exists():
        saved = json.loads(partial_p.read_text(encoding="utf-8"))
        for k, v in saved["boot"].items():
            boot[k] = v
        start_it = saved["done"]
        print(f"resuming bootstrap from {start_it}")
    rng = np.random.default_rng(42)
    bootdir = WORK / "resample"
    for it in range(N_BOOT):
        idx = rng.integers(0, n, n)  # 재개 시에도 동일 시퀀스 유지를 위해 항상 생성
        if it < start_it:
            continue
        if bootdir.exists():
            shutil.rmtree(bootdir)
        (bootdir / "images").mkdir(parents=True)
        (bootdir / "labels").mkdir(parents=True)
        for k, i in enumerate(idx):
            f = imgs[i]
            shutil.copyfile(f, bootdir / "images" / f"bs{k:03d}.jpg")
            shutil.copyfile(TEST / "labels" / (f.stem + ".txt"),
                            bootdir / "labels" / f"bs{k:03d}.txt")
        r = eval_dir(models, make_yaml(bootdir))
        agg = three_seed_mean(r)
        for c in NAMES:
            if agg[c] is not None:
                boot[c].append(agg[c])
        boot["overall_mAP50"].append(agg["overall_mAP50"])
        if (it + 1) % 25 == 0:
            print(f"boot {it + 1}/{N_BOOT}")
            partial_p.write_text(json.dumps({"done": it + 1, "boot": dict(boot)}),
                                 encoding="utf-8")

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)]

    out["bootstrap_ci_95"] = {k: {"ci": ci(v), "n": len(v)}
                              for k, v in boot.items()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["bootstrap_ci_95"], indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
