# -*- coding: utf-8 -*-
"""V5 사후 분석 (evaluation_plan v1.2): 이미지 단위 bootstrap CI + 라벨 보정 전 재채점.

동결된 게이트 D 모델 3시드의 test 예측을 1회 추출(conf=0.001 — 평가용 임계)한 뒤,
자체 AP50 구현(COCO 101-point, Ultralytics compute_ap와 동일 방식)으로
  (1) 전체 71장 AP를 게이트 D 실측과 대조해 구현 검증
  (2) 이미지 단위 percentile bootstrap 1,000회 → per-class AP50의 95% CI (3시드 평균 기준)
  (3) 보정 전 라벨(porosity 12)로 재채점 — 보정 전/후 병기용
모델·라벨·선택 불변(동결 측정의 재분석). 출력: 05_results/upgrade/bootstrap_ci.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from weldpaths import project_dir, runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
TEST = BASE / "04_experiments" / "data" / "final0_clean" / "test"
ORIG_LBL = BASE / "04_experiments" / "data" / "label_fixes_20260813" / "labels_orig"
GATE_D = BASE / "05_results" / "test_eval" / "gate_d_results.json"
OUT = BASE / "05_results" / "upgrade" / "bootstrap_ci.json"
RUNS = {"E2_v5n": 42, "E3_v5n_s43": 43, "E3_v5n_s44": 44}
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
N_BOOT = 1000


def load_gt(lbl_path, w, h):
    out = []
    if lbl_path.exists():
        for ln in lbl_path.read_text(encoding="utf-8").strip().splitlines():
            p = ln.split()
            if len(p) == 5:
                c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:])
                out.append((c, (xc - bw / 2) * w, (yc - bh / 2) * h,
                            (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def iou(box, boxes):
    if not len(boxes):
        return np.zeros(0)
    b = np.asarray(boxes, dtype=float)
    x1 = np.maximum(box[0], b[:, 0]); y1 = np.maximum(box[1], b[:, 1])
    x2 = np.minimum(box[2], b[:, 2]); y2 = np.minimum(box[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def compute_ap(tp, conf, npos):
    """Ultralytics 방식: 101-point interpolation."""
    if npos == 0:
        return None
    if not len(tp):
        return 0.0
    order = np.argsort(-np.asarray(conf))
    tp = np.asarray(tp, dtype=float)[order]
    fpc = np.cumsum(1 - tp)
    tpc = np.cumsum(tp)
    recall = tpc / npos
    precision = tpc / (tpc + fpc)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def match_image(dets, gts, iou_thr=0.5):
    """이미지 1장 매칭 — Ultralytics match_predictions와 동일 방식:
    (det, gt) 후보쌍을 IoU 내림차순으로 그리디 유일 배정(신뢰도는 매칭에 미사용,
    PR 정렬에만 사용). dets [(cls, conf, box)], gts [(cls, box)]."""
    out = defaultdict(lambda: ([], []))  # cls -> (tps, confs)
    for c in set(d[0] for d in dets) | set(g[0] for g in gts):
        cd = [d for d in dets if d[0] == c]
        cg = [g[1] for g in gts if g[0] == c]
        tp_flags = np.zeros(len(cd), dtype=int)
        if cd and cg:
            pairs = []
            for di, (_, _, box) in enumerate(cd):
                for gi, giou in enumerate(iou(box, cg)):
                    if giou >= iou_thr:
                        pairs.append((float(giou), di, gi))
            pairs.sort(key=lambda p: -p[0])
            used_d, used_g = set(), set()
            for _, di, gi in pairs:
                if di not in used_d and gi not in used_g:
                    used_d.add(di); used_g.add(gi)
                    tp_flags[di] = 1
        for di, (_, conf, _) in enumerate(cd):
            out[c][0].append(int(tp_flags[di])); out[c][1].append(conf)
    return out


def ap_over_images(per_img, indices):
    """per_img: list of (match_dict, gt_counts). indices로 뽑은 표본의 per-class AP."""
    tp = defaultdict(list); cf = defaultdict(list); npos = defaultdict(int)
    for i in indices:
        md, gc = per_img[i]
        for c, (tps, confs) in md.items():
            tp[c].extend(tps); cf[c].extend(confs)
        for c, k in gc.items():
            npos[c] += k
    aps = {}
    for c in range(5):
        aps[c] = compute_ap(tp[c], cf[c], npos[c])
    return aps


def main():
    from ultralytics import YOLO

    imgs = sorted((TEST / "images").glob("*.jpg"))
    sizes = {}
    gts_corr, gts_orig = [], []
    for f in imgs:
        with Image.open(f) as im:
            w, h = im.size
        sizes[f.stem] = (w, h)
        lbl = TEST / "labels" / (f.stem + ".txt")
        gts_corr.append([(c, (x1, y1, x2, y2))
                         for c, x1, y1, x2, y2 in load_gt(lbl, w, h)])
        olbl = ORIG_LBL / (f.stem + ".txt")
        use = olbl if olbl.exists() else lbl
        gts_orig.append([(c, (x1, y1, x2, y2))
                         for c, x1, y1, x2, y2 in load_gt(use, w, h)])

    gate = json.loads(GATE_D.read_text(encoding="utf-8"))
    out = {"n_boot": N_BOOT, "validation": {}, "per_run_full": {}}
    per_run_imgs_corr, per_run_imgs_orig = {}, {}

    for rid in RUNS:
        m = YOLO(str(runs_dir(r"D:\runs\weld2") / rid / "weights" / "best.pt"))
        preds = []
        for f in imgs:
            r = m.predict(str(f), imgsz=640, device=0, conf=0.001, iou=0.7,
                          max_det=300, verbose=False)[0]
            b = r.boxes
            preds.append([(int(c), float(cf), tuple(map(float, xy)))
                          for c, cf, xy in zip(b.cls.tolist(), b.conf.tolist(),
                                               b.xyxy.tolist())])
        def build(gts):
            per = []
            for det, gt in zip(preds, gts):
                gc = defaultdict(int)
                for c, _ in gt:
                    gc[c] += 1
                per.append((match_image(det, gt), dict(gc)))
            return per
        per_run_imgs_corr[rid] = build(gts_corr)
        per_run_imgs_orig[rid] = build(gts_orig)

        # 구현 검증: 전체셋 AP vs gate_d
        aps = ap_over_images(per_run_imgs_corr[rid], range(len(imgs)))
        full = {NAMES[c]: (None if aps[c] is None else round(aps[c], 4))
                for c in range(5)}
        out["per_run_full"][rid] = full
        deltas = {}
        for c in range(5):
            ref = gate["per_run"][rid]["per_class"][NAMES[c]]["AP50"]
            deltas[NAMES[c]] = round((aps[c] or 0) - ref, 4)
        out["validation"][rid] = deltas
        print(f"[{rid}] full-set AP deltas vs gate_d: {deltas}")

    # bootstrap: 3시드 평균 AP의 CI
    rng = np.random.default_rng(42)
    n = len(imgs)
    boot = defaultdict(list)   # cls -> list of 3-seed-mean AP
    boot_overall = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        per_cls_means = {}
        for c in range(5):
            vals = []
            for rid in RUNS:
                ap = ap_over_images(per_run_imgs_corr[rid], idx)[c]
                if ap is not None:
                    vals.append(ap)
            if vals:
                per_cls_means[c] = float(np.mean(vals))
                boot[c].append(per_cls_means[c])
        if per_cls_means:
            boot_overall.append(float(np.mean(list(per_cls_means.values()))))

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)]

    out["bootstrap_ci_95"] = {
        NAMES[c]: {"ci": ci(boot[c]), "n_boot_valid": len(boot[c])}
        for c in range(5)}
    out["bootstrap_ci_95"]["overall_mAP50"] = {"ci": ci(boot_overall)}

    # 보정 전 라벨 재채점 (전체셋, 3시드 평균)
    pre = {}
    for c in range(5):
        vals = []
        for rid in RUNS:
            ap = ap_over_images(per_run_imgs_orig[rid], range(n))[c]
            if ap is not None:
                vals.append(ap)
        pre[NAMES[c]] = round(float(np.mean(vals)), 4) if vals else None
    pre["overall_mAP50"] = round(float(np.mean(
        [v for v in pre.values() if v is not None])), 4)
    out["pre_correction_labels_3seed_mean"] = pre

    # 잔존 근접중복 영향 test 제외 민감도 (phash_impact.json의 test id — 동결 예측 재집계)
    impact_p = BASE / "05_results" / "upgrade" / "phash_impact.json"
    if impact_p.exists():
        imp = json.loads(impact_p.read_text(encoding="utf-8"))
        bad = set(imp["affected_clean_files"].get("test", {}).get("ids", []))
        keep = [k for k, f in enumerate(imgs) if f.stem not in bad]
        sens = {}
        for c in range(5):
            vals = []
            for rid in RUNS:
                ap = ap_over_images(per_run_imgs_corr[rid], keep)[c]
                if ap is not None:
                    vals.append(ap)
            sens[NAMES[c]] = round(float(np.mean(vals)), 4) if vals else None
        sens["overall_mAP50"] = round(float(np.mean(
            [v for v in sens.values() if v is not None])), 4)
        out["residual_dup_excluded"] = {
            "excluded_test_images": sorted(bad), "n_kept": len(keep),
            "per_class_3seed_mean": sens}

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out
                      if k in ("bootstrap_ci_95", "pre_correction_labels_3seed_mean",
                               "residual_dup_excluded")},
                     ensure_ascii=False, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
