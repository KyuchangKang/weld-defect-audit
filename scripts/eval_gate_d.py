"""게이트 D: 최종 모델(YOLOv5nu) 3시드 test 1회 평가 + RQ3(c) 확보.

evaluation_plan 준수:
  - test는 최종 1회만 평가 (§2.1) — 본 스크립트가 그 1회. 시드 42/43/44 각 best.pt를
    final0_clean test(71장)에서 평가해 mean±std 보고 (§1 재현성)
  - 논문 표 4 원천: per-class AP50 + P + R + 표본 수 (§1)
  - 혼동 행렬·PR 곡선 산출 (plots=True), 예측 시각화는 seed42 모델로 전 71장 저장 (§2.5)
  - RQ3 (c) = seed42 정제학습→정제test mAP50. (a)=e4_leak_eval.json에서 로드 → a−c 산출 (§2.4)

출력: 05_results/test_eval/gate_d_results.json + gate_d_table.csv
      + <run>_test/ (혼동행렬 등 플롯) + pred_vis_s42/ (예측 시각화 71장)
"""
import csv
import json
import shutil
import statistics
from pathlib import Path

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
CLEAN_YAML = BASE / "04_experiments" / "data" / "final0_clean" / "data.yaml"
TEST_IMAGES = BASE / "04_experiments" / "data" / "final0_clean" / "test" / "images"
LOCAL_RUNS = Path(r"D:\runs\weld2")
LOCAL_OUT = LOCAL_RUNS / "gateD"
OUT_DIR = BASE / "05_results" / "test_eval"
E4_JSON = BASE / "05_results" / "e4_leak_eval.json"

RUNS = {"E2_v5n": 42, "E3_v5n_s43": 43, "E3_v5n_s44": 44}
CLASS_ORDER = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
# test 라벨 인스턴스 수 (2026-08-13 보정 반영본 실측 — verify_final0_clean.py와 동일)
TEST_INSTANCES = {"Geometric defect": 58, "Non-fusion defect": 22, "crack": 22,
                  "porosity": 16, "spatters": 40}


def main():
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_run, rows = {}, []
    for rid, seed in RUNS.items():
        w = LOCAL_RUNS / rid / "weights" / "best.pt"
        m = YOLO(str(w))
        r = m.val(data=str(CLEAN_YAML), split="test", batch=16, imgsz=640,
                  device=0, workers=8, verbose=False, plots=True,
                  project=str(LOCAL_OUT), name=f"{rid}_test", exist_ok=True)
        cls = {}
        for i, ci in enumerate(r.box.ap_class_index.tolist()):
            cls[m.names[ci]] = {
                "AP50": round(float(r.box.ap50[i]), 4),
                "AP50_95": round(float(r.box.ap[i]), 4),
                "P": round(float(r.box.p[i]), 4),
                "R": round(float(r.box.r[i]), 4),
                "n_instances": TEST_INSTANCES.get(m.names[ci]),
            }
        per_run[rid] = {
            "seed": seed,
            "mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4),
            "per_class": cls,
        }
        print(f"[{rid} seed{seed}] test mAP50={per_run[rid]['mAP50']:.4f} "
              f"mAP50-95={per_run[rid]['mAP50_95']:.4f}")
        # 플롯(혼동행렬·PR곡선 등)을 Dropbox로 복사
        src = LOCAL_OUT / f"{rid}_test"
        dst = OUT_DIR / f"{rid}_test"
        dst.mkdir(exist_ok=True)
        for f in src.glob("*.png"):
            shutil.copy2(f, dst / f.name)
        rows.append({"run_id": rid, "seed": seed, **{
            k: per_run[rid][k] for k in ("mAP50", "mAP50_95", "precision", "recall")},
            **{f"ap50_{c.split()[0]}": cls.get(c, {}).get("AP50", "") for c in CLASS_ORDER}})

    # 3시드 집계 (mean ± std)
    def agg(key):
        vals = [per_run[r][key] for r in RUNS]
        return {"mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4), "values": vals}

    summary = {k: agg(k) for k in ("mAP50", "mAP50_95", "precision", "recall")}
    per_class_agg = {}
    for c in CLASS_ORDER:
        vals = [per_run[r]["per_class"][c]["AP50"] for r in RUNS if c in per_run[r]["per_class"]]
        per_class_agg[c] = {
            "AP50_mean": round(statistics.mean(vals), 4),
            "AP50_std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "n_instances": TEST_INSTANCES[c],
        }

    # RQ3: (a) raw→raw (e4) vs (c) clean→clean (seed42 본 결과)
    e4 = json.loads(E4_JSON.read_text(encoding="utf-8"))
    a = e4["a_raw_test"]["mAP50"]
    c_val = per_run["E2_v5n"]["mAP50"]
    rq3 = {"a_raw_train_raw_test": a,
           "c_clean_train_clean_test_seed42": c_val,
           "c_clean_3seed_mean": summary["mAP50"]["mean"],
           "inflation_a_minus_c_seed42": round(a - c_val, 4),
           "inflation_a_minus_c_3seed_mean": round(a - summary["mAP50"]["mean"], 4)}

    # 예측 시각화: seed42 모델로 test 전 71장 (논문 그림 후보 선정용)
    m42 = YOLO(str(LOCAL_RUNS / "E2_v5n" / "weights" / "best.pt"))
    m42.predict(source=str(TEST_IMAGES), imgsz=640, device=0, verbose=False,
                save=True, project=str(LOCAL_OUT), name="pred_vis_s42", exist_ok=True)
    vis_dst = OUT_DIR / "pred_vis_s42"
    vis_dst.mkdir(exist_ok=True)
    for f in (LOCAL_OUT / "pred_vis_s42").iterdir():
        if f.suffix.lower() in (".jpg", ".png"):
            shutil.copy2(f, vis_dst / f.name)

    out = {"protocol": "final0_clean test 71장, 3시드 각 1회 (evaluation_plan §1·§2)",
           "gpu": "RTX PRO 5000 Blackwell",
           "per_run": per_run, "summary_3seed": summary,
           "per_class_3seed": per_class_agg, "rq3": rq3}
    (OUT_DIR / "gate_d_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(OUT_DIR / "gate_d_table.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"\n=== 3-seed test mAP50: {summary['mAP50']['mean']:.4f} "
          f"± {summary['mAP50']['std']:.4f} ===")
    for c in CLASS_ORDER:
        a_ = per_class_agg[c]
        print(f"  {c:18s} AP50 {a_['AP50_mean']:.4f}±{a_['AP50_std']:.4f} (n={a_['n_instances']})")
    print(f"RQ3: a={a:.4f}  c(seed42)={c_val:.4f}  a-c={rq3['inflation_a_minus_c_seed42']:+.4f}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
