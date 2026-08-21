"""E4 누수 효과 이중 평가 (RQ3, evaluation_plan §2.4).

원 스플릿(final0 raw, 누수 포함)으로 학습한 모델(E4_v5n_raw best.pt)을
  (a) 원 스플릿 test  — 기존 관행 재현 (누수 포함 평가)
  (b) 정제 스플릿 test — 동일 모델을 누수 없는 test로 평가
두 조건에서 평가해 부풀림(= a − b)을 정량화한다.

주의: 본 스크립트는 원 스플릿 학습 모델만 다룬다. 최종 모델(정제 학습)의 test 평가(c)는
게이트 D에서 1회만 수행 (test 1회 원칙과 무관한 별도 모델임을 명시).

출력: 05_results/e4_leak_eval.json
"""
import json
from pathlib import Path

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
RAW_YAML = BASE / "04_experiments" / "data" / "final0" / "data.yaml"
CLEAN_YAML = BASE / "04_experiments" / "data" / "final0_clean" / "data.yaml"
WEIGHTS = Path(r"D:\runs\weld2") / "E4_v5n_raw" / "weights" / "best.pt"
OUT = BASE / "05_results" / "e4_leak_eval.json"


def run_eval(model, data_yaml, tag):
    r = model.val(data=str(data_yaml), split="test", batch=16, imgsz=640,
                  device=0, workers=8, verbose=False, plots=False)
    ap50 = {}
    for i, ci in enumerate(r.box.ap_class_index.tolist()):
        ap50[model.names[ci]] = round(float(r.box.ap50[i]), 4)
    res = {
        "data": str(data_yaml),
        "mAP50": round(float(r.box.map50), 4),
        "mAP50_95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "per_class_ap50": ap50,
    }
    print(f"[{tag}] mAP50={res['mAP50']:.4f} mAP50-95={res['mAP50_95']:.4f}")
    return res


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    out = {
        "model": str(WEIGHTS),
        "trained_on": "final0 raw (leaky original split)",
        "a_raw_test": run_eval(model, RAW_YAML, "a: raw test (leaky)"),
        "b_clean_test": run_eval(model, CLEAN_YAML, "b: clean test (leak-free)"),
    }
    out["inflation_mAP50_a_minus_b"] = round(
        out["a_raw_test"]["mAP50"] - out["b_clean_test"]["mAP50"], 4)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ninflation (a-b) mAP50 = {out['inflation_mAP50_a_minus_b']:+.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
