# -*- coding: utf-8 -*-
"""M4 — 3시드 재비교에서 선택이 뒤집힌 YOLOv5mu의 봉인 test 점수 (사후 분석).

⚠️ 이것은 **사전등록 프로토콜 밖의 post-hoc 분석**이다.
   동결된 모델 선택(v5nu)을 바꾸지 않으며, 원고에 그렇게 명시한다.
   리뷰어 요구: "medium으로 했으면 0.70을 넘었나?"라는 독자의 즉각적 질문에 답할 것.

프로토콜은 게이트 D(eval_gate_d.py)와 **완전히 동일**해야 비교가 성립한다:
  final0_clean test 71장 / batch 16 / imgsz 640 / device 0 / workers 8
학습 조건도 동일 확인됨: yolov5mu.pt, 100 epochs, batch 16, 640, lr0 0.01, seed 42/43/44.

출력: evidence/m4_v5mu_test.json
"""
import json, statistics, os
from pathlib import Path

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
CLEAN_YAML = BASE / "04_experiments" / "data" / "final0_clean" / "data.yaml"
LOCAL_RUNS = Path(r"D:\runs\weld2")
LOCAL_OUT = LOCAL_RUNS / "m4_v5mu"
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
GATE_D = BASE / "05_results" / "test_eval" / "gate_d_results.json"

RUNS = {"E1_v5m": (42, LOCAL_RUNS / "E1_v5m"),
        "V2_v5m_s43": (43, LOCAL_RUNS / "upgrade2" / "V2_v5m_s43"),
        "V2_v5m_s44": (44, LOCAL_RUNS / "upgrade2" / "V2_v5m_s44")}
CLASS_ORDER = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
TEST_INSTANCES = {"Geometric defect": 58, "Non-fusion defect": 22, "crack": 22,
                  "porosity": 16, "spatters": 40}


def main():
    from ultralytics import YOLO
    per_run = {}
    for rid, (seed, rdir) in RUNS.items():
        w = rdir / "weights" / "best.pt"
        assert w.exists(), w
        m = YOLO(str(w))
        r = m.val(data=str(CLEAN_YAML), split="test", batch=16, imgsz=640,
                  device=0, workers=8, verbose=False, plots=False,
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
        per_run[rid] = {"seed": seed, "mAP50": round(float(r.box.map50), 4),
                        "mAP50_95": round(float(r.box.map), 4),
                        "precision": round(float(r.box.mp), 4),
                        "recall": round(float(r.box.mr), 4), "per_class": cls}
        print("[%s seed%d] test mAP50=%.4f  mAP50-95=%.4f"
              % (rid, seed, per_run[rid]["mAP50"], per_run[rid]["mAP50_95"]))

    def agg(k):
        v = [per_run[r][k] for r in RUNS]
        return {"mean": round(statistics.mean(v), 4),
                "std": round(statistics.stdev(v), 4), "values": v}

    summary = {k: agg(k) for k in ("mAP50", "mAP50_95", "precision", "recall")}
    pc = {}
    for c in CLASS_ORDER:
        v = [per_run[r]["per_class"][c]["AP50"] for r in RUNS if c in per_run[r]["per_class"]]
        pc[c] = {"AP50_mean": round(statistics.mean(v), 4),
                 "AP50_std": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0,
                 "n_instances": TEST_INSTANCES[c]}

    out = {"note": "POST-HOC — 사전등록 프로토콜 밖. 동결된 v5nu 선택을 변경하지 않음.",
           "protocol": "final0_clean test 71장, 게이트 D와 동일 설정",
           "per_run": per_run, "summary_3seed": summary, "per_class_3seed": pc}

    # v5nu(게이트 D) 대조
    if GATE_D.exists():
        g = json.loads(GATE_D.read_text(encoding="utf-8"))
        n = g["summary_3seed"]["mAP50"]
        out["comparison_vs_v5nu"] = {
            "v5nu_test_mAP50": {"mean": n["mean"], "std": n["std"], "values": n["values"]},
            "v5mu_test_mAP50": summary["mAP50"],
            "delta_mu_minus_nu": round(summary["mAP50"]["mean"] - n["mean"], 4),
            "v5mu_exceeds_0.70": summary["mAP50"]["mean"] >= 0.70,
            "v5nu_per_class": {c: g["per_class_3seed"][c]["AP50_mean"] for c in CLASS_ORDER},
        }
    (HERE / "m4_v5mu_test.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== v5mu 3시드 test mAP50: %.4f ± %.4f ===  (값: %s)"
          % (summary["mAP50"]["mean"], summary["mAP50"]["std"], summary["mAP50"]["values"]))
    print("    mAP50-95: %.4f ± %.4f" % (summary["mAP50_95"]["mean"], summary["mAP50_95"]["std"]))
    for c in CLASS_ORDER:
        print("    %-18s AP50 %.4f ± %.4f (n=%d)"
              % (c, pc[c]["AP50_mean"], pc[c]["AP50_std"], pc[c]["n_instances"]))
    if "comparison_vs_v5nu" in out:
        cmp_ = out["comparison_vs_v5nu"]
        print("\n  v5nu(동결 선택) %.4f ± %.4f  vs  v5mu %.4f ± %.4f   Δ=%+.4f"
              % (cmp_["v5nu_test_mAP50"]["mean"], cmp_["v5nu_test_mAP50"]["std"],
                 summary["mAP50"]["mean"], summary["mAP50"]["std"], cmp_["delta_mu_minus_nu"]))
        print("  >>> v5mu가 0.70 기준을 넘는가? %s" % ("예" if cmp_["v5mu_exceeds_0.70"] else "아니오"))
    print("\n저장:", HERE / "m4_v5mu_test.json")


if __name__ == "__main__":
    main()
