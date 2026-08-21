"""런 결과 수집: best.pt를 valid 스플릿에서 재검증해 표준 표(results_table.csv)로 추출.

evaluation_plan §5 준수: 보고 수치는 로그/가중치에서 스크립트로 추출 (수기 전사 금지).
test 스플릿은 절대 사용하지 않음 (split='val' 고정 — 최종 test 1회 평가는 별도 스크립트로).

사용: D:\\venvs\\weld2\\Scripts\\python.exe collect_results.py E0_v8s E1_v3u ...
      (인자 없으면 D:\\runs\\weld2 의 전체 런 대상)
출력: 05_results/results_table.csv + 05_results/results_detail.json (per-class AP50 포함)
"""
import csv
import json
import sys
from pathlib import Path

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
DATA = BASE / "04_experiments" / "data" / "final0_clean" / "data.yaml"
LOCAL_RUNS = Path(r"D:\runs\weld2")
OUT_DIR = BASE / "05_results"

CLASS_ORDER = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]


def main():
    run_ids = sys.argv[1:] or sorted(
        p.name for p in LOCAL_RUNS.iterdir()
        if (p / "weights" / "best.pt").exists())
    from ultralytics import YOLO

    rows, detail = [], {}
    for rid in run_ids:
        w = LOCAL_RUNS / rid / "weights" / "best.pt"
        if not w.exists():
            print(f"[skip] {rid}: no best.pt")
            continue
        print(f"[val] {rid}")
        m = YOLO(str(w))
        r = m.val(data=str(DATA), split="val", batch=16, imgsz=640, device=0,
                  workers=8, verbose=False, plots=False)
        # 주의: ultralytics 8.4.118은 model_info(verbose=False)가 None 반환 → get_flops 사용
        from ultralytics.utils.torch_utils import get_flops
        n_p = sum(x.numel() for x in m.model.parameters())
        flops = get_flops(m.model, imgsz=640)
        ap50 = {}
        for i, ci in enumerate(r.box.ap_class_index.tolist()):
            ap50[m.names[ci]] = round(float(r.box.ap50[i]), 4)
        env = {}
        envp = LOCAL_RUNS / rid / "env.json"
        if envp.exists():
            env = json.loads(envp.read_text(encoding="utf-8"))
        row = {
            "run_id": rid,
            "params_M": round(n_p / 1e6, 2),
            "GFLOPs": round(flops, 1),
            "mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4),
            **{f"ap50_{c.split()[0]}": ap50.get(c, "") for c in CLASS_ORDER},
            "wall_h": env.get("wall_hours", ""),
            "peak_vram_gb": env.get("max_vram_alloc_gb", ""),
        }
        rows.append(row)
        detail[rid] = {"ap50": ap50, "env": env}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "results_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    (OUT_DIR / "results_detail.json").write_text(
        json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_csv}")
    for row in rows:
        print(f"{row['run_id']:10s} mAP50={row['mAP50']:.4f} "
              f"mAP50-95={row['mAP50_95']:.4f} params={row['params_M']}M")


if __name__ == "__main__":
    main()
