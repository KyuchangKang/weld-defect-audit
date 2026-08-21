"""E5: 추론 속도(FPS)·지연(ms/img)·params·GFLOPs 측정 (학습 없음).

evaluation_plan §1 준수:
  batch=1, imgsz=640, GPU 워밍업 10장 후 test 71장 평균, 3회 반복 중앙값, 측정 GPU 명시.
test 라벨은 일절 사용하지 않음 (이미지 파일만 읽어 순수 추론 시간 측정 — 게이트 D 무관).

대상: 아키텍처가 서로 다른 런만 (E3 시드·E4는 E2_v5n과 동일 아키텍처라 제외).
사용: D:\\venvs\\weld2\\Scripts\\python.exe measure_e5.py [run_id ...]
출력: 05_results/e5_speed.csv + e5_speed.json (반복별 원시값·속도 분해 포함)
"""
import csv
import json
import statistics
import sys
import time
from pathlib import Path

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
TEST_IMAGES = BASE / "04_experiments" / "data" / "final0_clean" / "test" / "images"
LOCAL_RUNS = Path(r"D:\runs\weld2")
OUT_DIR = BASE / "05_results"

DEFAULT_RUNS = [
    "E0_v8s",
    "E1_v3u", "E1_v5m", "E1_v8m", "E1_v11m", "E1_v12m",
    "E2_v5n", "E2_v5s", "E2_v5l", "E2_v5x",
]
WARMUP_N = 10
REPEATS = 3


def main():
    import torch
    from ultralytics import YOLO

    run_ids = sys.argv[1:] or DEFAULT_RUNS
    images = sorted(TEST_IMAGES.glob("*.jpg")) + sorted(TEST_IMAGES.glob("*.png"))
    assert len(images) == 71, f"test images expected 71, got {len(images)}"
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu} / images: {len(images)} / repeats: {REPEATS}")

    rows, detail = [], {"gpu": gpu, "protocol": {
        "batch": 1, "imgsz": 640, "warmup_images": WARMUP_N,
        "timed_images": len(images), "repeats": REPEATS, "reduce": "median-of-means",
    }, "runs": {}}

    for rid in run_ids:
        w = LOCAL_RUNS / rid / "weights" / "best.pt"
        if not w.exists():
            print(f"[skip] {rid}: no best.pt")
            continue
        m = YOLO(str(w))
        # 주의: ultralytics 8.4.118은 model_info(verbose=False)가 None 반환 → get_flops 사용
        from ultralytics.utils.torch_utils import get_flops
        n_p = sum(x.numel() for x in m.model.parameters())
        flops = get_flops(m.model, imgsz=640)

        rep_means, rep_speeds = [], []
        for rep in range(REPEATS):
            for img in images[:WARMUP_N]:  # 워밍업 10장 (비계측)
                m.predict(str(img), imgsz=640, device=0, verbose=False)
            torch.cuda.synchronize()
            lat, spd = [], {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
            for img in images:
                t0 = time.perf_counter()
                r = m.predict(str(img), imgsz=640, device=0, verbose=False)
                torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) * 1000.0)
                for k in spd:
                    spd[k] += r[0].speed[k]
            rep_means.append(statistics.mean(lat))
            rep_speeds.append({k: v / len(images) for k, v in spd.items()})

        med = statistics.median(rep_means)  # 3회 반복 중앙값 (ms/img, end-to-end)
        med_idx = rep_means.index(med)
        inf_ms = rep_speeds[med_idx]["inference"]
        row = {
            "run_id": rid,
            "params_M": round(n_p / 1e6, 2),
            "GFLOPs": round(flops, 1),
            "latency_ms": round(med, 2),
            "FPS": round(1000.0 / med, 1),
            "inference_ms": round(inf_ms, 2),
            "FPS_inference_only": round(1000.0 / inf_ms, 1),
            "gpu": gpu,
        }
        rows.append(row)
        detail["runs"][rid] = {
            "repeat_mean_latency_ms": [round(x, 3) for x in rep_means],
            "median_repeat_speed_breakdown_ms": {
                k: round(v, 3) for k, v in rep_speeds[med_idx].items()},
        }
        print(f"{rid:10s} latency={row['latency_ms']:7.2f}ms  FPS={row['FPS']:6.1f}  "
              f"(inference-only {row['FPS_inference_only']:.1f})  "
              f"params={row['params_M']}M  GFLOPs={row['GFLOPs']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "e5_speed.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    (OUT_DIR / "e5_speed.json").write_text(
        json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
