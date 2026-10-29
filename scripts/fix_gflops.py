"""GFLOPs 보정: e5_speed.csv/json + results_table.csv 의 GFLOPs 열을 재계산해 갱신.

배경: ultralytics 8.4.118에서 torch_utils.model_info(verbose=False)가 None을 반환해
measure_e5.py/collect_results.py 초기 실행분의 GFLOPs가 0.0으로 기록됨.
get_flops(model, imgsz=640)로 재계산 (수기 전사 금지 원칙 — 스크립트 갱신).
타이밍·mAP 등 다른 열은 건드리지 않음.
"""
import csv
import json
from pathlib import Path

from weldpaths import project_dir, runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
LOCAL_RUNS = runs_dir(r"D:\runs\weld2")
OUT_DIR = BASE / "05_results"


def main():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    cache = {}

    def gflops(rid):
        if rid not in cache:
            m = YOLO(str(LOCAL_RUNS / rid / "weights" / "best.pt"))
            cache[rid] = round(get_flops(m.model, imgsz=640), 1)
        return cache[rid]

    for name in ("e5_speed.csv", "results_table.csv"):
        p = OUT_DIR / name
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        for row in rows:
            row["GFLOPs"] = gflops(row["run_id"])
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"updated {p}")

    jp = OUT_DIR / "e5_speed.json"
    if jp.exists():
        d = json.loads(jp.read_text(encoding="utf-8"))
        d["gflops_at_640"] = {rid: cache[rid] for rid in sorted(cache)}
        d["note"] = ("GFLOPs는 fix_gflops.py로 재계산(get_flops, imgsz=640) — "
                     "8.4.118 model_info None 반환 이슈 보정")
        jp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"updated {jp}")

    for rid in sorted(cache):
        print(f"{rid:10s} GFLOPs={cache[rid]}")


if __name__ == "__main__":
    main()
