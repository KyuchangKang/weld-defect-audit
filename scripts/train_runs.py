"""Phase 5 학습 러너 (experiment_plan.md §2·§5 준수).

사용법 (venv: D:\\venvs\\weld2):
  D:\\venvs\\weld2\\Scripts\\python.exe train_runs.py E0_v8s        # 학습 실행
  D:\\venvs\\weld2\\Scripts\\python.exe train_runs.py E0_v8s --dry  # 설정 확인만

전 런 고정 조건(비교 공정성 — 변경 금지): final0_clean, epochs 100, batch 16,
imgsz 640, lr0 0.01, COCO 사전학습, Ultralytics 온라인 증강 기본값, 나머지 기본값.
RAM 16GB 제약: cache=False, workers<=8, 학습 동시 1런만.
test 스플릿은 학습·게이트 중 절대 사용하지 않음 (evaluation_plan: test 1회 원칙 —
학습 중 검증은 valid 스플릿만 사용됨).

산출물: 원시 러닝 로그 전체는 로컬 D:\\runs\\weld2\\<run_id>\\ 에 보존,
핵심 파일만 Dropbox 05_results/runs/<run_id>/ 로 복사(동기화 부담 최소화, 잠금 재시도).
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from weldpaths import project_dir, runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
DATA = BASE / "04_experiments" / "data" / "final0_clean" / "data.yaml"
LOCAL_RUNS = runs_dir(r"D:\runs\weld2")  # Dropbox 밖(원시 산출물·다운로드 가중치)
DBX_RUNS = BASE / "05_results" / "runs"  # 핵심 결과만

FIXED = dict(epochs=100, batch=16, imgsz=640, lr0=0.01, cache=False, workers=8,
             device=0, seed=42, exist_ok=False, plots=True)

# 실험 레지스트리. E2(크기)·E3(시드 43/44)·E4(원 스플릿)는 게이트 B/C 통과 후 추가.
# E1 모델 자산명은 실행 직전 설치된 ultralytics 버전에서 존재 여부 검증할 것.
RUNS = {
    "E0_v8s": dict(model="yolov8s.pt"),  # 파일럿 — 동일 조건이므로 본실험(E2 s)에 재사용 가능
    "E1_v3u": dict(model="yolov3u.pt"),
    "E1_v5m": dict(model="yolov5mu.pt"),
    "E1_v8m": dict(model="yolov8m.pt"),
    "E1_v11m": dict(model="yolo11m.pt"),
    "E1_v12m": dict(model="yolo12m.pt"),
    # E2: 게이트 B 선정 버전 = v5(u) — 크기 비교 (m은 E1_v5m 재사용)
    "E2_v5n": dict(model="yolov5nu.pt"),
    "E2_v5s": dict(model="yolov5su.pt"),
    "E2_v5l": dict(model="yolov5lu.pt"),
    "E2_v5x": dict(model="yolov5xu.pt"),
    # E3: 게이트 C 최종 모델(v5nu) 시드 반복 — seed 42는 E2_v5n 재사용
    "E3_v5n_s43": dict(model="yolov5nu.pt", seed=43),
    "E3_v5n_s44": dict(model="yolov5nu.pt", seed=44),
    # E4: 원 스플릿(final0 raw, 누수 포함) 학습 — RQ3. 이중 test 평가는 eval_e4.py
    "E4_v5n_raw": dict(model="yolov5nu.pt",
                       data=str(BASE / "04_experiments" / "data" / "final0" / "data.yaml")),
}

KEY_FILES = ["results.csv", "args.yaml", "results.png", "confusion_matrix.png",
             "confusion_matrix_normalized.png", "weights/best.pt", "env.json"]


def copy_retry(src: Path, dst: Path, tries: int = 5) -> bool:
    """Dropbox 파일 잠금 대비 재시도 복사 (세션 1 기술 메모 패턴)."""
    for i in range(tries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except OSError:
            time.sleep(0.5 * (i + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", choices=sorted(RUNS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    cfg = RUNS[args.run_id]
    overrides = {k: v for k, v in cfg.items() if k != "model"}
    kwargs = {**FIXED, "data": str(DATA), **overrides,
              "project": str(LOCAL_RUNS), "name": args.run_id}
    print(json.dumps({"run_id": args.run_id, "model": cfg["model"], **kwargs}, indent=2))
    if args.dry:
        return

    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    os.chdir(LOCAL_RUNS)  # 사전학습 가중치 .pt 다운로드도 로컬에 떨어지도록

    import torch
    import ultralytics
    from ultralytics import YOLO

    env = dict(python=".".join(map(str, __import__("sys").version_info[:3])),
               torch=torch.__version__, ultralytics=ultralytics.__version__,
               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    print("[env]", env)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = YOLO(cfg["model"])
    model.train(**kwargs)
    wall = time.time() - t0

    run_dir = LOCAL_RUNS / args.run_id
    env.update(wall_hours=round(wall / 3600, 3),
               max_vram_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    (run_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    dst_dir = DBX_RUNS / args.run_id
    copied, failed = [], []
    for rel in KEY_FILES:
        src = run_dir / rel
        if src.exists() and copy_retry(src, dst_dir / Path(rel).name):
            copied.append(rel)
        else:
            failed.append(rel)
    print(f"[done] {args.run_id}: wall {wall / 3600:.2f}h, "
          f"peak VRAM {env['max_vram_alloc_gb']}GB")
    print(f"[dropbox] copied={copied} failed_or_missing={failed}")


if __name__ == "__main__":
    main()
