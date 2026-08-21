"""test 스플릿 71장 전수 검수용 렌더링 (dataset_defense Q4 — E1 시작 전 완료 항목).

산출물:
  04_experiments/audit_samples/test71/        개별 71장 (bbox+클래스 오버레이, 원본 해상도)
  04_experiments/audit_samples/test71_sheets/ 2x2 콘택트시트 (1차 육안 검수용)

사용: D:\\venvs\\weld2\\Scripts\\python.exe render_test71.py
(CPU 전용 경량 작업 — E0 학습과 병행 가능)
"""
import math
from pathlib import Path

import cv2
import numpy as np

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2\04_experiments")
IMG_DIR = BASE / "data" / "final0_clean" / "test" / "images"
LBL_DIR = BASE / "data" / "final0_clean" / "test" / "labels"
OUT_IND = BASE / "audit_samples" / "test71"
OUT_SHEET = BASE / "audit_samples" / "test71_sheets"

NAMES = ["Geometric", "Non-fusion", "crack", "porosity", "spatters"]
COLORS = [  # BGR, 클래스별 구분 뚜렷하게
    (0, 165, 255),   # 0 Geometric defect - orange
    (255, 0, 255),   # 1 Non-fusion defect - magenta
    (0, 0, 255),     # 2 crack - red
    (255, 255, 0),   # 3 porosity - cyan
    (0, 220, 0),     # 4 spatters - green
]
TILE = 640
CAP_H = 26


def draw_one(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"read fail: {img_path}")
    h, w = img.shape[:2]
    lbl = LBL_DIR / (img_path.stem + ".txt")
    boxes = []
    if lbl.exists():
        for ln in lbl.read_text(encoding="utf-8").strip().splitlines():
            if not ln.strip():
                continue
            c, cx, cy, bw, bh = ln.split()
            c = int(c)
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            boxes.append((c, x1, y1, x2, y2))
    for c, x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[c], 2)
        tag = NAMES[c]
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 4 if y1 - th - 8 >= 0 else y2 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), COLORS[c], -1)
        cv2.putText(img, tag, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return img, boxes


def to_tile(img: np.ndarray, stem: str, boxes):
    h, w = img.shape[:2]
    s = min(TILE / w, TILE / h)
    r = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((TILE + CAP_H, TILE, 3), 30, np.uint8)
    y0 = (TILE - r.shape[0]) // 2
    x0 = (TILE - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    n_by_c = {}
    for c, *_ in boxes:
        n_by_c[c] = n_by_c.get(c, 0) + 1
    summary = " ".join(f"{NAMES[c]}x{n}" for c, n in sorted(n_by_c.items())) or "NO LABEL"
    cv2.putText(canvas, f"{stem}  [{summary}]", (6, TILE + CAP_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    OUT_IND.mkdir(parents=True, exist_ok=True)
    OUT_SHEET.mkdir(parents=True, exist_ok=True)
    imgs = sorted(IMG_DIR.glob("*"))
    print(f"{len(imgs)} test images")
    tiles = []
    for p in imgs:
        img, boxes = draw_one(p)
        cv2.imwrite(str(OUT_IND / (p.stem + ".jpg")), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append(to_tile(img, p.stem, boxes))
    per = 4  # 2x2
    n_sheet = math.ceil(len(tiles) / per)
    for i in range(n_sheet):
        chunk = tiles[i * per:(i + 1) * per]
        while len(chunk) < per:
            chunk.append(np.full_like(tiles[0], 30))
        top = np.hstack(chunk[0:2])
        bot = np.hstack(chunk[2:4])
        sheet = np.vstack([top, bot])
        cv2.imwrite(str(OUT_SHEET / f"sheet_{i + 1:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"done: {len(tiles)} individual -> {OUT_IND}")
    print(f"done: {n_sheet} sheets -> {OUT_SHEET}")


if __name__ == "__main__":
    main()
