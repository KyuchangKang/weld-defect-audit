# -*- coding: utf-8 -*-
"""Render bbox/polygon labels on random samples for visual audit."""
import random
from pathlib import Path
from PIL import Image, ImageDraw

from weldpaths import out_dir, project_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"C:\Sync_cloud\Dropbox\군산대-랩-공동작업장\Papers\paper_wedling2") / "04_experiments" / "data"
OUT = out_dir(r"C:\Users\ADMIN\AppData\Local\Temp\claude\C--Sync-cloud-Dropbox-------------Papers-paper-wedling2\f6c69a90-7d3e-48f2-9a5c-bf06671a8c4d\scratchpad\audit")
OUT.mkdir(exist_ok=True)
NAMES = {
    "final0": ['GeomDef', 'NonFusion', 'crack', 'porosity', 'spatters'],
    "ikkj": ['BadWeld', 'Crack', 'GoodWeld', 'Porosity', 'Reinf', 'Spatters'],
}
COLORS = ["red", "lime", "cyan", "yellow", "magenta", "orange"]
random.seed(42)

for ds in ["final0", "ikkj"]:
    imgs = [p for p in (BASE / ds / "train" / "images").glob("*.jpg") if len(str(p)) < 250]
    for i, img_path in enumerate(random.sample(imgs, 6)):
        lbl = BASE / ds / "train" / "labels" / (img_path.stem + ".txt")
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        dr = ImageDraw.Draw(im)
        if lbl.exists():
            for ln in open(lbl, encoding="utf-8").read().strip().splitlines():
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
                if len(vals) == 4:  # bbox
                    cx, cy, bw, bh = vals
                    box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                else:  # polygon -> bounding box of polygon
                    xs, ys = vals[0::2], vals[1::2]
                    box = [min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h]
                c = COLORS[cid % len(COLORS)]
                dr.rectangle(box, outline=c, width=3)
                dr.text((box[0] + 3, box[1] + 3), NAMES[ds][cid], fill=c)
        im.thumbnail((800, 800))
        im.save(OUT / f"{ds}_{i}.jpg", quality=85)
        print(f"{ds}_{i}.jpg <- {img_path.name[:70]}")
print("DONE")
