# -*- coding: utf-8 -*-
"""Validate weld-qwzyj v7: stats, color check, source overlap with final0."""
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import numpy as np

from weldpaths import project_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"C:\Sync_cloud\Dropbox\군산대-랩-공동작업장\Papers\paper_wedling2") / "04_experiments" / "data"
NAMES = ['crack', 'porosity', 'spatter']
SPLITS = ["train", "valid", "test"]
random.seed(7)


def source_stem(name: str) -> str:
    s = re.sub(r"\.rf\.[0-9a-f]{32}.*$", "", name)
    s = re.sub(r"_(jpeg|jpg|png|bmp)(_(jpeg|jpg|png|bmp))*$", "", s)
    return s.lower()


rep = {"splits": {}, "class_instances": Counter(), "images_with_class": Counter(),
       "label_issues": 0, "polygon_lines": 0}
stems = defaultdict(lambda: defaultdict(int))
sizes = Counter()

for split in SPLITS:
    img_dir = BASE / "qwzyj" / split / "images"
    lbl_dir = BASE / "qwzyj" / split / "labels"
    imgs = [p for p in img_dir.glob("*") if p.is_file()] if img_dir.exists() else []
    rep["splits"][split] = len(imgs)
    for img in imgs:
        stems[source_stem(img.name)][split] += 1
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            rep["label_issues"] += 1
            continue
        seen = set()
        for ln in open("\\\\?\\" + str(lbl), encoding="utf-8").read().strip().splitlines():
            parts = ln.split()
            if len(parts) == 5:
                cid = int(parts[0])
                rep["class_instances"][NAMES[cid]] += 1
                seen.add(NAMES[cid])
            elif len(parts) > 5:
                rep["polygon_lines"] += 1
        for c in seen:
            rep["images_with_class"][c] += 1

# color check: saturation of 40 random train images
train_imgs = [p for p in (BASE / "qwzyj" / "train" / "images").glob("*.jpg")]
sat_means, dims = [], []
for p in random.sample(train_imgs, min(40, len(train_imgs))):
    im = Image.open("\\\\?\\" + str(p)).convert("HSV")
    a = np.asarray(im)
    sat_means.append(float(a[:, :, 1].mean()))
    dims.append(im.size)
    sizes[im.size] += 1

rep["color_check"] = {
    "sampled": len(sat_means),
    "mean_saturation_0_255": round(float(np.mean(sat_means)), 1),
    "images_with_sat_below_10(≈grayscale)": int(sum(1 for s in sat_means if s < 10)),
    "common_sizes": {str(k): v for k, v in sizes.most_common(5)},
}

# source stats + leakage
leak = {k: dict(v) for k, v in stems.items() if len(v) > 1}
rep["unique_sources"] = len(stems)
rep["sources_spanning_splits"] = len(leak)

# overlap with final0
f0_stems = set()
for split in SPLITS:
    d = BASE / "final0" / split / "images"
    for p in d.glob("*"):
        if p.is_file():
            f0_stems.add(source_stem(p.name))
common = set(stems) & f0_stems
rep["common_sources_with_final0"] = len(common)
rep["common_examples"] = sorted(common)[:10]

rep["class_instances"] = dict(rep["class_instances"])
rep["images_with_class"] = dict(rep["images_with_class"])
with open(BASE / "qwzyj_report.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print(json.dumps(rep, ensure_ascii=False, indent=2))
