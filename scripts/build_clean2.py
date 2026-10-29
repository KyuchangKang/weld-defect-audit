# -*- coding: utf-8 -*-
"""Build leakage-free, class-stratified final0_clean dataset (v2).

Split unit: parent photo. Split sizing: unique-content counts (not augmented file counts).
Stratification: greedy assignment minimizing per-class deficit vs 70/20/10 targets.
train: all copies (baked augments kept). valid/test: one copy per content_id.
"""
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from weldpaths import project_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"C:\Sync_cloud\Dropbox\군산대-랩-공동작업장\Papers\paper_wedling2") / "04_experiments" / "data"
SRC = BASE / "final0"
DST = BASE / "final0_clean"
NAMES = ['Geometric defect', 'Non-fusion defect', 'crack', 'porosity', 'spatters']
SPLITS = ["train", "valid", "test"]
RATIOS = {"train": 0.70, "valid": 0.20, "test": 0.10}
random.seed(42)


def ep(p) -> str:
    return "\\\\?\\" + str(p)


def source_stem(name: str) -> str:
    s = re.sub(r"\.rf\.[0-9a-f]{32}.*$", "", name)
    s = re.sub(r"_(jpeg|jpg|png|bmp)(_(jpeg|jpg|png|bmp))*$", "", s, flags=re.I)
    return s.lower()


def content_id(stem: str) -> str:
    p = stem
    patterns = [r"[-_]+flip$", r"[-_]+(0|90|180|270)[-_]*$", r"[-_]+rot(0|90|180|270)$",
                r"[-_]+copy(-\d+)?$", r"[-_]+\(\d+\)$", r"[-_]+aug\d*$"]
    changed = True
    while changed:
        changed = False
        for pat in patterns:
            new = re.sub(pat, "", p)
            if new != p and len(new) >= 3:
                p, changed = new, True
    return p


def parent_id(cid: str) -> str:
    return re.sub(r"_block_\d+_\d+$", "", cid)


def label_classes(lbl: Path) -> Counter:
    c = Counter()
    if lbl.exists():
        for ln in open(ep(lbl), encoding="utf-8").read().strip().splitlines():
            parts = ln.split()
            if len(parts) == 5:
                c[int(parts[0])] += 1
    return c


# ---- gather ----
records = []
for split in SPLITS:
    img_dir, lbl_dir = SRC / split / "images", SRC / split / "labels"
    for img in img_dir.iterdir():
        st = source_stem(img.name)
        cid = content_id(st)
        records.append({"img": img, "lbl": lbl_dir / (img.stem + ".txt"),
                        "orig_split": split, "stem": st, "cid": cid, "pid": parent_id(cid)})

# representative copy per content (deterministic: shortest then lexicographic name)
by_content = defaultdict(list)
for r in records:
    by_content[r["cid"]].append(r)
rep_of = {cid: sorted(rs, key=lambda x: (len(x["img"].name), x["img"].name))[0]
          for cid, rs in by_content.items()}

# parent-level content vectors (class instances on representative copies)
par_contents = defaultdict(list)
for cid, r in rep_of.items():
    par_contents[r["pid"]].append(cid)
par_vec, par_ncont = {}, {}
for pid, cids in par_contents.items():
    v = Counter()
    for cid in cids:
        v += label_classes(rep_of[cid]["lbl"])
    par_vec[pid] = v
    par_ncont[pid] = len(cids)

total_vec = Counter()
for v in par_vec.values():
    total_vec += v
total_cont = sum(par_ncont.values())
targets = {s: {c: total_vec[c] * RATIOS[s] for c in range(5)} for s in SPLITS}
targets_cont = {s: total_cont * RATIOS[s] for s in SPLITS}

# greedy stratified assignment: rarest-class parents first
rarity = {c: 1.0 / max(1, total_vec[c]) for c in range(5)}
order = sorted(par_vec, key=lambda p: -sum(par_vec[p][c] * rarity[c] for c in range(5)))
cur = {s: Counter() for s in SPLITS}
cur_cont = {s: 0 for s in SPLITS}
assign = {}
for pid in order:
    v, n = par_vec[pid], par_ncont[pid]
    best, best_score = None, None
    for s in SPLITS:
        # deficit-weighted gain, normalized by target; content-count pressure as tiebreak
        gain = sum(rarity[c] * v[c] * max(0.0, targets[s][c] - cur[s][c]) / max(1.0, targets[s][c])
                   for c in range(5))
        pressure = (targets_cont[s] - cur_cont[s]) / targets_cont[s]
        score = gain + 0.05 * pressure
        if best_score is None or score > best_score:
            best, best_score = s, score
    assign[pid] = best
    cur[best] += v
    cur_cont[best] += n

# ---- build ----
if DST.exists():
    shutil.rmtree(ep(DST))
for s in SPLITS:
    (DST / s / "images").mkdir(parents=True)
    (DST / s / "labels").mkdir(parents=True)

manifest_rows, counters = [], {s: 0 for s in SPLITS}
inst = {s: Counter() for s in SPLITS}
imgs_with = {s: Counter() for s in SPLITS}
for pid, cids in par_contents.items():
    s = assign[pid]
    chosen = ([r for r in records if r["pid"] == pid] if s == "train"
              else [rep_of[cid] for cid in cids])
    for r in chosen:
        counters[s] += 1
        nid = f"{s}_{counters[s]:05d}"
        shutil.copyfile(ep(r["img"]), ep(DST / s / "images" / f"{nid}.jpg"))
        text = open(ep(r["lbl"]), encoding="utf-8").read() if r["lbl"].exists() else ""
        with open(ep(DST / s / "labels" / f"{nid}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        cc = label_classes(r["lbl"])
        for c, k in cc.items():
            inst[s][NAMES[c]] += k
        for c in cc:
            imgs_with[s][NAMES[c]] += 1
        manifest_rows.append([nid, s, r["orig_split"], r["img"].name, r["stem"], r["cid"], r["pid"]])

with open(ep(DST / "manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["new_id", "split", "orig_split", "orig_filename", "stem", "content_id", "parent_id"])
    w.writerows(manifest_rows)

with open(ep(DST / "data.yaml"), "w", encoding="utf-8") as f:
    f.write("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
            f"nc: 5\nnames: {NAMES}\n\n"
            "# leakage-free stratified re-split of Roboflow kaid-al welding-defect-final0 v1 (CC BY 4.0)\n"
            "# unit: parent photo; sizing: unique contents 70/20/10; seed 42\n"
            "# train keeps baked augment copies; valid/test deduped to one copy per content\n")

summary = {
    "images": counters,
    "parents": dict(Counter(assign.values())),
    "unique_contents_total": total_cont,
    "instances": {s: dict(inst[s]) for s in SPLITS},
    "images_with_class": {s: dict(imgs_with[s]) for s in SPLITS},
}
with open(ep(BASE / "final0_clean_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
