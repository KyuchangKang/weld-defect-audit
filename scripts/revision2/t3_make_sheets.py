# -*- coding: utf-8 -*-
"""T3 (R1-2): draw the audit units (design: results/T3_표본설계.md) and render blinded judgment sheets.

Outputs (results/T3/):
  key.json                 hidden key: id -> stratum, files, distance (do NOT open before judging)
  sheets_claude/P###.jpg   pair sheets, 8 pairs each, all pair units mixed and shuffled (ids P0001..)
  sheets_claude/G##.jpg    group sheets (U1 content groups, U2 tiled parents), ids G01..
  sheets_author/           stratified subset for the second reader, re-numbered (A01.., AG1..)
  판독표_저자.csv           recording form for the second reader
Sheets show only the random id and the image panels: no filename, distance, split or stratum.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir, runs_dir  # noqa: E402,F401  (path defaults; see weldpaths.py)
_T3 = Path(__file__).resolve().parents[2] / "results" / "revision2" / "T3"  # was work/results/T3

import numpy as np
from PIL import Image, ImageDraw, ImageFont

exec(open(Path(__file__).with_name("t3_population.py"), encoding="utf-8").read().split("def pc(x)")[0])
OUTD = OUT
Z = np.load(OUTD / "pairs.npz")
I, J, D, SC, SP = Z["I"], Z["J"], Z["D"], Z["same_c"], Z["same_p"]
rng = random.Random(2026)


def pcnt(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def dist(a, b):
    return int(pcnt(np.bitwise_xor(H[a, 0], H[b, :])).min())


# ---------- pair units ----------
units = []          # dicts: stratum, a, b, d, extra
# parent links: closest pair per parent pair
link = {}
for a, b, d, sp in zip(I, J, D, SP):
    if sp:
        continue
    k = tuple(sorted((pid[a], pid[b])))
    if k not in link or d < link[k][2]:
        link[k] = (int(a), int(b), int(d))
links = sorted(link.items())
L4 = [v for k, v in links if v[2] <= 4]
L58 = [v for k, v in links if 4 < v[2] <= 8]
L912 = [v for k, v in links if 8 < v[2] <= 12]
for a, b, d in L4:
    units.append(dict(stratum="U3_link_le4", a=a, b=b, d=d))
for a, b, d in rng.sample(L58, 100):
    units.append(dict(stratum="U4_link_5_8", a=a, b=b, d=d))
for a, b, d in rng.sample(L912, 50):
    units.append(dict(stratum="U5_link_9_12", a=a, b=b, d=d))

# U6: clean test images with a cross-split partner at d<=12 -> nearest partner in clean train/valid
man = list(csv.DictReader(open(BASE / "04_experiments/data/final0_clean/manifest.csv", encoding="utf-8-sig")))
key = {(r["orig_split"], r["orig_filename"]): (r["split"], r["new_id"]) for r in man}
cl = [key.get(f) for f in files]
best_t = {}
for a, b, d in zip(I, J, D):
    if d > 12 or not cl[a] or not cl[b] or cl[a][0] == cl[b][0]:
        continue
    for t, o in ((a, b), (b, a)):
        if cl[t][0] == "test" and cl[o][0] != "test":
            if t not in best_t or d < best_t[t][1]:
                best_t[t] = (int(o), int(d))
for t, (o, d) in sorted(best_t.items()):
    units.append(dict(stratum="U6_test_nearest", a=int(t), b=o, d=d, test_id=cl[t][1]))

# controls
conf = [(int(a), int(b), int(d)) for a, b, d, s in zip(I, J, D, SC) if s and 1 <= d <= 8]
for a, b, d in rng.sample(conf, 30):
    units.append(dict(stratum="C_pos_same_content", a=a, b=b, d=d))
neg = 0
while neg < 30:
    a, b = rng.randrange(len(files)), rng.randrange(len(files))
    if pid[a] == pid[b]:
        continue
    d = dist(a, b)
    if d > 16:
        units.append(dict(stratum="C_neg_far", a=a, b=b, d=d))
        neg += 1

# de-duplicate identical image pairs across strata (keep all stratum labels)
merged = {}
for u in units:
    k = tuple(sorted((u["a"], u["b"])))
    if k in merged:
        merged[k]["strata"].append(u["stratum"])
        merged[k].update({x: u[x] for x in u if x == "test_id"})
    else:
        merged[k] = dict(u, strata=[u["stratum"]])
pairs = list(merged.values())
rng.shuffle(pairs)
for n, p in enumerate(pairs, 1):
    p["id"] = f"P{n:04d}"
    p["files"] = [files[p["a"]], files[p["b"]]]

# ---------- group units ----------
grp = defaultdict(list)
for k, c in enumerate(cid):
    grp[c].append(k)
unconf = sorted({cid[a] for a, b, d, s in zip(I, J, D, SC) if s and d > 8})


def components(members, thr=8):
    par = {m: m for m in members}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    ms = set(members)
    for a, b, d in zip(I, J, D):
        if a in ms and b in ms and d <= thr:
            par[f(a)] = f(b)
    comp = defaultdict(list)
    for m in members:
        comp[f(m)].append(m)
    return sorted(comp.values(), key=lambda v: (-len(v), v[0]))


groups = []
for c in unconf:
    comps = components(grp[c])
    groups.append(dict(stratum="U1_content_group", content=c,
                       reps=[int(v[0]) for v in comps], sizes=[len(v) for v in comps]))
pgrp = defaultdict(lambda: defaultdict(list))
for k, (c, p) in enumerate(zip(cid, pid)):
    pgrp[p][c].append(k)
for p, cs in sorted(pgrp.items()):
    if len(cs) > 1:
        groups.append(dict(stratum="U2_tiled_parent", parent=p,
                           reps=[int(sorted(v)[0]) for _, v in sorted(cs.items())], sizes=[len(v) for _, v in sorted(cs.items())]))
rng.shuffle(groups)
for n, g in enumerate(groups, 1):
    g["id"] = f"G{n:02d}"
    g["files"] = [files[r] for r in g["reps"]]

# ---------- author subset (stratified, re-numbered) ----------
want = {"U3_link_le4": 15, "U4_link_5_8": 15, "U5_link_9_12": 10, "U6_test_nearest": 10,
        "C_pos_same_content": 5, "C_neg_far": 5}
sub = []
for s, k in want.items():
    pool = [p for p in pairs if p["strata"][0] == s]
    sub += rng.sample(pool, min(k, len(pool)))
rng.shuffle(sub)
author_pairs = [dict(aid=f"A{n:02d}", pid=p["id"]) for n, p in enumerate(sub, 1)]
gsub = rng.sample([g for g in groups if g["stratum"] == "U1_content_group"], 4) + \
    rng.sample([g for g in groups if g["stratum"] == "U2_tiled_parent"], 1)
rng.shuffle(gsub)
author_groups = [dict(aid=f"AG{n}", gid=g["id"]) for n, g in enumerate(gsub, 1)]

# ---------- rendering ----------
try:
    FONT = ImageFont.truetype("arial.ttf", 22)
    SMALL = ImageFont.truetype("arial.ttf", 16)
except OSError:
    FONT = SMALL = ImageFont.load_default()
TH = 380


def thumb(k, size=TH):
    sp, name = files[k]
    im = Image.open(SRC / sp / "images" / name).convert("RGB")
    im.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (40, 40, 40))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def pair_sheets(items, label_key, folder, prefix):
    folder.mkdir(parents=True, exist_ok=True)
    per = 8
    names = []
    for s in range(0, len(items), per):
        chunk = items[s:s + per]
        W, Hh = 2 * (2 * TH + 30), 4 * (TH + 40)
        sheet = Image.new("RGB", (W, Hh), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for n, it in enumerate(chunk):
            x0, y0 = (n % 2) * (2 * TH + 30), (n // 2) * (TH + 40)
            d.text((x0 + 6, y0 + 6), it[label_key], fill=(200, 0, 0), font=FONT)
            sheet.paste(thumb(it["a"]), (x0, y0 + 36))
            sheet.paste(thumb(it["b"]), (x0 + TH + 4, y0 + 36))
        name = f"{prefix}{s // per + 1:03d}.jpg"
        sheet.save(folder / name, quality=90)
        names.append(name)
    return names


def group_sheet(g, label, folder):
    reps = g["reps"]
    n = len(reps)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    t = 300
    sheet = Image.new("RGB", (cols * (t + 8), rows * (t + 34) + 40), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), label, fill=(200, 0, 0), font=FONT)
    for i, r in enumerate(reps):
        x0, y0 = (i % cols) * (t + 8), 40 + (i // cols) * (t + 34)
        d.text((x0 + 4, y0 + 2), chr(65 + i) if i < 26 else f"Z{i}", fill=(0, 0, 160), font=SMALL)
        sheet.paste(thumb(r, t), (x0, y0 + 26))
    sheet.save(folder / f"{label}.jpg", quality=90)


def main():
    cdir, adir = OUTD / "sheets_claude", OUTD / "sheets_author"
    for d_ in (cdir, adir):
        d_.mkdir(parents=True, exist_ok=True)
        for f in d_.glob("*.jpg"):
            f.unlink()
    pair_sheets(pairs, "id", cdir, "P")
    for g in groups:
        group_sheet(g, g["id"], cdir)
    byid = {p["id"]: p for p in pairs}
    a_items = [dict(byid[x["pid"]], aid=x["aid"]) for x in author_pairs]
    pair_sheets(a_items, "aid", adir, "A")
    gbyid = {g["id"]: g for g in groups}
    for x in author_groups:
        group_sheet(gbyid[x["gid"]], x["aid"], adir)
    key = {"pairs": pairs, "groups": groups, "author_pairs": author_pairs, "author_groups": author_groups,
           "counts": {"links_le4": len(L4), "links_5_8": len(L58), "links_9_12": len(L912),
                      "U6_test_images": len(best_t), "pairs_total_after_dedup": len(pairs),
                      "groups_U1": sum(g["stratum"] == "U1_content_group" for g in groups),
                      "groups_U2": sum(g["stratum"] == "U2_tiled_parent" for g in groups)}}
    (OUTD / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(OUTD / "판독표_저자_blank.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "판정 — 쌍(A01~A60)은 SAME / ADJ / DIFF / UNSURE; 그룹(AG1~AG5)은 묶음 표기(예: AB/C/D/E)", "메모"])
        for x in author_pairs:
            w.writerow([x["aid"], "", ""])
        for x in author_groups:
            w.writerow([x["aid"], "", ""])
    with open(OUTD / "판독표_Claude.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "judgment", "note"])
        for p in pairs:
            w.writerow([p["id"], "", ""])
        for g in groups:
            w.writerow([g["id"], "", ""])
    print(json.dumps(key["counts"], indent=1))
    print("claude sheets:", len(list(cdir.glob("P*.jpg"))), "pair sheets +", len(list(cdir.glob("G*.jpg"))), "group sheets")
    print("author sheets:", len(list(adir.glob("A*.jpg"))), "pair sheets +", len(list(adir.glob("AG*.jpg"))), "group sheets")


if __name__ == "__main__":
    main()
