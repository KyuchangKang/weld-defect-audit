# -*- coding: utf-8 -*-
"""R1-4 / R2-4: 부모 그루핑 규칙 민감도 (학습 0회, 해시 집계만).
G0 파일명 계보(논문 기준) / G1 픽셀 확인 분할(과대그루핑 제거: 같은 부모 안에서도 d<=8로 이어진 것만 한 부모)
G2 인접샷 병합(과소그루핑 제거: d<=4 교차 부모 연결을 합침) / G3 G1+G2
각 규칙에서 부모 수, 중복배수(파일/부모), 공식 분할을 가로지르는 부모 비율."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import imagehash

import pathlib as _pl, sys as _sys  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
SRC = BASE / "04_experiments" / "data" / "final0"
OUT = BASE / "08_revision" / "1st_revision" / "work" / "results" / "grouping_sensitivity.json"
CACHE = BASE / "08_revision" / "1st_revision" / "work" / "results" / "dihedral_phash_cache.npz"
sys.path.insert(0, str(BASE / "04_experiments" / "scripts"))

def source_stem(name):
    s = re.sub(r"\.rf\.[0-9a-f]{32}.*$", "", name)
    s = re.sub(r"_(jpeg|jpg|png|bmp)(_(jpeg|jpg|png|bmp))*$", "", s, flags=re.I)
    return s.lower()

def content_id(stem):
    p = stem
    pats = [r"[-_]+flip$", r"[-_]+(0|90|180|270)[-_]*$", r"[-_]+rot(0|90|180|270)$",
            r"[-_]+copy(-\d+)?$", r"[-_]+\(\d+\)$", r"[-_]+aug\d*$"]
    changed = True
    while changed:
        changed = False
        for pat in pats:
            new = re.sub(pat, "", p)
            if new != p and len(new) >= 3:
                p, changed = new, True
    return p

def parent_id(cid):
    return re.sub(r"_block_\d+_\d+$", "", cid)

files = []
for split in ["train", "valid", "test"]:
    for img in sorted((SRC / split / "images").iterdir()):
        files.append((split, img))
N = len(files)
if CACHE.exists():
    H = np.load(CACHE)["H"]
else:
    H = np.zeros((N, 8), dtype=np.uint64)
    for i, (_, p) in enumerate(files):
        im = Image.open(p).convert("L")
        k = 0
        for r in range(4):
            rr = im.rotate(90 * r, expand=True)
            for m in (rr, rr.transpose(Image.FLIP_LEFT_RIGHT)):
                H[i, k] = np.packbits(imagehash.phash(m).hash.flatten()).view(np.uint64)[0]; k += 1
        if i % 500 == 0: print("hash", i, flush=True)
    np.savez(CACHE, H=H)

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def dist_row(i, J):
    x = H[J, :] ^ H[i, 0]
    return POP[x.view(np.uint8).reshape(len(J), 8, 8)].sum(axis=2).min(axis=1)

split_of = [s for s, _ in files]
pid0 = [parent_id(content_id(source_stem(p.name))) for _, p in files]

class DSU:
    def __init__(s, n): s.p = list(range(n))
    def f(s, a):
        while s.p[a] != a: s.p[a] = s.p[s.p[a]]; a = s.p[a]
        return a
    def u(s, a, b): s.p[s.f(a)] = s.f(b)

by_pid = defaultdict(list)
for i, p in enumerate(pid0): by_pid[p].append(i)

# within-parent confirmed edges (d<=8) and cross-parent near edges (d<=4)
g1 = DSU(N)
for p, idx in by_pid.items():
    idx = np.array(idx)
    for a in range(len(idx)):
        if a + 1 < len(idx):
            d = dist_row(idx[a], idx[a + 1:])
            for b in idx[a + 1:][d <= 8]: g1.u(idx[a], b)
cross = []
allidx = np.arange(N)
for i in range(N - 1):
    J = allidx[i + 1:]
    d = dist_row(i, J)
    for j in J[d <= 4]:
        if pid0[i] != pid0[j]: cross.append((i, int(j)))
print("cross-parent d<=4 pairs", len(cross), flush=True)

def summarize(labels, name):
    grp = defaultdict(set)
    for i, g in enumerate(labels): grp[g].add(split_of[i])
    n = len(grp); leak = sum(1 for v in grp.values() if len(v) > 1)
    return {"rule": name, "parents": n, "redundancy": round(N / n, 2), "leaking": leak, "leak_pct": round(100 * leak / n, 1)}

res = {"n_files": N, "cross_parent_d4_pairs": len(cross)}
res["G0_filename"] = summarize(pid0, "filename lineage (paper)")
res["G1_pixel_confirmed"] = summarize([g1.f(i) for i in range(N)], "split filename parents into d<=8-connected components")
g2 = DSU(N)
for p, idx in by_pid.items():
    for b in idx[1:]: g2.u(idx[0], b)
for i, j in cross: g2.u(i, j)
res["G2_adjacent_merged"] = summarize([g2.f(i) for i in range(N)], "merge filename parents linked at d<=4")
g3 = DSU(N)
for i in range(N): g3.u(i, g1.f(i))
for i, j in cross: g3.u(i, j)
res["G3_split_and_merged"] = summarize([g3.f(i) for i in range(N)], "G1 then merge d<=4 links")
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
