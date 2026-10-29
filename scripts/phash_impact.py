# -*- coding: utf-8 -*-
"""V1 후속: 잔존 근접중복의 영향 범위 + 픽셀 수준 누수율 재추정.

(1) final0_clean 스플릿 간 잔존 근접중복(d<=4)에 연루된 test/valid 파일 목록·수
    → test 오염 규모 산정 + bootstrap_ci.py의 민감도 재집계 입력
(2) 원 스플릿 누수율의 픽셀 수준 재추정: dihedral pHash d<=4 연결 성분(지각 중복 클러스터)
    기준으로 스플릿을 걸치는 클러스터 비율 — 파일명 기반 57%(부모 수준)의 교차 추정치
해시는 npy로 저장해 재사용. 출력: 05_results/upgrade/phash_impact.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

from weldpaths import project_dir, runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
SRC = BASE / "04_experiments" / "data" / "final0"
MANIFEST = BASE / "04_experiments" / "data" / "final0_clean" / "manifest.csv"
HASH_NPY = runs_dir(r"D:\runs\weld2") / "upgrade" / "phash_dihedral.npy"
NAMES_NPY = runs_dir(r"D:\runs\weld2") / "upgrade" / "phash_files.json"
OUT = BASE / "05_results" / "upgrade" / "phash_impact.json"
THR = 4


def dihedral_hashes(img):
    hs = []
    for k in range(4):
        r = img.rotate(90 * k, expand=True)
        for m in (r, r.transpose(Image.FLIP_LEFT_RIGHT)):
            h = imagehash.phash(m)
            hs.append(np.packbits(h.hash.flatten()).view(np.uint64)[0])
    return hs


def main():
    import csv

    files, orig_split = [], []
    for split in ("train", "valid", "test"):
        for img in sorted((SRC / split / "images").iterdir()):
            files.append(img)
            orig_split.append(split)
    n = len(files)

    if HASH_NPY.exists() and NAMES_NPY.exists():
        H = np.load(HASH_NPY)
        saved = json.loads(NAMES_NPY.read_text(encoding="utf-8"))
        assert saved == [f.name for f in files], "file list mismatch"
        print("hashes loaded from cache")
    else:
        print(f"hashing {n} images x8 ...")
        H = np.zeros((n, 8), dtype=np.uint64)
        for i, f in enumerate(files):
            with Image.open(f) as im:
                H[i] = dihedral_hashes(im.convert("L"))
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{n}")
        HASH_NPY.parent.mkdir(parents=True, exist_ok=True)
        np.save(HASH_NPY, H)
        NAMES_NPY.write_text(json.dumps([f.name for f in files]), encoding="utf-8")

    # 근접쌍 (d<=THR)
    h0 = H[:, 0].copy()
    pairs = []
    CH = 512
    for s in range(0, n, CH):
        e = min(s + CH, n)
        dmin = np.full((e - s, n), 64, dtype=np.uint8)
        for t in range(8):
            x = np.bitwise_xor.outer(h0[s:e], H[:, t])
            np.minimum(dmin, np.bitwise_count(x).astype(np.uint8), out=dmin)
        ii, jj = np.nonzero(dmin <= THR)
        for a, b in zip(ii, jj):
            gi, gj = s + int(a), int(b)
            if gi < gj:
                pairs.append((gi, gj, int(dmin[a, b])))
    print(f"pairs d<={THR}: {len(pairs)}")

    # (1) final0_clean 잔존 영향 파일
    split_of, newid_of = {}, {}
    with open(MANIFEST, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split_of[row["orig_filename"]] = row["split"]
            newid_of[row["orig_filename"]] = row["new_id"]
    affected = defaultdict(set)  # clean split -> set of new_id
    residual_pairs = 0
    for i, j, d in pairs:
        si, sj = split_of.get(files[i].name), split_of.get(files[j].name)
        if si and sj and si != sj:
            residual_pairs += 1
            affected[si].add(newid_of[files[i].name])
            affected[sj].add(newid_of[files[j].name])
    impact = {s: sorted(affected[s]) for s in affected}

    # (2) 픽셀 수준 클러스터 누수율 (원 스플릿 기준, union-find)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, _ in pairs:
        union(i, j)
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    multi = {k: v for k, v in comps.items() if len(v) >= 2}
    spanning = sum(1 for v in multi.values()
                   if len(set(orig_split[i] for i in v)) > 1)
    files_in_spanning = sum(len(v) for v in multi.values()
                            if len(set(orig_split[i] for i in v)) > 1)

    out = {
        "threshold": THR,
        "residual_cross_split_pairs_in_clean": residual_pairs,
        "affected_clean_files": {s: {"count": len(v), "ids": v}
                                 for s, v in impact.items()},
        "pixel_level_official_split": {
            "perceptual_clusters_size_ge2": len(multi),
            "clusters_spanning_splits": spanning,
            "spanning_fraction": round(spanning / max(1, len(multi)), 4),
            "files_in_spanning_clusters": files_in_spanning,
            "files_fraction": round(files_in_spanning / n, 4),
        },
        "note": ("dihedral pHash d<=4는 회전/플립 사본과 재샷 근접중복을 잡지만 "
                 "크롭/타일·강한 밝기 변형은 놓침 — spanning_fraction은 하한 추정"),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("residual_cross_split_pairs_in_clean",
                       "pixel_level_official_split")}, indent=1))
    print("affected counts:", {s: len(v) for s, v in impact.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
