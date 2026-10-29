# -*- coding: utf-8 -*-
"""V1 감사 교차 검증 (evaluation_plan v1.2, 사용자 승인 2026-08-14).

파일명 계보 기반 그룹핑(콘텐츠/부모)을 지각 해싱으로 교차 검증:
  (1) 같은 콘텐츠 그룹(증강 사본) 쌍이 실제로 지각적 중복인지 — dihedral 불변 pHash 거리
  (2) 파일명이 놓친 근접중복(다른 부모인데 pHash 근접) 탐지
  (3) final0_clean 배정 파일들에서 스플릿 간 근접중복 잔존 전수 검사 (핵심)
출력: 05_results/upgrade/phash_validation.json
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

from weldpaths import project_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
SRC = BASE / "04_experiments" / "data" / "final0"
MANIFEST = BASE / "04_experiments" / "data" / "final0_clean" / "manifest.csv"
OUT = BASE / "05_results" / "upgrade" / "phash_validation.json"

THRESHOLDS = [0, 4, 8]


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


def dihedral_hashes(img):
    """8개 dihedral 변환 각각의 pHash (uint64)."""
    hs = []
    for k in range(4):
        r = img.rotate(90 * k, expand=True)
        for m in (r, r.transpose(Image.FLIP_LEFT_RIGHT)):
            h = imagehash.phash(m)
            hs.append(np.packbits(h.hash.flatten()).view(np.uint64)[0])
    return hs


def main():
    files, cids, pids = [], [], []
    for split in ("train", "valid", "test"):
        for img in sorted((SRC / split / "images").iterdir()):
            st = source_stem(img.name)
            files.append(img)
            cids.append(content_id(st))
            pids.append(parent_id(cid=content_id(st)))
    n = len(files)
    print(f"hashing {n} images x8 dihedral ...")
    H = np.zeros((n, 8), dtype=np.uint64)
    for i, f in enumerate(files):
        with Image.open(f) as im:
            H[i] = dihedral_hashes(im.convert("L"))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")

    h0 = H[:, 0].copy()
    cid_arr = np.array(cids)
    pid_arr = np.array(pids)

    # 최소 dihedral 거리 근접쌍 탐색 (chunked)
    near_pairs = []  # (i, j, dmin)
    CH = 512
    maxthr = max(THRESHOLDS)
    for s in range(0, n, CH):
        e = min(s + CH, n)
        dmin = np.full((e - s, n), 64, dtype=np.uint8)
        for t in range(8):
            x = np.bitwise_xor.outer(h0[s:e], H[:, t])
            d = np.bitwise_count(x).astype(np.uint8)
            np.minimum(dmin, d, out=dmin)
        ii, jj = np.nonzero(dmin <= maxthr)
        for a, b in zip(ii, jj):
            gi, gj = s + int(a), int(b)
            if gi < gj:
                near_pairs.append((gi, gj, int(dmin[a, b])))
        if s % (CH * 4) == 0:
            print(f"  pairs scan {s}/{n}")
    print(f"near pairs (d<= {maxthr}): {len(near_pairs)}")

    # (1) 같은 콘텐츠 쌍의 확인율 (파일명 그룹핑의 픽셀 검증)
    by_c = defaultdict(list)
    for i, c in enumerate(cids):
        by_c[c].append(i)
    same_content_pairs = sum(len(v) * (len(v) - 1) // 2 for v in by_c.values())
    near_map = {}
    for i, j, d in near_pairs:
        near_map[(i, j)] = d
    confirmed = {thr: 0 for thr in THRESHOLDS}
    worst = []
    for c, idxs in by_c.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = sorted((idxs[a], idxs[b]))
                d = near_map.get((i, j), 64)
                for thr in THRESHOLDS:
                    if d <= thr:
                        confirmed[thr] += 1
                if d > 8:
                    worst.append((files[i].name, files[j].name, d))

    # (2) 다른 부모인데 근접중복 = 파일명이 놓친 후보
    missed = [(i, j, d) for i, j, d in near_pairs if pid_arr[i] != pid_arr[j]]

    # (3) final0_clean 스플릿 간 잔존 검사 (배정 파일만)
    split_of = {}
    with open(MANIFEST, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split_of[row["orig_filename"]] = row["split"]
    residual = {thr: [] for thr in THRESHOLDS}
    for i, j, d in missed:
        si = split_of.get(files[i].name)
        sj = split_of.get(files[j].name)
        if si and sj and si != sj:
            for thr in THRESHOLDS:
                if d <= thr:
                    residual[thr].append(
                        {"a": files[i].name, "a_split": si, "b": files[j].name,
                         "b_split": sj, "dist": d,
                         "a_parent": pids[i], "b_parent": pids[j]})

    out = {
        "n_images": n,
        "method": "pHash 64-bit, min over 8 dihedral transforms",
        "same_content_pairs_total": same_content_pairs,
        "same_content_confirmed_at": {
            str(t): confirmed[t] for t in THRESHOLDS},
        "same_content_confirm_rate_d8": round(
            confirmed[8] / max(1, same_content_pairs), 4),
        "same_content_unconfirmed_examples": worst[:20],
        "cross_parent_near_pairs": {
            str(t): sum(1 for _, _, d in missed if d <= t) for t in THRESHOLDS},
        "cross_parent_examples": [
            {"a": files[i].name, "b": files[j].name, "d": d,
             "a_parent": pids[i], "b_parent": pids[j]}
            for i, j, d in sorted(missed, key=lambda x: x[2])[:30]],
        "CLEAN_cross_split_residual": {
            str(t): {"count": len(residual[t]), "pairs": residual[t][:30]}
            for t in THRESHOLDS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("cross_parent_examples",
                                   "same_content_unconfirmed_examples")},
                     ensure_ascii=False, indent=1, default=str)[:2000])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
