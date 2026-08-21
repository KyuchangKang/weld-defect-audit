# -*- coding: utf-8 -*-
"""교차 중복을 '고유 이미지 수' 기준으로 재집계 + 해시 캐시 저장."""
import os, re, json
from pathlib import Path
import numpy as np
from PIL import Image
import imagehash

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
PRIOR = Path(r'D:\Sync\Dropbox\paper_wedling2\04_experiments\data\prior_gts\The Welding Defect Dataset')
FINAL0 = Path(r'D:\Sync\Dropbox\paper_wedling2\04_experiments\data\final0')
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def collect(root):
    out = []
    for sp in ('train', 'valid', 'test'):
        d = root / sp / 'images'
        if d.is_dir():
            out += [(sp, f) for f in sorted(d.iterdir()) if f.suffix.lower() in EXT]
    return out

def dih(img):
    hs = []
    for k in range(4):
        r = img.rotate(90 * k, expand=True)
        for m in (r, r.transpose(Image.FLIP_LEFT_RIGHT)):
            hs.append(np.packbits(imagehash.phash(m).hash.flatten()).view(np.uint64)[0])
    return hs

def pc(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56

def hashes(items, cache):
    p = HERE / cache
    if p.exists(): return np.load(p)
    H = np.zeros((len(items), 8), dtype=np.uint64)
    for i, (_, f) in enumerate(items):
        try:
            with Image.open(f) as im: H[i] = dih(im.convert('L'))
        except Exception: pass
    np.save(p, H); return H

A, B = collect(PRIOR), collect(FINAL0)
HA, HB = hashes(A, 'phash_prior.npy'), hashes(B, 'phash_final0.npy')
res = {}
for d in (0, 4, 8):
    matched, matched_by_sp = set(), {}
    tgt_train = set()
    for i in range(len(A)):
        if HA[i, 0] == 0: continue
        dist = pc(np.bitwise_xor(HA[i][None, :], HB[:, 0][:, None])).min(axis=1)
        js = np.nonzero(dist <= d)[0]
        if len(js):
            matched.add(i)
            matched_by_sp.setdefault(A[i][0], set()).add(i)
            if any(B[int(j)][0] == 'train' for j in js): tgt_train.add(i)
    res[str(d)] = {
        'prior_images_with_match': len(matched),
        'pct_of_prior': round(100 * len(matched) / len(A), 1),
        'by_prior_split': {k: len(v) for k, v in sorted(matched_by_sp.items())},
        'prior_test_matching_final0_train': len([i for i in tgt_train if A[i][0] == 'test']),
        'prior_images_matching_final0_train': len(tgt_train),
    }
    r = res[str(d)]
    print('d=%d | [1] 이미지 중 final0에 대응 있음 %4d/%d (%.1f%%) | 분할별 %s'
          % (d, r['prior_images_with_match'], len(A), r['pct_of_prior'], r['by_prior_split']))
    print('      그중 final0.train과 겹치는 [1] 이미지 %d장 (그중 [1].test %d장 / 74)'
          % (r['prior_images_matching_final0_train'], r['prior_test_matching_final0_train']))
json.dump({'prior_n': len(A), 'final0_n': len(B), 'distinct': res},
          open(HERE / 'overlap_distinct.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\n저장:', HERE / 'overlap_distinct.json')
