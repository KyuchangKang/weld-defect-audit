# -*- coding: utf-8 -*-
"""M5b 보강 — 선행논문 [1] 데이터셋과 본 연구 final0 사이의 이미지 공유 검사.

두 배포본 모두 Roboflow 익스포트(.rf.<hash>)이므로 (a) 소스 stem 대조와
(b) dihedral 불변 pHash 대조를 함께 수행한다. 특히 [1]의 test가 final0의 train과
겹치는지가 중요하다(겹치면 교차 데이터셋 오염).

출력: evidence/cross_dataset_overlap.json
"""
import os, re, json, hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
import pathlib as _pl, sys as _sys  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir  # noqa: E402  (path defaults; see weldpaths.py)
PRIOR = project_dir(r"D:\Sync\Dropbox\paper_wedling2") / "04_experiments" / "data" / "prior_gts" / "The Welding Defect Dataset"
FINAL0 = project_dir(r"D:\Sync\Dropbox\paper_wedling2") / "04_experiments" / "data" / "final0"
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def stem(name):
    s = re.sub(r'\.rf\.[0-9a-f]{32}.*$', '', name)
    s = re.sub(r'\.(jpg|jpeg|png|bmp|webp)$', '', s, flags=re.I)
    s = re.sub(r'_(jpg|jpeg|png|bmp)$', '', s, flags=re.I)
    # [1] 배포본은 앞에 uuid 조각이 붙은 경우가 있다: 02fd0af7-51ab46bb-c13
    s = re.sub(r'^[0-9a-f]{8}-[0-9a-f]{8}-', '', s, flags=re.I)
    return s.lower()


def collect(root):
    out = []
    for sp in ('train', 'valid', 'test'):
        d = root / sp / 'images'
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                out.append((sp, f))
    return out


def dih(img):
    hs = []
    for k in range(4):
        r = img.rotate(90 * k, expand=True)
        for m in (r, r.transpose(Image.FLIP_LEFT_RIGHT)):
            hs.append(np.packbits(imagehash.phash(m).hash.flatten()).view(np.uint64)[0])
    return hs


def popcount64(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


A = collect(PRIOR)      # [1] 데이터셋
B = collect(FINAL0)     # 본 연구 원 배포본
print('[1] 데이터셋 %d장 / final0 %d장' % (len(A), len(B)))

# ---------- (a) stem 대조 ----------
sa = defaultdict(list); sb = defaultdict(list)
for sp, f in A: sa[stem(f.name)].append(sp)
for sp, f in B: sb[stem(f.name)].append(sp)
shared = sorted(set(sa) & set(sb))
print('\n=== (a) 소스 stem 공유 ===')
print('  [1] 고유 stem %d / final0 고유 stem %d / 공유 %d' % (len(sa), len(sb), len(shared)))
mat = defaultdict(int)
for s in shared:
    for x in set(sa[s]):
        for y in set(sb[s]):
            mat['%s->%s' % (x, y)] += 1
for k in sorted(mat): print('    [1].%-6s x final0.%-6s : %d' % tuple(k.split('->') + [mat[k]]))
print('  예시:', shared[:10])

# ---------- (b) MD5 ----------
ma = {}
for sp, f in A: ma.setdefault(md5(f), []).append(sp)
mb = {}
for sp, f in B: mb.setdefault(md5(f), []).append(sp)
md5_shared = set(ma) & set(mb)
print('\n=== (b) MD5 동일 파일 ===')
print('  동일 바이트 이미지 %d장' % len(md5_shared))

# ---------- (c) pHash ----------
print('\n=== (c) dihedral pHash 대조 (계산 중) ===')
def hashes(items, label):
    H = np.zeros((len(items), 8), dtype=np.uint64)
    for i, (sp, f) in enumerate(items):
        try:
            with Image.open(f) as im:
                H[i] = dih(im.convert('L'))
        except Exception:
            pass
        if (i + 1) % 1000 == 0: print('   %s %d/%d' % (label, i + 1, len(items)))
    return H

HA = hashes(A, '[1]'); HB = hashes(B, 'final0')
res = {}
for d in (0, 4, 8):
    pairs = []
    for i in range(len(A)):
        if HA[i, 0] == 0: continue
        dist = popcount64(np.bitwise_xor(HA[i][None, :], HB[:, 0][:, None])).min(axis=1)
        for j in np.nonzero(dist <= d)[0]:
            pairs.append((A[i][0], B[int(j)][0]))
    cnt = defaultdict(int)
    for x, y in pairs: cnt['%s->%s' % (x, y)] += 1
    imgs_a = len({i for i, _ in enumerate(pairs)})
    res[str(d)] = {'pairs': len(pairs), 'by_split': dict(cnt)}
    print('  d=%d | 교차 근접쌍 %5d' % (d, len(pairs)))
    for k in sorted(cnt): print('        [1].%-6s x final0.%-6s : %d' % tuple(k.split('->') + [cnt[k]]))

out = {'prior_n': len(A), 'final0_n': len(B),
       'stem_shared': len(shared), 'stem_shared_examples': shared[:40],
       'stem_split_matrix': dict(mat), 'md5_identical': len(md5_shared),
       'phash_cross': res}
json.dump(out, open(HERE / 'cross_dataset_overlap.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('\n저장:', HERE / 'cross_dataset_overlap.json')
