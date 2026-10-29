# -*- coding: utf-8 -*-
"""원 test 310장을 부모 사진 단위(230개)로 묶어 대표 이미지 시트를 만든다 (M2 도메인 판정용)."""
import os, re, json, math, collections
from PIL import Image, ImageDraw

import pathlib as _pl, sys as _sys  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
RAW_UN = str(runs_dir(r"D:\runs\weld2") / "review4" / "test_unseen" / "images")
RAW_SE = str(runs_dir(r"D:\runs\weld2") / "review4" / "test_seen" / "images")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'parent_sheets'); os.makedirs(OUTD, exist_ok=True)

def source_stem(n):
    s = re.sub(r'\.rf\.[0-9a-f]{32}.*$', '', n)
    return re.sub(r'_(jpeg|jpg|png|bmp)(_(jpeg|jpg|png|bmp))*$', '', s, flags=re.I).lower()
def content_id(st):
    p, pats = st, [r'[-_]+flip$', r'[-_]+(0|90|180|270)[-_]*$', r'[-_]+rot(0|90|180|270)$',
                   r'[-_]+copy(-\d+)?$', r'[-_]+\(\d+\)$', r'[-_]+aug\d*$']
    ch = True
    while ch:
        ch = False
        for pt in pats:
            new = re.sub(pt, '', p)
            if new != p and len(new) >= 3: p, ch = new, True
    return p
def parent_id(n): return re.sub(r'_block_\d+_\d+$', '', content_id(source_stem(n)))

g = collections.OrderedDict()
for lbl, d in (('seen', RAW_SE), ('unseen', RAW_UN)):
    for f in sorted(os.listdir(d)):
        g.setdefault(parent_id(f), []).append((lbl, d, f))

parents = list(g.items())
index = [{'i': i + 1, 'parent': p, 'split': v[0][0], 'n_files': len(v), 'rep': v[0][2]}
         for i, (p, v) in enumerate(parents)]
json.dump(index, open(os.path.join(HERE, 'parent_index.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

PER, C, TH, PAD, LAB = 30, 6, 250, 8, 30
for s in range(0, len(parents), PER):
    chunk = parents[s:s + PER]
    R = math.ceil(len(chunk) / C)
    W, H = C * (TH + PAD) + PAD, R * (TH + LAB + PAD) + PAD
    sheet = Image.new('RGB', (W, H), 'white'); dr = ImageDraw.Draw(sheet)
    for k, (p, v) in enumerate(chunk):
        lbl, d, f = v[0]
        im = Image.open(os.path.join(d, f)).convert('RGB'); im.thumbnail((TH, TH))
        x = PAD + (k % C) * (TH + PAD); y = PAD + (k // C) * (TH + LAB + PAD)
        sheet.paste(im, (x + (TH - im.width) // 2, y))
        dr.text((x, y + TH + 3), '#%d [%s] %s' % (s + k + 1, lbl[:2].upper(), p[:30]), fill='black')
        dr.text((x, y + TH + 16), 'n=%d' % len(v), fill='gray')
    op = os.path.join(OUTD, 'parents_%03d-%03d.png' % (s + 1, s + len(chunk)))
    sheet.save(op); print('saved', os.path.basename(op), sheet.size)
print('총 부모', len(parents))
