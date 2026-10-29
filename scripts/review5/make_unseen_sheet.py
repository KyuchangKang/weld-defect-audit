# -*- coding: utf-8 -*-
"""이분 대조의 unseen-parent 30장 육안 확인 시트 (M2 도메인 판정용)."""
import os, math
from PIL import Image, ImageDraw
import pathlib as _pl, sys as _sys  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
SRC = str(runs_dir(r"D:\runs\weld2") / "review4" / "test_unseen" / "images")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unseen30_sheet.png')
fs = sorted(os.listdir(SRC))
C, TH, PAD, LAB = 6, 260, 8, 26
R = math.ceil(len(fs) / C)
W, H = C * (TH + PAD) + PAD, R * (TH + LAB + PAD) + PAD
sheet = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(sheet)
for i, f in enumerate(fs):
    im = Image.open(os.path.join(SRC, f)).convert('RGB'); im.thumbnail((TH, TH))
    x = PAD + (i % C) * (TH + PAD); y = PAD + (i // C) * (TH + LAB + PAD)
    sheet.paste(im, (x + (TH - im.width) // 2, y))
    name = f.split('.rf.')[0]
    d.text((x, y + TH + 4), '%02d %s' % (i + 1, name[:34]), fill='black')
sheet.save(OUT)
print('saved', OUT, sheet.size, '| %d images' % len(fs))
for i, f in enumerate(fs, 1):
    print('%02d  %s' % (i, f.split('.rf.')[0]))
