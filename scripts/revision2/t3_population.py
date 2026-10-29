# -*- coding: utf-8 -*-
"""T3 (R1-2): sampling frames for the grouping audit. Reuses the cached dihedral pHash array
(07_submission/.../evidence/phash_final0.npy, order = train/valid/test, sorted names) and the grouping
rules of 04_experiments/scripts/phash_validate.py. Writes results/T3/population.json and pairs.npz.
No image is judged here."""
import importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir, runs_dir  # noqa: E402,F401  (path defaults; see weldpaths.py)
_T3 = Path(__file__).resolve().parents[2] / "results" / "revision2" / "T3"  # was work/results/T3
import numpy as np

BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
SRC = BASE / "04_experiments" / "data" / "final0"
H = np.load(_T3 / "phash_final0.npy")
OUT = _T3
spec = importlib.util.spec_from_file_location("pv", Path(__file__).resolve().parents[1] / "phash_validate.py")
src = open(spec.origin, encoding="utf-8").read().split("def dihedral_hashes")[0]
ns = {}; exec(src, ns)
source_stem, content_id, parent_id = ns["source_stem"], ns["content_id"], ns["parent_id"]

files = []
for sp in ("train", "valid", "test"):
    for f in sorted((SRC / sp / "images").iterdir()):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            files.append((sp, f.name))
assert len(files) == len(H) == 4958
stem = [source_stem(n) for _, n in files]
cid = [content_id(s) for s in stem]
pid = [parent_id(c) for c in cid]

def pc(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

n = len(H)
I, J, D = [], [], []
for i in range(n - 1):
    d = pc(np.bitwise_xor(H[i, 0], H[i + 1:, :])).min(axis=1).astype(np.int16)  # min over dihedral of j
    js = np.arange(i + 1, n)
    keep = (d <= 16) | np.array([cid[i] == cid[j] or pid[i] == pid[j] for j in js])
    I += [i] * int(keep.sum()); J += js[keep].tolist(); D += d[keep].tolist()
I, J, D = np.array(I), np.array(J), np.array(D)
same_c = np.array([cid[a] == cid[b] for a, b in zip(I, J)])
same_p = np.array([pid[a] == pid[b] for a, b in zip(I, J)])
tile = same_p & ~same_c
cross = ~same_p
np.savez_compressed(OUT / "pairs.npz", I=I, J=J, D=D, same_c=same_c, same_p=same_p)
pop = {
    "n_images": n,
    "same_content_pairs": int(same_c.sum()),
    "same_content_confirmed_d0_d4_d8": [int((same_c & (D <= t)).sum()) for t in (0, 4, 8)],
    "same_content_unconfirmed_d_gt8": int((same_c & (D > 8)).sum()),
    "content_groups_with_unconfirmed_pair": len({cid[a] for a, b, d, s in zip(I, J, D, same_c) if s and d > 8}),
    "content_groups_multi": sum(1 for c, k in __import__("collections").Counter(cid).items() if k > 1),
    "tile_sibling_pairs (same parent, different content)": int(tile.sum()),
    "parents_with_tiles": len({pid[a] for a, t in zip(I, tile) if t}),
    "cross_parent_d0": int((cross & (D == 0)).sum()),
    "cross_parent_d_le4": int((cross & (D <= 4)).sum()),
    "cross_parent_d5_8": int((cross & (D > 4) & (D <= 8)).sum()),
    "cross_parent_d_le8": int((cross & (D <= 8)).sum()),
    "cross_parent_d9_12": int((cross & (D > 8) & (D <= 12)).sum()),
    "cross_parent_d13_16": int((cross & (D > 12) & (D <= 16)).sum()),
    "paper_reference": {"same_content_pairs": 44747, "confirmed_d0_d4_d8": [13549, 29554, 30928],
                         "cross_parent_d0": 403, "cross_parent_d_le4": 2571, "cross_parent_d_le8": 6878},
}
(OUT / "population.json").write_text(json.dumps(pop, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(pop, ensure_ascii=False, indent=1))
