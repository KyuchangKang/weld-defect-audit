# -*- coding: utf-8 -*-
"""T3 finding (2026-10-28): the content-ID rule strips rotation suffixes (_0/_90/_180/_270) iteratively and,
in the original code (04_experiments/scripts/build_clean2.py, and phash_validate.py), also eats the trailing
column/row index of the tile suffix _block_<row>_<col> when it is 0, before the parent rule can strip the
tile suffix. Tiles of one photograph then fall into different parents.

This script recomputes, WITHOUT changing any frozen file:
  (1) raw distribution: content groups, parents, leaking parents  -- original rule vs rule as specified
  (2) final0_clean: corrected parents spanning splits, and the valid/test images affected
Rule as specified (paper Section 3.1): content ID = stem minus augmentation suffixes; parent ID = content ID
minus _block_<row>_<col>. All 923 tile stems are exactly img_<n>_block_<r>_<c> (no augmentation suffix),
so the corrected rule protects that suffix from the rotation stripper.
Output: results/T3/tile_fix.json
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir, runs_dir  # noqa: E402,F401  (path defaults; see weldpaths.py)
_T3 = Path(__file__).resolve().parents[2] / "results" / "revision2" / "T3"  # was work/results/T3

BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
SRC = BASE / "04_experiments" / "data" / "final0"
MAN = BASE / "04_experiments" / "data" / "final0_clean" / "manifest.csv"
OUT = _T3 / "tile_fix.json"
ns = {}
_src = open(Path(__file__).resolve().parents[1] / "build_clean2.py", encoding="utf-8").read()
exec("import re" + chr(10) + _src[_src.index("def source_stem"):_src.index("def label_classes")], ns)  # rule functions only
source_stem, content_id_orig, parent_id_orig = ns["source_stem"], ns["content_id"], ns["parent_id"]
TILE = re.compile(r"^(.*)_block_\d+_\d+$")


def content_id_fixed(stem):
    return stem if TILE.match(stem) else content_id_orig(stem)


def parent_id_fixed(cid):
    m = TILE.match(cid)
    return m.group(1) if m else parent_id_orig(cid)


files = [(sp, f.name) for sp in ("train", "valid", "test") for f in sorted((SRC / sp / "images").iterdir())
         if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def audit(cfun, pfun):
    c2s, p2s = defaultdict(set), defaultdict(set)
    for sp, n in files:
        c = cfun(source_stem(n))
        p = pfun(c)
        c2s[c].add(sp)
        p2s[p].add(sp)
    leak = sum(len(v) > 1 for v in p2s.values())
    return {"files": len(files), "content_groups": len(c2s), "parents": len(p2s),
            "leaking_parents": leak, "leaking_share": round(leak / len(p2s), 4)}


raw = {"original_rule": audit(content_id_orig, lambda c: parent_id_orig(c)),
       "rule_as_specified": audit(content_id_fixed, parent_id_fixed)}

rows = list(csv.DictReader(open(MAN, encoding="utf-8-sig")))
par = defaultdict(set)
for r in rows:
    par[parent_id_fixed(content_id_fixed(r["stem"]))].add(r["split"])
span = {p: sorted(s) for p, s in par.items() if len(s) > 1}
affected = defaultdict(list)
for r in rows:
    p = parent_id_fixed(content_id_fixed(r["stem"]))
    if p in span and r["split"] in ("valid", "test"):
        affected[r["split"]].append(r["new_id"])
clean = {"manifest_files": len(rows), "parents_rule_as_specified": len(par),
         "parents_spanning_splits": len(span), "spanning": span,
         "valid_images_affected": sorted(affected["valid"]), "test_images_affected": sorted(affected["test"]),
         "n_valid_affected": len(affected["valid"]), "n_test_affected": len(affected["test"]),
         "manifest_parent_ids_with_block": sum("block" in r["parent_id"] for r in rows)}
out = {"raw_distribution": raw, "final0_clean": clean}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"raw": raw, **{k: v for k, v in clean.items() if k != "spanning"}}, ensure_ascii=False, indent=1))
