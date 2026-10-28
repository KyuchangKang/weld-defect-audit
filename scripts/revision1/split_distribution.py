# -*- coding: utf-8 -*-
"""R1-8: final0_clean 분할별 클래스 인스턴스 수 / 해당 클래스를 가진 부모 수 (학습 0회, 집계만)."""
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
CLEAN = BASE / "04_experiments" / "data" / "final0_clean"
OUT = BASE / "08_revision" / "1st_revision" / "work" / "results" / "split_distribution.json"
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
rows = list(csv.DictReader(open(CLEAN / "manifest.csv", encoding="utf-8-sig")))
inst = defaultdict(Counter); par = defaultdict(lambda: defaultdict(set)); files = Counter(); parents = defaultdict(set)
for r in rows:
    s = r["split"]; files[s] += 1; parents[s].add(r["parent_id"])
    lbl = CLEAN / s / "labels" / (r["new_id"] + ".txt")
    if lbl.exists():
        for ln in lbl.read_text(encoding="utf-8").split("\n"):
            p = ln.split()
            if len(p) == 5:
                c = NAMES[int(p[0])]; inst[s][c] += 1; par[s][c].add(r["parent_id"])
res = {s: {"files": files[s], "parents": len(parents[s]),
           "instances": {c: inst[s][c] for c in NAMES},
           "parents_with_class": {c: len(par[s][c]) for c in NAMES}} for s in ["train", "valid", "test"]}
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for s, v in res.items():
    print(s, v["files"], v["parents"]); print("  inst", v["instances"]); print("  par ", v["parents_with_class"])
