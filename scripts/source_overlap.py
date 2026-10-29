# -*- coding: utf-8 -*-
"""Source-level overlap via Roboflow filename stems (strip .rf.<hash> and aug suffixes)."""
import json
import re
from collections import defaultdict
from pathlib import Path

from weldpaths import project_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"C:\Sync_cloud\Dropbox\군산대-랩-공동작업장\Papers\paper_wedling2") / "04_experiments" / "data"
SPLITS = ["train", "valid", "test"]


def source_stem(name: str) -> str:
    # Roboflow names: <original>_jpg.rf.<32hex>.jpg  /  <original>_jpeg_jpg.rf.<hex>.jpg
    s = re.sub(r"\.rf\.[0-9a-f]{32}.*$", "", name)
    s = re.sub(r"_(jpeg|jpg|png|bmp)(_(jpeg|jpg|png|bmp))*$", "", s)
    return s.lower()


maps = {}
for ds in ["final0", "ikkj"]:
    m = defaultdict(lambda: defaultdict(int))  # stem -> split -> count
    for split in SPLITS:
        d = BASE / ds / split / "images"
        for p in d.glob("*"):
            if p.is_file():
                m[source_stem(p.name)][split] += 1
    maps[ds] = m

out = {}
for ds, m in maps.items():
    n_sources = len(m)
    multi = {k: dict(v) for k, v in m.items() if sum(v.values()) > 1}
    leak = {k: dict(v) for k, v in m.items() if len(v) > 1}  # same source in >1 split
    out[ds] = {
        "unique_sources": n_sources,
        "sources_with_multiple_copies": len(multi),
        "SOURCES_SPANNING_SPLITS(LEAKAGE)": len(leak),
        "leak_examples": dict(list(leak.items())[:5]),
    }

common = set(maps["final0"]) & set(maps["ikkj"])
out["cross_dataset_common_sources"] = len(common)
out["common_examples"] = sorted(common)[:10]
# where do common sources sit in each dataset (split distribution)?
split_dist = defaultdict(int)
for s in common:
    for split, c in maps["final0"][s].items():
        split_dist[f"final0_{split}"] += c
    for split, c in maps["ikkj"][s].items():
        split_dist[f"ikkj_{split}"] += c
out["common_sources_image_counts"] = dict(split_dist)

with open(BASE / "source_overlap_report.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
