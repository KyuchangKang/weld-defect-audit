# -*- coding: utf-8 -*-
"""Step 1: parent-photo grouping statistics for final0 (dry run before building clean set)."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from weldpaths import project_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"C:\Sync_cloud\Dropbox\군산대-랩-공동작업장\Papers\paper_wedling2") / "04_experiments" / "data" / "final0"
SPLITS = ["train", "valid", "test"]


def source_stem(name: str) -> str:
    s = re.sub(r"\.rf\.[0-9a-f]{32}.*$", "", name)
    s = re.sub(r"_(jpeg|jpg|png|bmp)(_(jpeg|jpg|png|bmp))*$", "", s, flags=re.I)
    return s.lower()


def parent_id(stem: str) -> str:
    p = stem
    # strip roboflow-style pre-upload aug suffixes, applied repeatedly from the end
    patterns = [
        r"[-_]+flip$",
        r"[-_]+(90|180|270)[-_]*$",
        r"[-_]+rot(90|180|270)$",
        r"_block_\d+_\d+$",          # tile crops -> group with parent photo
        r"[-_]+copy(-\d+)?$",
        r"[-_]+\(\d+\)$",
        r"[-_]+aug\d*$",
    ]
    changed = True
    while changed:
        changed = False
        for pat in patterns:
            new = re.sub(pat, "", p)
            if new != p and len(new) >= 3:
                p = new
                changed = True
    return p


files = []
for split in SPLITS:
    for img in (BASE / split / "images").glob("*"):
        if img.is_file():
            files.append((split, img.name))

stems = {name: source_stem(name) for _, name in files}
parents = {name: parent_id(stems[name]) for _, name in files}

par_map = defaultdict(lambda: defaultdict(int))  # parent -> split -> count
for split, name in files:
    par_map[parents[name]][split] += 1

n_parents = len(par_map)
leak_parents = {k: dict(v) for k, v in par_map.items() if len(v) > 1}
copies_per_parent = Counter(sum(v.values()) for v in par_map.values())

out = {
    "total_files": len(files),
    "unique_stems": len(set(stems.values())),
    "unique_parents": n_parents,
    "parents_spanning_splits(leakage)": len(leak_parents),
    "copies_per_parent_hist_top": dict(sorted(copies_per_parent.items())[:12]),
    "biggest_parents": sorted(((sum(v.values()), k) for k, v in par_map.items()), reverse=True)[:8],
    "leak_examples": dict(list(leak_parents.items())[:5]),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
