# -*- coding: utf-8 -*-
"""Dataset statistics + duplicate analysis for final0 / ikkj."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from weldpaths import project_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"C:\Sync_cloud\Dropbox\군산대-랩-공동작업장\Papers\paper_wedling2") / "04_experiments" / "data"
DATASETS = {
    "final0": {
        "root": BASE / "final0",
        "names": ['Geometric defect', 'Non-fusion defect', 'crack', 'porosity', 'spatters'],
    },
    "ikkj": {
        "root": BASE / "ikkj",
        "names": ['Bad Welding', 'Crack', 'Good Welding', 'Porosity', 'Reinforcement', 'Spatters'],
    },
}
SPLITS = ["train", "valid", "test"]


def ext_path(p: Path) -> str:
    return "\\\\?\\" + str(p)


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(ext_path(p), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


report = {}
hash_maps = {}  # ds -> {md5: [relpaths]}

for ds, cfg in DATASETS.items():
    root, names = cfg["root"], cfg["names"]
    ds_rep = {"splits": {}, "class_instances": {}, "images_with_class": {},
              "empty_labels": 0, "label_issues": []}
    hashes = defaultdict(list)
    inst_counter = Counter()
    img_class_counter = Counter()

    for split in SPLITS:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        imgs = sorted(img_dir.glob("*")) if img_dir.exists() else []
        lbls = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}
        n_img, n_lbl_matched = 0, 0
        for img in imgs:
            if img.is_dir():
                continue
            n_img += 1
            hashes[md5_of(img)].append(f"{split}/{img.name}")
            lbl = lbls.get(img.stem)
            if lbl is None:
                ds_rep["label_issues"].append(f"no label: {split}/{img.name}")
                continue
            n_lbl_matched += 1
            classes_here = set()
            try:
                text = open(ext_path(lbl), encoding="utf-8").read().strip()
            except OSError as e:
                ds_rep["label_issues"].append(f"unreadable label: {split}/{lbl.name}: {e}")
                continue
            if not text:
                ds_rep["empty_labels"] += 1
                continue
            for ln in text.splitlines():
                parts = ln.split()
                if len(parts) != 5:
                    ds_rep["label_issues"].append(f"bad line ({len(parts)} fields): {split}/{lbl.name}")
                    continue
                cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if cid >= len(names) or any(not (0 <= c <= 1.0001) for c in coords):
                    ds_rep["label_issues"].append(f"out-of-range: {split}/{lbl.name}: {ln[:60]}")
                    continue
                cname = names[cid]
                inst_counter[cname] += 1
                classes_here.add(cname)
            for c in classes_here:
                img_class_counter[c] += 1
        ds_rep["splits"][split] = {"images": n_img, "labels_matched": n_lbl_matched}

    ds_rep["class_instances"] = dict(inst_counter)
    ds_rep["images_with_class"] = dict(img_class_counter)
    # cap issue list
    ds_rep["n_label_issues"] = len(ds_rep["label_issues"])
    ds_rep["label_issues"] = ds_rep["label_issues"][:10]
    # internal duplicates
    internal_dups = {h: v for h, v in hashes.items() if len(v) > 1}
    ds_rep["internal_dup_groups"] = len(internal_dups)
    ds_rep["internal_dup_extra_images"] = sum(len(v) - 1 for v in internal_dups.values())
    # cross-split leakage (same image in different splits)
    leak = 0
    for v in internal_dups.values():
        splits_seen = {x.split("/")[0] for x in v}
        if len(splits_seen) > 1:
            leak += 1
    ds_rep["dup_groups_spanning_splits"] = leak
    report[ds] = ds_rep
    hash_maps[ds] = hashes

# cross-dataset duplicates
common = set(hash_maps["final0"]) & set(hash_maps["ikkj"])
report["cross_dataset"] = {
    "common_md5": len(common),
    "final0_images_in_common": sum(len(hash_maps["final0"][h]) for h in common),
    "ikkj_images_in_common": sum(len(hash_maps["ikkj"][h]) for h in common),
    "examples": [hash_maps["final0"][h][0] + "  <->  " + hash_maps["ikkj"][h][0]
                 for h in list(common)[:5]],
}

out = BASE / "analysis_report.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
