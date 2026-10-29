# -*- coding: utf-8 -*-
"""V6: 인접샷 병합 재분할 (v0.5 리뷰 B-4b — 사용자 승인 2026-08-14 마이너 수정 범위).

지각 해시(d<=4) 근접중복으로 연결된 부모들을 하나의 부모로 병합한 뒤 층화 재분할을
1회 수행하고 v5nu(시드 42)를 학습, 그 스플릿의 test를 1회 평가 → c_merged.
"잔존 인접샷 제외(-0.007)" 민감도를 "병합 재분할" 방식으로 완결하는 실험.
출력: 05_results/upgrade/merged_resplit.json
"""
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from weldpaths import project_dir, runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
SRC = BASE / "04_experiments" / "data" / "final0"
HASH_NPY = runs_dir(r"D:\runs\weld2") / "upgrade" / "phash_dihedral.npy"
NAMES_JSON = runs_dir(r"D:\runs\weld2") / "upgrade" / "phash_files.json"
LOCAL = runs_dir(r"D:\runs\weld2") / "upgrade2"
OUT = BASE / "05_results" / "upgrade" / "merged_resplit.json"
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
SPLITS = ["train", "valid", "test"]
RATIOS = {"train": 0.70, "valid": 0.20, "test": 0.10}
THR = 4


def ep(p):
    return "\\\\?\\" + str(p)


def source_stem(name):
    s = re.sub(r"\.rf\.[0-9a-f]{32}.*$", "", name)
    s = re.sub(r"_(jpeg|jpg|png|bmp)(_(jpeg|jpg|png|bmp))*$", "", s, flags=re.I)
    return s.lower()


def content_id(stem):
    p = stem
    pats = [r"[-_]+flip$", r"[-_]+(0|90|180|270)[-_]*$", r"[-_]+rot(0|90|180|270)$",
            r"[-_]+copy(-\d+)?$", r"[-_]+\(\d+\)$", r"[-_]+aug\d*$"]
    changed = True
    while changed:
        changed = False
        for pat in pats:
            new = re.sub(pat, "", p)
            if new != p and len(new) >= 3:
                p, changed = new, True
    return p


def parent_id(cid):
    return re.sub(r"_block_\d+_\d+$", "", cid)


def label_classes(lbl):
    c = Counter()
    if lbl.exists():
        for ln in open(ep(lbl), encoding="utf-8").read().strip().splitlines():
            parts = ln.split()
            if len(parts) == 5:
                c[int(parts[0])] += 1
    return c


def main():
    files, recs = [], []
    for split in SPLITS:
        for img in sorted((SRC / split / "images").iterdir()):
            st = source_stem(img.name)
            cid = content_id(st)
            files.append(img.name)
            recs.append({"img": img, "lbl": SRC / split / "labels" / (img.stem + ".txt"),
                         "cid": cid, "pid": parent_id(cid)})
    saved = json.loads(NAMES_JSON.read_text(encoding="utf-8"))
    assert saved == files, "hash cache mismatch"
    H = np.load(HASH_NPY)
    n = len(files)

    # 근접쌍 → 부모 병합 (union-find over parent ids)
    pids = sorted(set(r["pid"] for r in recs))
    pindex = {p: i for i, p in enumerate(pids)}
    up = list(range(len(pids)))

    def find(x):
        while up[x] != x:
            up[x] = up[up[x]]
            x = up[x]
        return x

    h0 = H[:, 0].copy()
    merged_edges = 0
    CH = 512
    for s in range(0, n, CH):
        e = min(s + CH, n)
        dmin = np.full((e - s, n), 64, dtype=np.uint8)
        for t in range(8):
            x = np.bitwise_xor.outer(h0[s:e], H[:, t])
            np.minimum(dmin, np.bitwise_count(x).astype(np.uint8), out=dmin)
        ii, jj = np.nonzero(dmin <= THR)
        for a, b in zip(ii, jj):
            gi, gj = s + int(a), int(b)
            if gi < gj:
                pa, pb = find(pindex[recs[gi]["pid"]]), find(pindex[recs[gj]["pid"]])
                if pa != pb:
                    up[pa] = pb
                    merged_edges += 1
    mpid_of = {r["pid"]: find(pindex[r["pid"]]) for r in recs}
    n_merged = len(set(mpid_of.values()))
    print(f"parents {len(pids)} -> merged parents {n_merged} (edges {merged_edges})")

    # 병합 부모 기준 층화 재분할 (build_clean2 로직, seed 42 결정적)
    by_content = defaultdict(list)
    for r in recs:
        by_content[r["cid"]].append(r)
    rep_of = {cid: sorted(rs, key=lambda x: (len(x["img"].name), x["img"].name))[0]
              for cid, rs in by_content.items()}
    par_contents = defaultdict(list)
    for cid, r in rep_of.items():
        par_contents[mpid_of[r["pid"]]].append(cid)
    par_vec, par_ncont = {}, {}
    for pid, cids in par_contents.items():
        v = Counter()
        for cid in cids:
            v += label_classes(rep_of[cid]["lbl"])
        par_vec[pid], par_ncont[pid] = v, len(cids)
    total_vec = Counter()
    for v in par_vec.values():
        total_vec += v
    total_cont = sum(par_ncont.values())
    targets = {s: {c: total_vec[c] * RATIOS[s] for c in range(5)} for s in SPLITS}
    targets_cont = {s: total_cont * RATIOS[s] for s in SPLITS}
    rarity = {c: 1.0 / max(1, total_vec[c]) for c in range(5)}
    order = sorted(par_vec, key=lambda p: -sum(par_vec[p][c] * rarity[c] for c in range(5)))
    cur = {s: Counter() for s in SPLITS}
    cur_cont = {s: 0 for s in SPLITS}
    assign = {}
    for pid in order:
        v, ncont = par_vec[pid], par_ncont[pid]
        best, best_score = None, None
        for s in SPLITS:
            gain = sum(rarity[c] * v[c] * max(0.0, targets[s][c] - cur[s][c])
                       / max(1.0, targets[s][c]) for c in range(5))
            pressure = (targets_cont[s] - cur_cont[s]) / targets_cont[s]
            score = gain + 0.05 * pressure
            if best_score is None or score > best_score:
                best, best_score = s, score
        assign[pid] = best
        cur[best] += v
        cur_cont[best] += ncont

    dst = LOCAL / "merged_split"
    if dst.exists():
        shutil.rmtree(ep(dst))
    for s in SPLITS:
        (dst / s / "images").mkdir(parents=True)
        (dst / s / "labels").mkdir(parents=True)
    counters = {s: 0 for s in SPLITS}
    inst = {s: Counter() for s in SPLITS}
    for pid, cids in par_contents.items():
        s = assign[pid]
        chosen = ([r for r in recs if mpid_of[r["pid"]] == pid] if s == "train"
                  else [rep_of[cid] for cid in cids])
        for r in chosen:
            counters[s] += 1
            nid = f"{s}_{counters[s]:05d}"
            shutil.copyfile(ep(r["img"]), ep(dst / s / "images" / f"{nid}.jpg"))
            text = (open(ep(r["lbl"]), encoding="utf-8").read()
                    if r["lbl"].exists() else "")
            with open(ep(dst / s / "labels" / f"{nid}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            for c, k in label_classes(r["lbl"]).items():
                inst[s][NAMES[c]] += k
    (dst / "data.yaml").write_text(
        f"train: {dst / 'train' / 'images'}\nval: {dst / 'valid' / 'images'}\n"
        f"test: {dst / 'test' / 'images'}\n\nnc: 5\nnames: {NAMES}\n", encoding="utf-8")
    ok = all(inst[s][nm] > 0 for s in ("valid", "test") for nm in NAMES)
    print(f"built merged split ok={ok} files={counters} test_inst={dict(inst['test'])}")

    from ultralytics import YOLO
    m = YOLO("yolov5nu.pt")
    m.train(data=str(dst / "data.yaml"), project=str(LOCAL), name="V6_merged_s42",
            seed=42, epochs=100, batch=16, imgsz=640, lr0=0.01, device=0,
            workers=4, cache=False, exist_ok=True, plots=False, verbose=False)
    w = LOCAL / "V6_merged_s42" / "weights" / "best.pt"
    mm = YOLO(str(w))
    r = mm.val(data=str(dst / "data.yaml"), split="test", batch=16, imgsz=640,
               device=0, workers=4, verbose=False, plots=False)
    ap50 = {mm.names[ci]: round(float(r.box.ap50[i]), 4)
            for i, ci in enumerate(r.box.ap_class_index.tolist())}
    out = {
        "threshold": THR, "parents_before": len(pids), "parents_after_merge": n_merged,
        "merge_edges": merged_edges, "files": counters,
        "test_instances": dict(inst["test"]),
        "c_merged_mAP50": round(float(r.box.map50), 4),
        "mAP50_95": round(float(r.box.map), 4), "per_class_ap50": ap50,
        "a_raw_ref": 0.8313,
        "inflation_a_minus_c_merged": round(0.8313 - float(r.box.map50), 4),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
