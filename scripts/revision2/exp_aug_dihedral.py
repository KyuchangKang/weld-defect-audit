# -*- coding: utf-8 -*-
"""IR2605 2nd revision, R1-5: retained pre-generated copies vs online 90-degree rotation + vertical flip.

Design (fixed before running): ../results/T6_학습설계.md
  A  3,567 files x 100 ep, default aug        -> existing (seeds 42-44, valid 0.661 +/- 0.019)
  B  495 unique x 720 ep, default aug         -> seed 42 existing (0.570); seeds 43, 44 here
  C  495 unique x 720 ep, + rot90 (k uniform) + flipud 0.5 -> seeds 42, 43, 44 here
All other settings as exp_diversity.py (M8): yolov5nu, batch 16, imgsz 640, lr0 0.01, workers 4,
cache False, patience = epochs; best.pt re-validated on the 142-image validation split (workers 0).
The sealed test split is never read.

Usage (venv D:\\venvs\\weld2):
  python exp_aug_dihedral.py build              # dedup dataset (identical rule to exp_diversity.py)
  python exp_aug_dihedral.py verify             # checks 1-3 of the design (no training)
  python exp_aug_dihedral.py smoke              # check 4: 2-epoch run of condition C
  python exp_aug_dihedral.py run C 42           # one full run; skipped if its result exists
  python exp_aug_dihedral.py collect            # aggregate -> results/aug_dihedral.json
"""
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import rot90_patch  # noqa: F401  (must be imported at top level: re-applied in spawn workers)

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
CLEAN = BASE / "04_experiments" / "data" / "final0_clean"
M8_DATA = Path(r"D:\runs\weld2\diversity\data")          # dataset used by the existing B seed 42
LOCAL = Path(r"D:\runs\weld2\aug_r2")
WORK = BASE / "08_revision" / "2nd_revision" / "work"
RES = WORK / "results"
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
EPOCHS = 720
TRAIN_KW = dict(batch=16, imgsz=640, lr0=0.01, device=0, workers=4, cache=False,
                exist_ok=True, plots=False, verbose=False, patience=EPOCHS, epochs=EPOCHS)
COND = {"B": dict(flipud=0.0, rot90=False), "C": dict(flipud=0.5, rot90=True)}
REF = {"A_full_3567_100ep_seeds42_44": [0.6808, 0.6439, 0.6571], "B_seed42_M8": 0.570}


def build():
    dst = LOCAL / "data"
    yaml = dst / "data.yaml"
    rows = list(csv.DictReader(open(CLEAN / "manifest.csv", encoding="utf-8-sig")))
    for s in ("train", "valid"):
        for sub in ("images", "labels"):
            (dst / s / sub).mkdir(parents=True, exist_ok=True)
    by_content = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by_content[r["content_id"]].append(r)
    for cid, rs in by_content.items():
        r = sorted(rs, key=lambda x: (len(x["orig_filename"]), x["orig_filename"]))[0]
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            shutil.copyfile(CLEAN / "train" / sub / (r["new_id"] + ext), dst / "train" / sub / (r["new_id"] + ext))
    for sub in ("images", "labels"):
        for p in (CLEAN / "valid" / sub).iterdir():
            shutil.copyfile(p, dst / "valid" / sub / p.name)
    yaml.write_text(f"train: {dst / 'train' / 'images'}\nval: {dst / 'valid' / 'images'}\n"
                    f"test: {dst / 'valid' / 'images'}\n\nnc: 5\nnames: {NAMES}\n", encoding="utf-8")
    # identity check against the dataset of the existing B run (seed 42)
    import hashlib

    def md5s(root):
        out = {}
        for s in ("train", "valid"):
            for sub in ("images", "labels"):
                for p in sorted((root / s / sub).iterdir()):
                    out[f"{s}/{sub}/{p.name}"] = hashlib.md5(p.read_bytes()).hexdigest()
        return out
    mine, m8 = md5s(dst), md5s(M8_DATA)
    same = mine == m8
    info = {"train_files": len(list((dst / "train" / "images").iterdir())),
            "valid_files": len(list((dst / "valid" / "images").iterdir())),
            "identical_to_M8_dataset": same,
            "n_entries": len(mine), "n_entries_M8": len(m8)}
    print(json.dumps(info, indent=1))
    (RES / "aug_dataset_check.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return yaml


def _train_dataset(flipud):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(str(LOCAL / "data" / "data.yaml"))
    cfg = get_cfg(overrides=dict(imgsz=640, flipud=flipud, cache=False))
    return build_yolo_dataset(cfg, data["train"], 16, data, mode="train")


def verify():
    import numpy as np
    from PIL import Image, ImageDraw
    out = RES / "T6_verify"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    # --- 1. unit test ---------------------------------------------------------------------
    # (a) boxes vs an independent implementation (rotate the 4 corners, take min/max)
    # (b) pixels: crop under the rotated box == rot90(crop under the original box).
    #     Outward rounding floor(v*H + eps) / ceil(v*H - eps) is exactly consistent under the
    #     rotation (floor(H - a) = H - ceil(a)); eps = 1e-3 px absorbs float32 noise. A first
    #     version used round() and failed only because Python rounds half to even (test artifact).
    import math
    os.environ["WELD_ROT90"] = "0"
    ds = _train_dataset(0.0)
    rng = random.Random(0)
    idx = sorted(rng.sample(range(len(ds)), 40))
    eps = 1e-3

    def corners(o, k):
        x1, y1, x2, y2 = o
        pts = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float64)
        for _ in range(k):
            pts = np.stack([pts[:, 1], 1 - pts[:, 0]], 1)
        return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    worst_px = 0.0
    worst_box = 0.0
    n_boxes = 0
    shape_mis = 0
    panels = []
    for j, i in enumerate(idx):
        base = ds.get_image_and_label(i)
        img0 = base["img"]
        H = img0.shape[0]
        base["instances"].convert_bbox("xyxy")
        b0 = base["instances"].bboxes.astype(np.float64).copy()
        fl = lambda v: int(math.floor(v * H + eps))
        ce = lambda v: int(math.ceil(v * H - eps))
        for k in range(4):
            lab = rot90_patch.rot90_label(ds.get_image_and_label(i), k)
            lab["instances"].convert_bbox("xyxy")
            bb = lab["instances"].bboxes.astype(np.float64)
            for b, o in zip(bb, b0):
                n_boxes += 1
                worst_box = max(worst_box, float(np.abs(b - corners(o, k)).max()))
                cr = lab["img"][max(fl(b[1]), 0):min(ce(b[3]), H), max(fl(b[0]), 0):min(ce(b[2]), H)]
                rf = np.rot90(img0[max(fl(o[1]), 0):min(ce(o[3]), H), max(fl(o[0]), 0):min(ce(o[2]), H)], k)
                if cr.shape != rf.shape:
                    shape_mis += 1
                    continue
                worst_px = max(worst_px, float(np.abs(cr.astype(np.float32) - rf.astype(np.float32)).mean()))
            if j < 4:
                im = Image.fromarray(lab["img"][..., ::-1].copy())
                d = ImageDraw.Draw(im)
                for (x1, y1, x2, y2), c in zip(bb, lab["cls"].ravel()):
                    d.rectangle([x1 * H, y1 * H, x2 * H, y2 * H],
                                outline=[(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 160, 0), (200, 0, 200)][int(c)], width=4)
                d.text((8, 8), f"k={k}", fill=(255, 255, 0))
                panels.append(im.resize((320, 320)))
    sheet = Image.new("RGB", (320 * 4, 320 * 4))
    for n, p in enumerate(panels):
        sheet.paste(p, ((n % 4) * 320, (n // 4) * 320))
    sheet.save(out / "1_unit_rot90_k0-3.jpg", quality=90)
    report["1_unit"] = {"images": len(idx), "boxes_x_4_rotations": n_boxes,
                        "max_abs_err_vs_independent_corner_rotation": worst_box,
                        "pixel_shape_mismatches": shape_mis,
                        "max_mean_abs_diff_crop_vs_rotated_reference": round(worst_px, 6),
                        "pass": worst_box < 1e-6 and shape_mis == 0 and worst_px == 0.0}
    # --- 2. pipeline: real training transforms (mosaic etc.), condition C -------------------
    os.environ["WELD_ROT90"] = "1"
    logdir = out / "klog"
    shutil.rmtree(logdir, ignore_errors=True)
    logdir.mkdir()
    os.environ["WELD_ROT90_LOG"] = str(logdir)
    dsC = _train_dataset(0.5)
    random.seed(1)
    sheet = Image.new("RGB", (320 * 4, 320 * 4))
    for n in range(16):
        s = dsC[n * 7]
        img = s["img"].numpy().transpose(1, 2, 0).copy()  # Format() already returns RGB
        H, W = img.shape[:2]
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        for (cx, cy, w, h), c in zip(s["bboxes"].numpy(), s["cls"].numpy().ravel()):
            d.rectangle([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H],
                        outline=[(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 160, 0), (200, 0, 200)][int(c)], width=3)
        sheet.paste(im.resize((320, 320)), ((n % 4) * 320, (n // 4) * 320))
    sheet.save(out / "2_pipeline_mosaic_C.jpg", quality=90)
    report["2_pipeline"] = {"samples_rendered": 16, "file": "2_pipeline_mosaic_C.jpg", "pass": "visual"}
    # --- 3. k distribution inside DataLoader worker processes -------------------------------
    for f in logdir.iterdir():
        f.unlink()
    from ultralytics.data.build import build_dataloader
    dl = build_dataloader(dsC, 16, 4, shuffle=True, rank=-1)
    n_batches = 0
    for _ in dl:
        n_batches += 1
        if n_batches >= 32:
            break
    del dl
    time.sleep(2)
    ks = Counter()
    pids = 0
    for f in logdir.iterdir():
        pids += 1
        ks.update(f.read_text())
    tot = sum(ks.values())
    report["3_k_distribution"] = {"draws": tot, "worker_processes_logging": pids,
                                  "counts": {k: ks.get(k, 0) for k in "0123"},
                                  "shares": {k: round(ks.get(k, 0) / max(tot, 1), 3) for k in "0123"},
                                  "pass": pids >= 2 and tot >= 2000 and all(0.2 < ks.get(k, 0) / tot < 0.3 for k in "0123")}
    os.environ.pop("WELD_ROT90_LOG", None)
    (RES / "T6_verify.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


def _train(cond, seed, epochs, name):
    from ultralytics import YOLO
    os.environ["WELD_ROT90"] = "1" if COND[cond]["rot90"] else "0"
    kw = {**TRAIN_KW, "epochs": epochs, "patience": epochs, "flipud": COND[cond]["flipud"]}
    t0 = time.time()
    YOLO("yolov5nu.pt").train(data=str(LOCAL / "data" / "data.yaml"), project=str(LOCAL), name=name,
                              seed=seed, **kw)
    return time.time() - t0


def smoke():
    sec = _train("C", 42, 2, "smoke_C")
    print(json.dumps({"smoke_C_2ep_seconds": round(sec, 1)}))
    (RES / "T6_smoke.json").write_text(json.dumps({"smoke_C_2ep_seconds": round(sec, 1),
                                                   "ok": (LOCAL / "smoke_C" / "weights" / "best.pt").exists()}),
                                       encoding="utf-8")


def run(cond, seed):
    from ultralytics import YOLO
    name = f"{cond}_s{seed}"
    res_f = RES / "aug_runs" / f"{name}.json"
    if res_f.exists():
        print("skip (done)", name)
        return
    best = LOCAL / name / "weights" / "best.pt"
    sec = None
    if not best.exists():
        sec = _train(cond, seed, EPOCHS, name)
    os.environ["WELD_ROT90"] = "0"
    m = YOLO(str(best))
    r = m.val(data=str(LOCAL / "data" / "data.yaml"), split="val", batch=16, imgsz=640, device=0,
              workers=0, verbose=False, plots=False)
    idx = r.box.ap_class_index.tolist()
    out = {"condition": cond, "seed": seed, "epochs": EPOCHS, "train_files": 495,
           "image_passes": 495 * EPOCHS, "online_aug_extra": COND[cond],
           "valid_mAP50": round(float(r.box.map50), 4), "valid_mAP50_95": round(float(r.box.map), 4),
           "per_class_ap50": {m.names[c]: round(float(r.box.ap50[i]), 4) for i, c in enumerate(idx)},
           "train_seconds": None if sec is None else round(sec, 1)}
    res_f.parent.mkdir(parents=True, exist_ok=True)
    res_f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def collect():
    import statistics as st
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RES / "aug_runs").glob("*.json"))]
    B = [REF["B_seed42_M8"]] + [r["valid_mAP50"] for r in runs if r["condition"] == "B"]
    C = [r["valid_mAP50"] for r in runs if r["condition"] == "C"]
    A = REF["A_full_3567_100ep_seeds42_44"]
    s = lambda v: {"values": v, "mean": round(st.mean(v), 4), "sd": round(st.stdev(v), 4) if len(v) > 1 else None}
    out = {"A": s(A), "B": s(B), "C": s(C), "runs": runs}
    a, b, c = st.mean(A), st.mean(B), st.mean(C)
    sdA = st.stdev(A)
    out["rule"] = ("reproduced" if c >= a - sdA else "partial" if c > b + sdA else "not_explained")
    out["share_of_gap_explained"] = round((c - b) / (a - b), 3) if a != b else None
    (RES / "aug_dihedral.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("A", "B", "C", "rule", "share_of_gap_explained")}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "verify":
        verify()
    elif cmd == "smoke":
        smoke()
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]))
    elif cmd == "collect":
        collect()
