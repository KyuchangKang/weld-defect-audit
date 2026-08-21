# -*- coding: utf-8 -*-
"""리뷰 대응 로버스트니스 실험 (evaluation_plan v1.1 — 사용자 승인 2026-08-13).

R2: 조건 (d) — 정제학습 모델(E2_v5n)을 원본 누수 test로 평가 → 2x2 완성
R1: 재분할 로버스트니스 — 분할 시드 4개로 부모 단위 층화 재분할 반복,
    v5nu(학습 시드 42 고정) 학습 → 각 스플릿의 test 1회 평가 → c 분포
R3: 제2 아키텍처 — YOLO11n raw/clean 각 1런 → 부풀림 아키텍처 비의존성

순차 실행(동시 1런 정책). 결과는 단계마다 05_results/upgrade/upgrade_results.json에 기록.
분할 로직은 build_clean2.py와 동일하되 시드 주입(순서 셔플 + 점수 지터)만 추가.
"""
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2")
DATA = BASE / "04_experiments" / "data"
SRC = DATA / "final0"
CLEAN_YAML = DATA / "final0_clean" / "data.yaml"
RAW_YAML = SRC / "data.yaml"
LOCAL = Path(r"D:\runs\weld2\upgrade")
OUT_DIR = BASE / "05_results" / "upgrade"
OUT = OUT_DIR / "upgrade_results.json"

NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]
SPLITS = ["train", "valid", "test"]
RATIOS = {"train": 0.70, "valid": 0.20, "test": 0.10}
TRAIN_KW = dict(epochs=100, batch=16, imgsz=640, lr0=0.01, seed=42, device=0,
                workers=8, cache=False, exist_ok=True, plots=False, verbose=False)
RESPLIT_SEEDS = [101, 102, 103, 104]
BACKUP_SEEDS = [105, 106, 107, 108, 109, 110]


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


def gather():
    records = []
    for split in SPLITS:
        for img in (SRC / split / "images").iterdir():
            st = source_stem(img.name)
            cid = content_id(st)
            records.append({"img": img, "lbl": SRC / split / "labels" / (img.stem + ".txt"),
                            "stem": st, "cid": cid, "pid": parent_id(cid)})
    return records


def build_resplit(records, seed, dst):
    rng = random.Random(seed)
    by_content = defaultdict(list)
    for r in records:
        by_content[r["cid"]].append(r)
    rep_of = {cid: sorted(rs, key=lambda x: (len(x["img"].name), x["img"].name))[0]
              for cid, rs in by_content.items()}
    par_contents = defaultdict(list)
    for cid, r in rep_of.items():
        par_contents[r["pid"]].append(cid)
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
    pids = list(par_vec)
    rng.shuffle(pids)  # 시드 주입 1: 동순위 순서 무작위화
    order = sorted(pids, key=lambda p: -sum(par_vec[p][c] * rarity[c] for c in range(5)))
    cur = {s: Counter() for s in SPLITS}
    cur_cont = {s: 0 for s in SPLITS}
    assign = {}
    for pid in order:
        v, n = par_vec[pid], par_ncont[pid]
        best, best_score = None, None
        for s in SPLITS:
            gain = sum(rarity[c] * v[c] * max(0.0, targets[s][c] - cur[s][c])
                       / max(1.0, targets[s][c]) for c in range(5))
            pressure = (targets_cont[s] - cur_cont[s]) / targets_cont[s]
            score = gain + 0.05 * pressure + rng.uniform(0, 0.03)  # 시드 주입 2: 지터
            if best_score is None or score > best_score:
                best, best_score = s, score
        assign[pid] = best
        cur[best] += v
        cur_cont[best] += n

    if dst.exists():
        shutil.rmtree(ep(dst))
    for s in SPLITS:
        (dst / s / "images").mkdir(parents=True)
        (dst / s / "labels").mkdir(parents=True)
    counters = {s: 0 for s in SPLITS}
    inst = {s: Counter() for s in SPLITS}
    for pid, cids in par_contents.items():
        s = assign[pid]
        chosen = ([r for r in records if r["pid"] == pid] if s == "train"
                  else [rep_of[cid] for cid in cids])
        for r in chosen:
            counters[s] += 1
            nid = f"{s}_{counters[s]:05d}"
            shutil.copyfile(ep(r["img"]), ep(dst / s / "images" / f"{nid}.jpg"))
            text = open(ep(r["lbl"]), encoding="utf-8").read() if r["lbl"].exists() else ""
            with open(ep(dst / s / "labels" / f"{nid}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            for c, k in label_classes(r["lbl"]).items():
                inst[s][NAMES[c]] += k
    with open(ep(dst / "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"train: {dst / 'train' / 'images'}\nval: {dst / 'valid' / 'images'}\n"
                f"test: {dst / 'test' / 'images'}\n\nnc: 5\nnames: {NAMES}\n")
    ok = all(inst[s][n] > 0 for s in ("valid", "test") for n in NAMES)
    return ok, {"files": counters, "parents": dict(Counter(assign.values())),
                "instances": {s: dict(inst[s]) for s in SPLITS}}


def evaluate(weights, data_yaml, tag):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data=str(data_yaml), split="test", batch=16, imgsz=640, device=0,
              workers=8, verbose=False, plots=False)
    ap50 = {m.names[ci]: round(float(r.box.ap50[i]), 4)
            for i, ci in enumerate(r.box.ap_class_index.tolist())}
    res = {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
           "per_class_ap50": ap50}
    print(f"[{tag}] mAP50={res['mAP50']:.4f}")
    return res


def train(model_name, data_yaml, run_name):
    from ultralytics import YOLO
    m = YOLO(model_name)
    m.train(data=str(data_yaml), project=str(LOCAL), name=run_name, **TRAIN_KW)
    return LOCAL / run_name / "weights" / "best.pt"


def save(results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    # ---- R2: 조건 (d) ----
    if "R2_condition_d" not in results:
        w = Path(r"D:\runs\weld2\E2_v5n\weights\best.pt")
        results["R2_condition_d"] = {
            "desc": "clean-trained v5nu (seed42) on RAW leaky test",
            **evaluate(w, RAW_YAML, "R2 (d) clean->raw")}
        save(results)

    # ---- R1: 재분할 로버스트니스 ----
    records = gather()
    results.setdefault("R1_resplit", {})
    pool = RESPLIT_SEEDS + BACKUP_SEEDS
    done = 0
    for seed in pool:
        if done >= len(RESPLIT_SEEDS):
            break
        key = f"rs{seed}"
        if key in results["R1_resplit"]:
            done += 1
            continue
        dst = LOCAL / f"resplit_{seed}"
        ok, summary = build_resplit(records, seed, dst)
        print(f"[R1 {key}] built ok={ok} files={summary['files']} "
              f"test_inst={summary['instances']['test']}")
        if not ok:
            results["R1_resplit"][key] = {"skipped": "class missing in valid/test",
                                          "summary": summary}
            save(results)
            continue
        w = train("yolov5nu.pt", dst / "data.yaml", f"R1_v5n_rs{seed}")
        ev = evaluate(w, dst / "data.yaml", f"R1 {key}")
        results["R1_resplit"][key] = {"split_seed": seed, "train_seed": 42,
                                      "summary": summary, **ev}
        done += 1
        save(results)

    # ---- R3: 제2 아키텍처 (YOLO11n) ----
    results.setdefault("R3_arch2", {})
    if "raw" not in results["R3_arch2"]:
        w = train("yolo11n.pt", RAW_YAML, "R3_11n_raw")
        results["R3_arch2"]["raw"] = {
            "desc": "YOLO11n raw-trained on raw leaky test (a')",
            **evaluate(w, RAW_YAML, "R3 11n a'")}
        save(results)
    if "clean" not in results["R3_arch2"]:
        w = train("yolo11n.pt", CLEAN_YAML, "R3_11n_clean")
        results["R3_arch2"]["clean"] = {
            "desc": "YOLO11n clean-trained on clean test (c')",
            **evaluate(w, CLEAN_YAML, "R3 11n c'")}
        save(results)

    # ---- 요약 ----
    import statistics
    a = 0.8313
    cs = [v["mAP50"] for v in results["R1_resplit"].values() if "mAP50" in v]
    if cs:
        cs_all = cs + [0.5770]  # 본 실험 seed42 스플릿 포함
        results["summary"] = {
            "a_raw": a,
            "c_gate_d_split42": 0.5770,
            "c_resplits": cs,
            "c_all_splits_mean": round(statistics.mean(cs_all), 4),
            "c_all_splits_std": round(statistics.stdev(cs_all), 4),
            "inflation_mean": round(a - statistics.mean(cs_all), 4),
            "inflation_per_split": [round(a - c, 4) for c in cs_all],
        }
        if "raw" in results["R3_arch2"] and "clean" in results["R3_arch2"]:
            ap = results["R3_arch2"]["raw"]["mAP50"]
            cp = results["R3_arch2"]["clean"]["mAP50"]
            results["summary"]["arch2_inflation"] = round(ap - cp, 4)
        save(results)
        print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
