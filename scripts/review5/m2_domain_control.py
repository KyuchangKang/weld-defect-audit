# -*- coding: utf-8 -*-
"""M2 — 이분(bisection) 대조의 도메인·클래스 구성 보고 + 통제 대조 재채점.

리뷰어 지적: (a-seen) 0.846 vs (a-unseen) 0.617의 격차 중 일부가 누출 부재가 아니라
             도메인 시프트 때문일 수 있다. 30장의 도메인 구성과 클래스 분포를 보고하라.

절차:
  1) 원 test 310장을 부모 사진 단위(230개)로 묶고, 부모 대표 이미지를 육안 판정
     (evidence/parent_sheets/*.png, 판정 결과는 아래 OOD_PARENTS)
  2) 부모 라벨을 모든 파일에 전파 → seen/unseen의 도메인·클래스 구성 산출
  3) 통제 대조 재채점 (조건 (a) 모델 고정, 평가셋만 교체 — 학습 없음):
       - field-only  : 현장 사진 부모만
       - geometric-only : 양쪽에 n이 충분한 유일 클래스
     둘 다 seen/unseen 각각 평가해 격차를 비교

⚠️ 도메인 판정은 주관적이다. 본 스크립트의 판정자는 원 감사자(저자)가 아니므로
   저자 확인이 필요하다. 경계 사례는 AMBIGUOUS에 기록했다.

출력: evidence/m2_domain_control.json
"""
import os, re, json, shutil, collections
from pathlib import Path

import pathlib as _pl, sys as _sys  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir, runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
CLEAN = BASE / "04_experiments" / "data" / "final0_clean"
UN = runs_dir(r"D:\runs\weld2") / "review4" / "test_unseen"
SE = runs_dir(r"D:\runs\weld2") / "review4" / "test_seen"
RAW_BEST = runs_dir(r"D:\runs\weld2") / "E4_v5n_raw" / "weights" / "best.pt"   # 조건 (a) 모델
LOCAL = runs_dir(r"D:\runs\weld2") / "m2_domain"
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["Geometric defect", "Non-fusion defect", "crack", "porosity", "spatters"]

# ── 육안 판정 결과 (parent_index.json의 1-based 번호) ─────────────────────────
OOD_PARENTS = set([8, 9, 10] + [127, 128, 129, 130, 131, 132]
                  + list(range(155, 181)) + list(range(181, 203))
                  + list(range(223, 231)))
AMBIGUOUS = {126: "green-mat crack macro — field로 판정(경계)",
             200: "welds_imperfections-56- — OOD로 판정(원 감사는 같은 계열 84/86을 field로 판정)",
             201: "welds_imperfections-63- — 동상",
             202: "welds_imperfections-65- — 동상",
             133: "img20220609081234 — field로 판정(경계)"}


def source_stem(n):
    s = re.sub(r"\.rf\.[0-9a-f]{32}.*$", "", n)
    return re.sub(r"_(jpeg|jpg|png|bmp)(_(jpeg|jpg|png|bmp))*$", "", s, flags=re.I).lower()


def content_id(st):
    p, pats = st, [r"[-_]+flip$", r"[-_]+(0|90|180|270)[-_]*$", r"[-_]+rot(0|90|180|270)$",
                   r"[-_]+copy(-\d+)?$", r"[-_]+\(\d+\)$", r"[-_]+aug\d*$"]
    ch = True
    while ch:
        ch = False
        for pt in pats:
            new = re.sub(pt, "", p)
            if new != p and len(new) >= 3:
                p, ch = new, True
    return p


def parent_id(n):
    return re.sub(r"_block_\d+_\d+$", "", content_id(source_stem(n)))


def classes_of(lbl_path):
    out = []
    if lbl_path.exists():
        for ln in lbl_path.read_text().splitlines():
            if ln.strip():
                out.append(NAMES[int(ln.split()[0])])
    return out


# ── 1) 부모 그룹 재구성 (make_parent_sheets.py와 동일 순서) ──────────────────
g = collections.OrderedDict()
for lbl, d in (("seen", SE), ("unseen", UN)):
    for f in sorted(os.listdir(d / "images")):
        g.setdefault(parent_id(f), []).append((lbl, d, f))
parents = list(g.items())
assert len(parents) == 230, len(parents)
pdom = {p: ("ood" if (i + 1) in OOD_PARENTS else "field") for i, (p, _) in enumerate(parents)}

# ── 2) 구성 집계 ─────────────────────────────────────────────────────────────
comp = {}
for split in ("seen", "unseen"):
    files = [(p, d, f) for p, v in parents for (l, d, f) in v if l == split]
    par = sorted({p for p, _, _ in files})
    inst = collections.Counter(); inst_dom = collections.defaultdict(collections.Counter)
    nimg = collections.Counter()
    for p, d, f in files:
        cs = classes_of(d / "labels" / (os.path.splitext(f)[0] + ".txt"))
        for c in cs:
            inst[c] += 1; inst_dom[pdom[p]][c] += 1
        nimg[pdom[p]] += 1
    comp[split] = {
        "n_images": len(files),
        "n_parents": len(par),
        "parents_field": sum(1 for x in par if pdom[x] == "field"),
        "parents_ood": sum(1 for x in par if pdom[x] == "ood"),
        "pct_parents_ood": round(100 * sum(1 for x in par if pdom[x] == "ood") / len(par), 1),
        "images_field": nimg["field"], "images_ood": nimg["ood"],
        "pct_images_ood": round(100 * nimg["ood"] / len(files), 1),
        "instances_total": {c: inst[c] for c in NAMES},
        "instances_field": {c: inst_dom["field"][c] for c in NAMES},
        "instances_ood": {c: inst_dom["ood"][c] for c in NAMES},
    }

print("=== M2 구성 보고 ===")
for s in ("seen", "unseen"):
    c = comp[s]
    print("  %-7s 이미지 %3d (field %3d / OOD %3d = %4.1f%%) | 부모 %3d (field %3d / OOD %3d = %4.1f%%)"
          % (s, c["n_images"], c["images_field"], c["images_ood"], c["pct_images_ood"],
             c["n_parents"], c["parents_field"], c["parents_ood"], c["pct_parents_ood"]))
print("  클래스 인스턴스 (전체 / field만)")
for c_ in NAMES:
    print("    %-18s seen %4d / %4d      unseen %3d / %3d"
          % (c_, comp["seen"]["instances_total"][c_], comp["seen"]["instances_field"][c_],
             comp["unseen"]["instances_total"][c_], comp["unseen"]["instances_field"][c_]))


# ── 3) 통제 대조용 부분집합 구성 + 재채점 ────────────────────────────────────
def build(split, mode, dst):
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True); (dst / "labels").mkdir(parents=True)
    n = 0
    for p, v in parents:
        for (l, d, f) in v:
            if l != split:
                continue
            if mode == "field" and pdom[p] != "field":
                continue
            base = os.path.splitext(f)[0]
            src_l = d / "labels" / (base + ".txt")
            if mode == "geom":
                keep = [ln for ln in (src_l.read_text().splitlines() if src_l.exists() else [])
                        if ln.strip() and int(ln.split()[0]) == 0]
                if not keep:
                    continue
                (dst / "labels" / (base + ".txt")).write_text("\n".join(keep) + "\n")
            else:
                shutil.copyfile(src_l, dst / "labels" / (base + ".txt")) if src_l.exists() else None
            shutil.copyfile(d / "images" / f, dst / "images" / f)
            n += 1
    return n


def yaml_for(test_dir, path, nc=5, names=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("train: %s\nval: %s\ntest: %s\n\nnc: %d\nnames: %s\n"
                    % (CLEAN / "train" / "images", CLEAN / "valid" / "images",
                       test_dir, nc, names or NAMES), encoding="utf-8")
    return path


def evaluate(yaml_path, tag):
    from ultralytics import YOLO
    m = YOLO(str(RAW_BEST))
    r = m.val(data=str(yaml_path), split="test", batch=16, imgsz=640, device=0,
              workers=0, verbose=False, plots=False,
              project=str(LOCAL), name=tag, exist_ok=True)
    idx = r.box.ap_class_index.tolist()
    res = {"mAP50": round(float(r.box.map50), 4),
           "mAP50_95": round(float(r.box.map), 4),
           "per_class_ap50": {m.names[c]: round(float(r.box.ap50[i]), 4)
                              for i, c in enumerate(idx)}}
    print("  [%-22s] mAP50=%.4f" % (tag, res["mAP50"]))
    return res


print("\n=== 통제 대조 재채점 (조건 (a) 모델 고정, 학습 0회) ===")
out = {"domain_labels": {"ood_parent_indices": sorted(OOD_PARENTS),
                         "ambiguous": AMBIGUOUS,
                         "note": "판정자는 원 감사자(저자)가 아님 — 저자 확인 필요"},
       "composition": comp, "contrasts": {}}

for mode, label in (("field", "field-only"), ("geom", "geometric-only")):
    r = {}
    for split in ("seen", "unseen"):
        dst = LOCAL / ("%s_%s" % (mode, split))
        n = build(split, mode, dst)
        y = yaml_for(dst / "images", LOCAL / ("%s_%s.yaml" % (mode, split)))
        r[split] = evaluate(y, "%s_%s" % (mode, split))
        r[split]["n_images"] = n
    r["gap_seen_minus_unseen"] = round(r["seen"]["mAP50"] - r["unseen"]["mAP50"], 4)
    out["contrasts"][label] = r
    print("  >>> %-14s seen %.4f (n=%d) - unseen %.4f (n=%d) = %+.4f"
          % (label, r["seen"]["mAP50"], r["seen"]["n_images"],
             r["unseen"]["mAP50"], r["unseen"]["n_images"], r["gap_seen_minus_unseen"]))

(HERE / "m2_domain_control.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n저장:", HERE / "m2_domain_control.json")
print("[대조] 원 보고: seen 0.846 vs unseen 0.617 = +0.229 (전체 310장)")
