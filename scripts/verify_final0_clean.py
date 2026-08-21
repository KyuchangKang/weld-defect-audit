"""final0_clean 무결성 재확인 (experiment_plan.md §7 체크리스트 항목).

검사 항목:
  1. 스플릿별 이미지/라벨 수 (기대: train 3567 / valid 142 / test 71)
  2. 이미지-라벨 1:1 스템 매칭
  3. 라벨 형식(5컬럼) / 클래스 ID 0..4 / 좌표 [0,1] 범위
  4. 스플릿별 5클래스 전부 존재 + train/test 인스턴스 수를 CLAUDE.md 기록값과 대조
  5. manifest.csv 행수·파일 대응, parent_id 단위 누수 0건
결과: 콘솔 출력 + 04_experiments/data/final0_clean_verify.json

주의: BASE 절대경로는 PC에 맞게 수정 (현재 RTX PRO 5000 워크스테이션 기준).
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(r"D:\Sync\Dropbox\paper_wedling2\04_experiments\data\final0_clean")
EXPECTED_COUNTS = {"train": 3567, "valid": 142, "test": 71}
NC = 5
# CLAUDE.md 핵심 수치: 클래스 순서 Geometric/Non-fusion/crack/porosity/spatters
# test porosity 12→16: 2026-08-13 문서화 라벨 보정 (data/label_fixes_20260813/CHANGELOG.md)
EXPECTED_INSTANCES = {
    "train": {0: 1307, 1: 682, 2: 1044, 3: 1789, 4: 4353},
    "test": {0: 58, 1: 22, 2: 22, 3: 16, 4: 40},
}


def main():
    report = {"checks": [], "pass": True}

    def check(name, ok, detail=""):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
        if not ok:
            report["pass"] = False
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    for split, exp in EXPECTED_COUNTS.items():
        imgs = sorted((BASE / split / "images").glob("*"))
        lbls = sorted((BASE / split / "labels").glob("*.txt"))
        check(f"{split}: image count == {exp}", len(imgs) == exp, f"actual {len(imgs)}")
        check(f"{split}: label count == {exp}", len(lbls) == exp, f"actual {len(lbls)}")
        istems = {p.stem for p in imgs}
        lstems = {p.stem for p in lbls}
        check(f"{split}: image-label 1:1", istems == lstems,
              f"img-only {len(istems - lstems)}, lbl-only {len(lstems - istems)}")

        bad_fmt, bad_range, empty = 0, 0, 0
        classes = Counter()
        for lp in lbls:
            lines = [ln for ln in lp.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
            if not lines:
                empty += 1
                continue
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    bad_fmt += 1
                    continue
                try:
                    c = int(parts[0])
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    bad_fmt += 1
                    continue
                if not 0 <= c < NC:
                    bad_range += 1
                elif not all(0.0 <= v <= 1.0 for v in vals):
                    bad_range += 1
                else:
                    classes[c] += 1
        check(f"{split}: label format (5-col)", bad_fmt == 0, f"bad lines {bad_fmt}")
        check(f"{split}: class/coord range", bad_range == 0, f"out-of-range {bad_range}")
        check(f"{split}: all {NC} classes present", len(classes) == NC,
              dict(sorted(classes.items())))
        if split in EXPECTED_INSTANCES:
            check(f"{split}: instance counts == CLAUDE.md record",
                  dict(classes) == EXPECTED_INSTANCES[split],
                  f"actual {dict(sorted(classes.items()))}")
        report[f"{split}_instances"] = dict(sorted(classes.items()))
        report[f"{split}_empty_labels"] = empty

    with open(BASE / "manifest.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    check("manifest: row count == total files", len(rows) == sum(EXPECTED_COUNTS.values()),
          f"actual {len(rows)}")
    by_split = defaultdict(set)
    parent_splits = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["new_id"])
        parent_splits[r["parent_id"]].add(r["split"])
    for split in EXPECTED_COUNTS:
        stems = {p.stem for p in (BASE / split / "images").glob("*")}
        check(f"{split}: files == manifest new_id", stems == by_split[split],
              f"file-only {len(stems - by_split[split])}, manifest-only {len(by_split[split] - stems)}")
    leaks = {p: sorted(s) for p, s in parent_splits.items() if len(s) > 1}
    check("parent-level split leak == 0", not leaks, f"{len(leaks)} leaking parents")
    report["n_parents"] = len(parent_splits)

    out = BASE.parent / "final0_clean_verify.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'ALL PASS' if report['pass'] else 'FAILURES PRESENT'} -> {out}")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
