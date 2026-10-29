# -*- coding: utf-8 -*-
"""T3 inter-reader agreement: author (2nd reader) vs Claude, both under the author's rule v3.

Inputs (results/T3/):
  판독표_저자.csv       author, rule v3 (판독규칙_v3_저자확정.md)
  판독표_Claude_v3.csv   Claude re-reading under rule v3, frozen before the author sheet was opened
  judge_claude.jsonl     Claude 1st reading (rule v1), mapped through key.json author_pairs
Primary kappa = same scene (SAME+ADJ) vs DIFF on the 48 pairs that are not rule examples, UNSURE excluded.
"""
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir, runs_dir  # noqa: E402,F401  (path defaults; see weldpaths.py)
_T3 = Path(__file__).resolve().parents[2] / "results" / "revision2" / "T3"  # was work/results/T3

T3 = _T3
EXAMPLES = {"A06", "A25", "A40", "A08", "A28", "A26", "A29", "A47", "A30", "A51", "A57", "A23"}  # rule v3 §5


TYPOS = {"UNSRUE": "UNSURE"}  # author sheet A51, obvious transposition (recorded in agreement.json)
typo_log = []


def read_rows(p):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return list(csv.reader(open(p, encoding=enc, newline="")))
        except UnicodeDecodeError:
            continue


def read_sheet(p, col=1, note_col=2):
    out = {}
    for r in read_rows(p)[1:]:
        if not r or not r[0].strip():
            continue
        j = r[col].strip().upper()
        if j in TYPOS:
            typo_log.append(f"{p.name}:{r[0]} {j}->{TYPOS[j]}")
            j = TYPOS[j]
        out[r[0].strip()] = (j, r[note_col].strip() if len(r) > note_col else "")
    return out


def read_practice(p):
    """Column 5 of the author sheet: a separate 'Claude practice reading' (header says: not the author's reading)."""
    out = {}
    for r in read_rows(p)[1:]:
        if len(r) > 4 and r[4].strip():
            f = next(csv.reader([r[4]]))
            out[f[0].strip()] = (f[1].strip().upper(), ",".join(f[2:]).strip())
    return out


def scene(j):
    return {"SAME": "S", "ADJ": "S", "DIFF": "D", "UNSURE": "U"}[j]


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None, 0
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (round((po - pe) / (1 - pe), 3) if pe < 1 else None), n


def comembership(part, letters):
    blocks = [set(b) for b in part.split("/")]
    assert sorted("".join(sorted(b)) for b in blocks) and set().union(*blocks) == set(letters), part
    blk = {c: i for i, b in enumerate(blocks) for c in b}
    return {(x, y): blk[x] == blk[y] for x, y in combinations(sorted(letters), 2)}


au, cl = read_sheet(T3 / "판독표_저자.csv"), read_sheet(T3 / "판독표_Claude_v3.csv")
pr = read_practice(T3 / "판독표_저자.csv")
key = json.load(open(T3 / "key.json", encoding="utf-8"))
v1 = {json.loads(l)["id"]: json.loads(l)["j"] for l in open(T3 / "judge_claude.jsonl", encoding="utf-8")}
a2p = {x["aid"]: x["pid"] for x in key["author_pairs"]}
pstrat = {p["id"]: p["strata"][0] for p in key["pairs"]}

ids = sorted(a2p)
assert set(ids) <= set(au) and set(ids) <= set(cl), "missing ids"
bad = [i for i in ids if au[i][0] not in ("SAME", "ADJ", "DIFF", "UNSURE")]
assert not bad, bad

out = {"n_pairs": len(ids), "examples_excluded": sorted(EXAMPLES)}
for name, sel in (("primary_48", [i for i in ids if i not in EXAMPLES]), ("all_60", ids)):
    A = [au[i][0] for i in sel]
    C = [cl[i][0] for i in sel]
    four = sum(a == c for a, c in zip(A, C))
    three = [(scene(a), scene(c)) for a, c in zip(A, C)]
    binary = [(x, y) for x, y in three if "U" not in (x, y)]
    k2, n2 = kappa(binary)
    k3, n3 = kappa(three)
    out[name] = {
        "n": len(sel),
        "author_counts": dict(Counter(A)), "claude_counts": dict(Counter(C)),
        "agree_4cat": f"{four}/{len(sel)}",
        "agree_3cat": f"{sum(x == y for x, y in three)}/{len(sel)}",
        "kappa_3cat_scene_diff_unsure": k3,
        "binary_n_both_decided": n2,
        "binary_agree": f"{sum(x == y for x, y in binary)}/{n2}",
        "kappa_binary_scene_vs_diff": k2,
        "unsure_either": sum("U" in p for p in three),
        "confusion_3cat": {f"{x}->{y}": c for (x, y), c in sorted(Counter(three).items())},
    }
# disagreements, with stratum and Claude v1
out["disagreements_3cat"] = [
    {"id": i, "author": au[i][0], "claude_v3": cl[i][0], "claude_v1": v1.get(a2p[i]),
     "stratum": pstrat.get(a2p[i]), "example": i in EXAMPLES, "author_note": au[i][1], "claude_note": cl[i][1]}
    for i in ids if scene(au[i][0]) != scene(cl[i][0])]
# Claude self-consistency v1 -> v3 and author vs v1
out["claude_v1_vs_v3_3cat"] = dict(Counter(f"{scene(v1[a2p[i]])}->{scene(cl[i][0])}" for i in ids))
three_v1 = [(scene(au[i][0]), scene(v1[a2p[i]])) for i in ids if i not in EXAMPLES]
out["author_vs_claude_v1_primary"] = {
    "agree_3cat": f"{sum(x == y for x, y in three_v1)}/{len(three_v1)}",
    "kappa_3cat": kappa(three_v1)[0],
    "kappa_binary": kappa([(x, y) for x, y in three_v1 if "U" not in (x, y)])[0]}
# by stratum (primary)
bys = {}
for i in ids:
    if i in EXAMPLES:
        continue
    s = pstrat[a2p[i]]
    d = bys.setdefault(s, Counter())
    d["n"] += 1
    d["agree3"] += scene(au[i][0]) == scene(cl[i][0])
    d["author_U"] += au[i][0] == "UNSURE"
    d["claude_U"] += cl[i][0] == "UNSURE"
out["by_stratum_primary"] = {s: dict(c) for s, c in sorted(bys.items())}
# groups
g = {}
for x in key["author_groups"]:
    aid = x["aid"]
    pa, pc = au[aid][0].replace(" ", ""), cl[aid][0].replace(" ", "")
    letters = set(pc.replace("/", ""))
    ma, mc = comembership(pa, letters), comembership(pc, letters)
    g[aid] = {"author": pa, "claude_v3": pc, "exact": pa == pc or comembership(pa, letters) == mc,
              "pair_links_agree": f"{sum(ma[k] == mc[k] for k in ma)}/{len(ma)}",
              "author_note": au[aid][1]}
out["groups"] = g
out["typos_normalised"] = typo_log
# supplementary: the practice reading in column 5 (not the author's reading; provenance to be confirmed by the author)
if pr:
    sup = {}
    for name, sel in (("primary_48", [i for i in ids if i not in EXAMPLES]), ("all_60", ids)):
        sel = [i for i in sel if i in pr]
        for other, src in (("author", au), ("claude_v3", cl)):
            three = [(scene(pr[i][0]), scene(src[i][0])) for i in sel]
            sup[f"{name}_practice_vs_{other}"] = {
                "n": len(three), "agree_3cat": f"{sum(x == y for x, y in three)}/{len(three)}",
                "kappa_3cat": kappa(three)[0],
                "kappa_binary": kappa([(x, y) for x, y in three if "U" not in (x, y)])[0]}
    sup["practice_counts_all_60"] = dict(Counter(pr[i][0] for i in ids if i in pr))
    sup["practice_groups"] = {x["aid"]: pr.get(x["aid"], ("", ""))[0] for x in key["author_groups"]}
    out["supplementary_practice_reading"] = sup
json.dump(out, open(T3 / "agreement.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("primary_48", "all_60", "author_vs_claude_v1_primary", "claude_v1_vs_v3_3cat")},
                 ensure_ascii=False, indent=1))
print("groups:", {k: (v["author"], v["claude_v3"], v["pair_links_agree"]) for k, v in g.items()})
print("disagreements:", len(out["disagreements_3cat"]))
for d in out["disagreements_3cat"]:
    print(d["id"], d["author"], "vs", d["claude_v3"], "| v1", d["claude_v1"], "|", d["stratum"], "|", d["author_note"])
