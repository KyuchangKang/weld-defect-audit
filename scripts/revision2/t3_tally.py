# -*- coding: utf-8 -*-
"""T3: unblind Claude's first-pass judgments against key.json and tally per stratum.
Does NOT read the author's form (판독표_저자.csv); agreement is computed later by t3_agreement.py.
Output: results/T3/tally_claude.json"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir, runs_dir  # noqa: E402,F401  (path defaults; see weldpaths.py)
_T3 = Path(__file__).resolve().parents[2] / "results" / "revision2" / "T3"  # was work/results/T3

T3 = _T3
key = json.loads((T3 / "key.json").read_text(encoding="utf-8"))
J = {d["id"]: d for d in map(json.loads, open(T3 / "judge_claude.jsonl", encoding="utf-8"))}
G = {d["id"]: d for d in map(json.loads, open(T3 / "judge_claude_groups.jsonl", encoding="utf-8"))}
assert len(J) == len(key["pairs"]) and len(G) == len(key["groups"])

# ---- pairs: per stratum (a pair can carry several stratum labels; count it in each) ----
by = defaultdict(Counter)
dist = defaultdict(list)
for p in key["pairs"]:
    j = J[p["id"]]["j"]
    for s in p["strata"]:
        by[s][j] += 1
        dist[s].append((p["d"], j))
pairs_out = {}
for s, c in sorted(by.items()):
    n = sum(c.values())
    same_scene = c["SAME"] + c["ADJ"]
    decided = n - c["UNSURE"]
    pairs_out[s] = {"n": n, **{k: c[k] for k in ("SAME", "ADJ", "DIFF", "UNSURE")},
                    "same_scene_share_of_all": round(same_scene / n, 3),
                    "same_scene_share_of_decided": round(same_scene / decided, 3) if decided else None}

# ---- controls ----
pos = by["C_pos_same_content"]
neg = by["C_neg_far"]
controls = {"positive_correct": pos["SAME"] + pos["ADJ"], "positive_n": sum(pos.values()),
            "negative_false_same_scene": neg["SAME"] + neg["ADJ"], "negative_n": sum(neg.values())}
controls["pass"] = (controls["positive_correct"] / controls["positive_n"] >= 0.9 and
                    controls["negative_false_same_scene"] / controls["negative_n"] <= 0.1)

# ---- U6: test images whose nearest cross-split partner is the same scene ----
u6 = [p for p in key["pairs"] if "U6_test_nearest" in p["strata"]]
u6_hit = sorted({p["test_id"] for p in u6 if J[p["id"]]["j"] in ("SAME", "ADJ")})
u6_unsure = sorted({p["test_id"] for p in u6 if J[p["id"]]["j"] == "UNSURE"})
u6_by_d = Counter(("le4" if p["d"] <= 4 else "5-8" if p["d"] <= 8 else "9-12", J[p["id"]]["j"]) for p in u6)

# ---- groups ----
def nblocks(s):
    return len(s.split("/"))

gout = defaultdict(lambda: {"n": 0, "split_needed": 0, "single_component": 0, "blocks_total": 0, "reps_total": 0})
over = []
for g in key["groups"]:
    j = G[g["id"]]
    o = gout[g["stratum"]]
    o["n"] += 1
    k = nblocks(j["j"])
    o["blocks_total"] += k
    o["reps_total"] += len(g["reps"])
    if len(g["reps"]) == 1:
        o["single_component"] += 1
    if k > 1:
        o["split_needed"] += 1
    if "over-grouping" in j["n"]:
        over.append(g["id"])

out = {"pairs": pairs_out, "controls": controls,
       "U6": {"test_images": len({p["test_id"] for p in u6}), "same_scene_test_ids": u6_hit,
              "unsure_test_ids": u6_unsure, "by_distance": {f"{a}:{b}": v for (a, b), v in sorted(u6_by_d.items())}},
       "groups": dict(gout), "groups_flagged_over_grouping": over}
(T3 / "tally_claude.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
