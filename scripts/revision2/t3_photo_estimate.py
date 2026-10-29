# -*- coding: utf-8 -*-
"""T3: estimate the number of independent photographs after merging parent links that the blinded visual
audit judged to be the same scene (SAME or ADJ). Parents use the rule as specified (tile fix).
U3 (d<=4) is a census; U4 (d 5-8) and U5 (d 9-12) are random samples -> unsampled links in those bands are
merged with the band's judged same-scene rate in a Monte Carlo (UNSURE counted as not same, then as same,
to bracket). Links at d > 12 are not merged (not audited).
Output: results/T3/photo_estimate.json"""
import json
import random
import re
from pathlib import Path
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir, runs_dir  # noqa: E402,F401  (path defaults; see weldpaths.py)
_T3 = Path(__file__).resolve().parents[2] / "results" / "revision2" / "T3"  # was work/results/T3

import numpy as np

HERE = Path(__file__).resolve().parent
exec(open(HERE / "t3_population.py", encoding="utf-8").read().split("def pc(x)")[0])
T3 = OUT
TILE = re.compile(r"^(.*)_block_\d+_\d+$")
pid_fixed = [TILE.match(s).group(1) if TILE.match(s) else p for s, p in zip(stem, pid)]
key = json.loads((T3 / "key.json").read_text(encoding="utf-8"))
J = {d["id"]: d["j"] for d in map(json.loads, open(T3 / "judge_claude.jsonl", encoding="utf-8"))}
Z = np.load(T3 / "pairs.npz")
I, D = Z["I"], Z["D"]
Jj = Z["J"]

# parent-level links (closest pair) under the fixed parent rule
link = {}
for a, b, d in zip(I, Jj, D):
    pa, pb = pid_fixed[a], pid_fixed[b]
    if pa == pb or d > 12:
        continue
    k = tuple(sorted((pa, pb)))
    if k not in link or d < link[k][0]:
        link[k] = (int(d), int(a), int(b))
judged = {}
for p in key["pairs"]:
    k = tuple(sorted((pid_fixed[p["a"]], pid_fixed[p["b"]])))
    if k[0] != k[1]:
        judged[k] = J[p["id"]]
parents = sorted(set(pid_fixed))


def components(edges):
    par = {p: p for p in parents}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in edges:
        par[f(a)] = f(b)
    return len({f(p) for p in parents})


def band(d):
    return "le4" if d <= 4 else "5-8" if d <= 8 else "9-12"


rates = {}
for unsure_as in ("not", "same"):
    r = {}
    for bnd in ("le4", "5-8", "9-12"):
        js = [judged[k] for k, v in link.items() if band(v[0]) == bnd and k in judged]
        pos = sum(j in ("SAME", "ADJ") or (unsure_as == "same" and j == "UNSURE") for j in js)
        r[bnd] = (pos / len(js)) if js else None
    rates[unsure_as] = r

res = {"parents_rule_as_specified": len(parents),
       "links_by_band": {b: sum(band(v[0]) == b for v in link.values()) for b in ("le4", "5-8", "9-12")},
       "judged_links_by_band": {b: sum(band(v[0]) == b and k in judged for k, v in link.items()) for b in ("le4", "5-8", "9-12")},
       "same_scene_rate": rates}
# deterministic bounds
judged_pos = [k for k, j in judged.items() if j in ("SAME", "ADJ") and k in link]
res["photos_merge_judged_positive_only"] = components(judged_pos)
res["photos_merge_all_links_le12"] = components(list(link))
res["photos_merge_all_links_le4"] = components([k for k, v in link.items() if v[0] <= 4])
rng = random.Random(7)
for unsure_as in ("not", "same"):
    sims = []
    for _ in range(200):
        e = []
        for k, v in link.items():
            if k in judged:
                j = judged[k]
                if j in ("SAME", "ADJ") or (unsure_as == "same" and j == "UNSURE"):
                    e.append(k)
            elif rng.random() < rates[unsure_as][band(v[0])]:
                e.append(k)
        sims.append(components(e))
    sims.sort()
    res[f"photos_montecarlo_unsure_{unsure_as}"] = {"median": sims[100], "p2.5": sims[5], "p97.5": sims[194]}
(T3 / "photo_estimate.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
