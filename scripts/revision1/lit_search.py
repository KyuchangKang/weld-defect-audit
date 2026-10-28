# -*- coding: utf-8 -*-
"""R1-2 / R2-2,3: 재현 가능한 문헌 검색 기록 (OpenAlex API). 제목+초록 검색, 2023-01-01 ~ 2026-08-24 (투고일).
검색 결과 레코드 전부를 저장하고, 제목/초록에서 데이터 무결성 관련 용어 언급 여부를 선별한다."""
import json, re, time, urllib.parse, urllib.request
from pathlib import Path
OUT = Path(r"D:\Sync\Dropbox\paper_wedling2\08_revision\1st_revision\work\results")
QUERIES = {
  "Q1": '(weld OR welding) AND defect AND (YOLO OR "object detection" OR "deep learning")',
}
FILTER = "from_publication_date:2023-01-01,to_publication_date:2026-08-24,type:article"
TERMS = re.compile(r"leak|duplicat|near-duplicate|dedup|data split|split integrity|overlap between (the )?(training|train) and test|train/test overlap|contaminat", re.I)
def inv(ab):
    if not ab: return ""
    pos = [(p, w) for w, ps in ab.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))
def fetch(q):
    recs, cursor = [], "*"
    while cursor:
        url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(q) + "&filter=" + FILTER +
               "&per-page=200&cursor=" + urllib.parse.quote(cursor) + "&select=id,doi,title,publication_year,abstract_inverted_index,primary_location")
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "weld-audit-revision"})))
        recs += d["results"]; cursor = d["meta"].get("next_cursor")
        if not d["results"]: break
        time.sleep(0.2)
    return recs, d["meta"]["count"]
summary = {"run_date": time.strftime("%Y-%m-%d"), "database": "OpenAlex", "filter": FILTER, "queries": {}}
for k, q in QUERIES.items():
    recs, n = fetch(q)
    rows = []
    for r in recs:
        t = r.get("title") or ""; a = inv(r.get("abstract_inverted_index"))
        rel = bool(re.search(r"weld", t + " " + a, re.I)) and bool(re.search(r"defect|flaw|imperfection", t + " " + a, re.I))
        rows.append({"id": r["id"], "doi": r.get("doi"), "year": r.get("publication_year"), "title": t,
                     "has_abstract": bool(a), "weld_defect_topic": rel,
                     "integrity_terms": sorted(set(m.group(0).lower() for m in TERMS.finditer(t + " " + a)))})
    on = [x for x in rows if x["weld_defect_topic"]]
    summary["queries"][k] = {"query": q, "count": n, "retrieved": len(rows), "weld_defect_topic": len(on),
                             "with_abstract": sum(x["has_abstract"] for x in on),
                             "integrity_term_hits": [x for x in on if x["integrity_terms"]]}
    json.dump(rows, open(OUT / f"lit_search_{k}_records.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
json.dump(summary, open(OUT / "lit_search_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in summary["queries"].items():
    print(k, v["count"], v["retrieved"], v["weld_defect_topic"], v["with_abstract"], len(v["integrity_term_hits"]))
    for h in v["integrity_term_hits"]: print("   ", h["year"], h["integrity_terms"], h["title"][:110], h["doi"])
