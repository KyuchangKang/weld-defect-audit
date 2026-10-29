# -*- coding: utf-8 -*-
import re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import weldpaths  # noqa: E402,F401  -- reads --root/--runs before sys.argv is reset below
sys.argv=['x']
exec(open(r'scripts/revision1/grouping_sensitivity.py',encoding='utf-8').read().split('files = []')[0])
files=[]
for split in ["train","valid","test"]:
    for img in sorted((SRC/split/"images").iterdir()): files.append((split,img))
for name,fn in [("stem",lambda n:source_stem(n)),("content",lambda n:content_id(source_stem(n))),("parent",lambda n:parent_id(content_id(source_stem(n))))]:
    g=defaultdict(set)
    for s,p in files: g[fn(p.name)].add(s)
    multi=sum(len(v)>1 for v in g.values())
    tv=sum(('train' in v) and len(v)>1 for v in g.values())
    print(name,len(g),'spanning',multi,'train+other',tv, 'pairs', {k:sum(v=={*k.split('+')} for v in g.values()) for k in ['train+valid','train+test','valid+test','train+valid+test']})
