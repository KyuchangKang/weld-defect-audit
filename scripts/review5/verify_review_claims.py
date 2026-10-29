# -*- coding: utf-8 -*-
"""2026-08-21 리뷰(논문심사의견_weld-defect-audit.md) 지적의 사실 검증.
   학습 없음 — 기존 산출물과 원 배포본만 사용. 결과는 evidence/claims.json 에 저장."""
import csv, re, os, json, collections

import pathlib as _pl, sys as _sys  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # weldpaths.py is in scripts/
from weldpaths import project_dir, runs_dir  # noqa: E402  (path defaults; see weldpaths.py)
ROOT = str(project_dir(r"D:\Sync\Dropbox\paper_wedling2"))
RAW  = os.path.join(ROOT, r'04_experiments\data\final0')
MAN  = os.path.join(ROOT, r'04_experiments\data\final0_clean\manifest.csv')
BIS  = str(runs_dir(r"D:\runs\weld2") / "review4")
OODJ = os.path.join(ROOT, r'05_results\upgrade\ood_reanalysis.json')
NAMES = ['Geometric defect', 'Non-fusion defect', 'crack', 'porosity', 'spatters']
out = {}

# ---------- M1: 누출 메커니즘 ----------
AUG = re.compile(r'_(?:0|90|180|270)(?:_flip)?$|_flip$', re.I)
def base(fn):
    s = re.sub(r'\.rf\.[0-9a-f]{32}.*$', '', fn, flags=re.I)
    return re.sub(r'_(jpg|jpeg|png)$', '', s, flags=re.I)

m1 = {}
for sp in ('train', 'valid', 'test'):
    fs = os.listdir(os.path.join(RAW, sp, 'images'))
    aug = [f for f in fs if AUG.search(base(f))]
    m1[sp] = {'files': len(fs), 'augmented': len(aug),
              'pct': round(100 * len(aug) / len(fs), 1),
              'tiles': sum(1 for f in fs if '_block_' in f.lower())}
readme = open(os.path.join(RAW, 'README.roboflow.txt'), encoding='utf-8').read()
m1['roboflow_readme_states_no_augmentation'] = 'No image augmentation techniques were applied' in readme
m1['preprocessing_declared'] = re.findall(r'\* (.+)', readme)[:4]
out['M1_leakage_mechanism'] = m1

# ---------- M2: 이분 30장의 구성 ----------
def compo(split_dir):
    imgs = sorted(os.listdir(os.path.join(split_dir, 'images')))
    inst, nimg = collections.Counter(), collections.Counter()
    for f in imgs:
        lp = os.path.join(split_dir, 'labels', os.path.splitext(f)[0] + '.txt')
        seen = set()
        if os.path.exists(lp):
            for ln in open(lp):
                if ln.strip():
                    c = NAMES[int(ln.split()[0])]; inst[c] += 1; seen.add(c)
        for c in seen: nimg[c] += 1
    return imgs, {n: {'instances': inst[n], 'images': nimg[n]} for n in NAMES}

un_files, un_cls = compo(os.path.join(BIS, 'test_unseen'))
se_files, se_cls = compo(os.path.join(BIS, 'test_seen'))

# 웹 출처 의심 = 숫자-숫자- 패턴이 아닌 서술형 파일명
web = [f for f in un_files if not re.match(r'^\d+-\d+-', f)]
rows = list(csv.DictReader(open(MAN, encoding='utf-8-sig')))
by = {r['new_id']: r for r in rows}
ood_ids = json.load(open(OODJ, encoding='utf-8'))['ood_ids']
ood_names = [by['test_' + i]['orig_filename'] for i in ood_ids if 'test_' + i in by]

def family(fn):   # 파일명 계열 (숫자·해시 제거)
    s = base(fn).lower()
    return re.sub(r'[-_]?\d+[-_]?$', '', re.sub(r'[-_]\d+[-_]', '-', s)).strip('-_')
ood_fams = {family(f) for f in ood_names}
matched = [(f, family(f)) for f in web if family(f) in ood_fams]

out['M2_bisection_composition'] = {
    'unseen': {'n_images': len(un_files), 'classes': un_cls,
               'web_style_filenames': len(web), 'web_files': web},
    'seen':   {'n_images': len(se_files), 'classes': se_cls},
    'clean_test_ood_14_filenames': ood_names,
    'web_files_sharing_family_with_confirmed_ood': matched,
    'NOTE': '도메인 최종 판정은 육안 확인 필요 — 파일명은 정황증거',
}

# ---------- M7: FPS 역전 ----------
sp = os.path.join(ROOT, r'05_results\e5_speed.csv')
if os.path.exists(sp):
    r = {x['model']: x for x in csv.DictReader(open(sp, encoding='utf-8-sig'))} \
        if 'model' in open(sp, encoding='utf-8-sig').readline() else {}
    out['M7_fps_note'] = {'source': 'e5_speed.csv', 'rows': list(r)[:12]}

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'claims.json'),
                    'w', encoding='utf-8'), ensure_ascii=False, indent=2)

print('=== M1 원 배포본 공식 스플릿별 증강 사본 ===')
for sp_ in ('train', 'valid', 'test'):
    d = m1[sp_]
    print('  %-6s %4d files / 증강 %4d (%4.1f%%) / 타일 %3d' %
          (sp_, d['files'], d['augmented'], d['pct'], d['tiles']))
print('  Roboflow README "No image augmentation techniques were applied":',
      m1['roboflow_readme_states_no_augmentation'])
print()
print('=== M2 이분 구성 ===')
print('  %-18s %-22s %s' % ('class', 'unseen(30)', 'seen(280)'))
for n in NAMES:
    print('  %-18s inst %4d / img %3d      inst %4d / img %3d'
          % (n, un_cls[n]['instances'], un_cls[n]['images'],
             se_cls[n]['instances'], se_cls[n]['images']))
print('  웹출처 의심 파일명: %d / 30' % len(web))
print('  그중 확인된 OOD 14장과 파일명 계열 일치: %d' % len(matched))
for f, fam in matched: print('     %-58s  (계열: %s)' % (f[:58], fam))
