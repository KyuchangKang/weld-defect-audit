# -*- coding: utf-8 -*-
"""M5b — 선행논문 [1]이 사용한 공개 데이터셋의 분할 무결성 감사.

대상: Kaggle "Welding Defect - Object Detection" (sukmaadhiwijaya, CC0)
      3 classes: bad weld / good weld / defect  — [1]의 3클래스 과제와 일치

본 연구(final0)에 적용한 것과 **동일한 프로토콜**로 감사해 수치를 직접 비교 가능하게 한다:
  (1) 파일 인벤토리 (공식 train/valid/test)
  (2) MD5 완전 중복 — 스플릿 내 / 스플릿 간
  (3) dihedral 불변 pHash 근접중복 (d = 0 / 4 / 8) — 파일명 규약과 무관하게 동작
  (4) 근접중복 연결요소 = '부모 사진' 근사 → 스플릿을 걸치는 부모 비율 (핵심 수치)
  (5) 파일명 계보(있으면) — 증강 접미사가 valid/test에 존재하는지 = 업로드 후 분할 증거

사용:
  python audit_prior_dataset.py <데이터셋_루트>
  (루트 아래에 train/valid(또는 val)/test 각각 images/ 가 있는 구조를 기대)

출력: evidence/prior_dataset_audit.json  +  콘솔 요약
"""
import sys, os, re, json, hashlib
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

THRESHOLDS = [0, 4, 8]
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
AUG_RE = re.compile(r'[-_](?:0|90|180|270)(?:[-_]?flip)?$|[-_]flip$|[-_]rot(?:0|90|180|270)$'
                    r'|[-_]copy(?:-\d+)?$|[-_]aug\d*$', re.I)


def find_splits(root):
    """루트 아래에서 train/valid/test 이미지 폴더를 찾는다 (중첩 허용)."""
    root = Path(root)
    cands = {}
    for p in root.rglob('*'):
        if not p.is_dir():
            continue
        n = p.name.lower()
        if n in ('train', 'valid', 'val', 'test'):
            imgs = p / 'images'
            d = imgs if imgs.is_dir() else p
            files = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXT] if d.is_dir() else []
            if files:
                key = 'valid' if n == 'val' else n
                # 같은 이름이 여러 번 나오면 파일이 더 많은 쪽 채택 (v1/v2 중첩 대비)
                if key not in cands or len(files) > len(cands[key][1]):
                    cands[key] = (d, files)
    return cands


def md5(path, buf=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(buf), b''):
            h.update(chunk)
    return h.hexdigest()


def dihedral_hashes(img):
    hs = []
    for k in range(4):
        r = img.rotate(90 * k, expand=True)
        for m in (r, r.transpose(Image.FLIP_LEFT_RIGHT)):
            hs.append(np.packbits(imagehash.phash(m).hash.flatten()).view(np.uint64)[0])
    return hs


def popcount64(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def main(root):
    splits = find_splits(root)
    if not splits:
        print('[!] train/valid/test 이미지 폴더를 찾지 못했습니다:', root); return
    out = {'root': str(root), 'splits': {}}
    print('=== (1) 인벤토리 ===')
    files, owner = [], []
    for sp in ('train', 'valid', 'test'):
        if sp not in splits:
            print('  %-6s 없음' % sp); continue
        d, fs = splits[sp]
        out['splits'][sp] = {'dir': str(d), 'n_images': len(fs)}
        print('  %-6s %5d  (%s)' % (sp, len(fs), d))
        files += fs; owner += [sp] * len(fs)
    out['total_images'] = len(files)
    print('  합계 %d' % len(files))

    # --- (2) MD5 ---
    print('\n=== (2) MD5 완전 중복 ===')
    md5map = defaultdict(list)
    for f, sp in zip(files, owner):
        md5map[md5(f)].append(sp)
    dup_groups = {k: v for k, v in md5map.items() if len(v) > 1}
    cross = {k: v for k, v in dup_groups.items() if len(set(v)) > 1}
    print('  고유 MD5 %d / 중복 그룹 %d / 그중 스플릿을 걸침 %d'
          % (len(md5map), len(dup_groups), len(cross)))
    out['md5'] = {'unique': len(md5map), 'dup_groups': len(dup_groups),
                  'cross_split_groups': len(cross),
                  'cross_split_pairs_example': [sorted(set(v)) for v in list(cross.values())[:10]]}

    # --- (3) pHash ---
    print('\n=== (3) dihedral pHash 계산 ===')
    H = np.zeros((len(files), 8), dtype=np.uint64)
    for i, f in enumerate(files):
        try:
            with Image.open(f) as im:
                H[i] = dihedral_hashes(im.convert('L'))
        except Exception as e:
            H[i] = 0
            print('   [skip] %s (%s)' % (f.name, e))
        if (i + 1) % 500 == 0:
            print('   %d/%d' % (i + 1, len(files)))
    base = H[:, 0].copy()

    print('\n=== (4) 근접중복과 스플릿 누출 ===')
    res = {}
    for d in THRESHOLDS:
        pairs, adj = [], defaultdict(set)
        for i in range(len(files)):
            if base[i] == 0:
                continue
            x = np.bitwise_xor(H[i][None, :], base[i + 1:, None])   # (N-i-1, 8)
            dist = popcount64(x).min(axis=1)
            for off in np.nonzero(dist <= d)[0]:
                j = i + 1 + int(off)
                pairs.append((i, j))
                adj[i].add(j); adj[j].add(i)
        # 연결요소 = 부모 근사
        seen, comps = set(), []
        for i in range(len(files)):
            if i in seen:
                continue
            stack, comp = [i], []
            seen.add(i)
            while stack:
                u = stack.pop(); comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v); stack.append(v)
            comps.append(comp)
        multi = [c for c in comps if len(c) > 1]
        spanning = [c for c in multi if len({owner[k] for k in c}) > 1]
        cross_pairs = [(i, j) for i, j in pairs if owner[i] != owner[j]]
        n_par = len(comps)
        res[str(d)] = {
            'near_pairs': len(pairs),
            'cross_split_pairs': len(cross_pairs),
            'components_total': n_par,
            'components_multi_image': len(multi),
            'components_spanning_splits': len(spanning),
            'pct_of_multi_spanning': round(100 * len(spanning) / len(multi), 1) if multi else 0.0,
            'pct_of_all_parents_spanning': round(100 * len(spanning) / n_par, 1) if n_par else 0.0,
            'redundancy_factor': round(len(files) / n_par, 2) if n_par else None,
        }
        print('  d=%d | 근접쌍 %6d (스플릿간 %5d) | 부모근사 %5d | 다중이미지 %5d '
              '| 스플릿 걸침 %4d (%.1f%% of 다중, %.1f%% of 전체) | 중복배수 %.2f'
              % (d, len(pairs), len(cross_pairs), n_par, len(multi), len(spanning),
                 res[str(d)]['pct_of_multi_spanning'],
                 res[str(d)]['pct_of_all_parents_spanning'],
                 res[str(d)]['redundancy_factor'] or 0))
    out['phash'] = res

    # --- (5) 파일명 계보 ---
    print('\n=== (5) 파일명: 증강 접미사 분포 (업로드 후 분할 증거) ===')
    lin = {}
    for sp in ('train', 'valid', 'test'):
        if sp not in splits:
            continue
        _, fs = splits[sp]
        def stem(n):
            s = re.sub(r'\.rf\.[0-9a-f]{32}.*$', '', n)
            s = re.sub(r'\.(jpg|jpeg|png|bmp|webp)$', '', s, flags=re.I)
            return re.sub(r'_(jpg|jpeg|png|bmp)$', '', s, flags=re.I)
        aug = [f for f in fs if AUG_RE.search(stem(f.name))]
        lin[sp] = {'files': len(fs), 'augmented_suffix': len(aug),
                   'pct': round(100 * len(aug) / len(fs), 1)}
        print('  %-6s %5d files / 증강 접미사 %4d (%.1f%%)' % (sp, len(fs), len(aug), lin[sp]['pct']))
    out['filename_lineage'] = lin

    op = HERE / 'prior_dataset_audit.json'
    json.dump(out, open(op, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n저장:', op)
    print('\n[비교 기준] 본 연구 final0: 부모 686개 중 391개(57%)가 공식 분할을 걸침, 중복배수 7.2')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
