"""원본 배포 아카이브의 파일 단위 해시 매니페스트 생성.

목적 (심사 대응):
  논문 [22]는 아카이브 **전체** MD5(76fcb391b1c7a5ec1b5ef88830ba5ff2)만 제시한다.
  Roboflow 배포본은 갱신·삭제될 수 있으므로, 원본이 사라지면 "재실행하면 708 content
  group / 686 parent photograph가 재현된다"는 주장을 제3자가 검증할 수 없다.
  파일 단위 해시 목록이 있으면 **이미지 자체를 재배포하지 않고도**(따라서 CC BY 4.0
  재배포 문제 없이) 손에 든 사본이 우리가 감사한 것과 동일한지 확인할 수 있다.

출력 (07_submission/github_release/manifest/):
  source_archive_manifest.csv   path,bytes,md5   (아카이브 내 전 엔트리)
  source_archive_manifest.json  아카이브 MD5·엔트리 통계·검증 방법

검증 방법(제3자):
  같은 아카이브를 내려받아 이 스크립트를 --verify 로 실행하면 경로별 MD5를 대조한다.

사용:
  python make_source_manifest.py            # 생성
  python make_source_manifest.py --verify   # 기존 매니페스트와 대조만
"""
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter

BASE = r'D:\Sync\Dropbox\paper_wedling2'
ARCHIVE = os.path.join(BASE, '04_experiments', 'data',
                       'welding defect final0.v1i.yolov8.zip')
OUTDIR = os.path.join(BASE, '07_submission', 'github_release', 'manifest')
CSV_PATH = os.path.join(OUTDIR, 'source_archive_manifest.csv')
JSON_PATH = os.path.join(OUTDIR, 'source_archive_manifest.json')

ARCHIVE_MD5_PAPER = '76fcb391b1c7a5ec1b5ef88830ba5ff2'


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def scan(archive):
    """아카이브 내 전 엔트리의 (path, bytes, md5). 디렉터리 엔트리는 제외."""
    rows = []
    with zipfile.ZipFile(archive) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue
            h = hashlib.md5()
            with z.open(info) as f:
                for b in iter(lambda: f.read(1 << 20), b''):
                    h.update(b)
            rows.append((info.filename, info.file_size, h.hexdigest()))
    return rows


def summarize(rows):
    ext = Counter(r[0].rsplit('.', 1)[-1].lower() if '.' in r[0] else '(none)'
                  for r in rows)
    split = Counter(r[0].split('/')[0] for r in rows if '/' in r[0])
    dup = Counter(r[2] for r in rows)
    return {
        'files': len(rows),
        'total_bytes': sum(r[1] for r in rows),
        'by_extension': dict(sorted(ext.items())),
        'by_top_level_dir': dict(sorted(split.items())),
        'distinct_md5': len(dup),
        'files_sharing_an_md5': sum(c for c in dup.values() if c > 1),
    }


def write(rows, archive_md5):
    os.makedirs(OUTDIR, exist_ok=True)
    with io.open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'bytes', 'md5'])
        w.writerows(rows)

    meta = {
        'archive_filename': os.path.basename(ARCHIVE),
        'archive_md5': archive_md5,
        'archive_md5_matches_paper_reference': archive_md5 == ARCHIVE_MD5_PAPER,
        'source': ('Roboflow Universe, kaid-al "welding defect final0" v1 '
                   '(YOLOv8 export), CC BY 4.0; reference [22] of the paper'),
        'contents': summarize(rows),
        'purpose': ('Per-file integrity reference for the source distribution. '
                    'Image and label files themselves are not redistributed here; '
                    'this manifest lets a reader confirm that a copy obtained from '
                    'the original source is byte-identical to the one audited.'),
        'how_to_verify': ('Download the archive from the URL in reference [22], then run '
                          'scripts/make_source_manifest.py --verify. It recomputes every '
                          'per-file MD5 and reports any path that is missing, extra, or '
                          'different.'),
        'related': {
            'cleaned_split_manifest': 'manifest/final0_clean_manifest.csv',
            'audit_scripts': 'scripts/',
        },
    }
    with io.open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


def verify(rows):
    if not os.path.exists(CSV_PATH):
        print('매니페스트가 없다:', CSV_PATH)
        return 1
    with io.open(CSV_PATH, encoding='utf-8', newline='') as f:
        ref = {r['path']: (int(r['bytes']), r['md5']) for r in csv.DictReader(f)}
    cur = {p: (b, m) for p, b, m in rows}
    missing = sorted(set(ref) - set(cur))
    extra = sorted(set(cur) - set(ref))
    diff = sorted(p for p in set(ref) & set(cur) if ref[p] != cur[p])
    print('매니페스트 %d개 / 현재 %d개' % (len(ref), len(cur)))
    print('  누락 %d · 추가 %d · 불일치 %d' % (len(missing), len(extra), len(diff)))
    for label, items in (('누락', missing), ('추가', extra), ('불일치', diff)):
        for p in items[:10]:
            print('    [%s] %s' % (label, p))
        if len(items) > 10:
            print('    ... 외 %d개' % (len(items) - 10))
    ok = not (missing or extra or diff)
    print('결과:', 'OK — 전 파일 일치' if ok else 'MISMATCH')
    return 0 if ok else 1


def main():
    if not os.path.exists(ARCHIVE):
        print('아카이브를 찾을 수 없다:', ARCHIVE)
        return 1
    print('아카이브 :', os.path.basename(ARCHIVE))
    amd5 = file_md5(ARCHIVE)
    print('전체 MD5 :', amd5,
          '(논문 [22] 기재값과 일치)' if amd5 == ARCHIVE_MD5_PAPER else '★ 논문 기재값과 불일치')
    print('엔트리 해시 계산 중 ...')
    rows = scan(ARCHIVE)

    if '--verify' in sys.argv:
        return verify(rows)

    meta = write(rows, amd5)
    c = meta['contents']
    print('\n파일 %d개 / %.1f MB' % (c['files'], c['total_bytes'] / 1e6))
    print('  확장자      :', c['by_extension'])
    print('  최상위 폴더 :', c['by_top_level_dir'])
    print('  고유 MD5    : %d (동일 해시를 공유하는 파일 %d개)'
          % (c['distinct_md5'], c['files_sharing_an_md5']))
    print('\n생성:', CSV_PATH)
    print('생성:', JSON_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
