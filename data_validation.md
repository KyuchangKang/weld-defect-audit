# 데이터셋 검증 보고서 (2026-08-07)

> 대상: `data/final0` (Roboflow kaid-al, v1), `data/ikkj` (Roboflow ikkj, **v2**)
> 원시 분석 결과: `data/analysis_report.json`, `data/source_overlap_report.json`
> 감사 렌더링 샘플: `audit_samples/` (라벨 박스 오버레이 12장)

## 1. 기본 구조 (✅ 정상)

| | final0 | ikkj (v2) |
|---|---|---|
| train / valid / test | 4,027 / 621 / 310 = **4,958** | 6,335 / 601 / 300 = **7,236** |
| 클래스 (data.yaml) | Geometric defect, Non-fusion defect, crack, porosity, spatters (5) | Bad Welding, Crack, Good Welding, Porosity, Reinforcement, Spatters (6) |
| 라이선스 | CC BY 4.0 (yaml에 명기) | CC BY 4.0 (yaml에 명기) |
| 이미지-라벨 매칭 | 100% | 100% |
| 라벨 포맷 | bbox (5필드) 정상 | **폴리곤(세그먼트) 라인 다수 혼재** (112,059 라인) — Ultralytics가 학습 시 bbox로 자동 변환 가능하나 통합 시 명시적 변환 권장 |

## 2. 클래스 인스턴스 분포

**final0** (인스턴스 수): spatters 5,829 / porosity 2,732 / Geometric 1,844 / crack 1,453 / Non-fusion 982 — 균형 양호
**ikkj** (인스턴스 수): Spatters 1,458 / Reinforcement 1,305 / Bad 1,171 / Good 1,107 / Crack 900 / **Porosity 106 (이미지 41장뿐) ⚠️ 심각한 불균형**

## 3. 🚨 중대 발견: 소스 수준 중복 및 스플릿 누수(leakage)

MD5 완전 중복은 사실상 없음(final0 내부 2쌍). 그러나 Roboflow 파일명의 원본 스템 분석 결과:

| 항목 | final0 | ikkj (v2) |
|---|---|---|
| 고유 원본(소스) 수 | **1,853** (4,958장 중) | **848** (7,236장 중) |
| 증강 배수 | 약 2.7× | 약 8.5× (노이즈/그레이스케일/모자이크 증강이 이미지에 박혀 있음) |
| **스플릿 간 누수 소스** | **167개** | **472개** |

- **누수 의미**: 같은 원본의 증강 사본이 train과 valid/test에 동시에 존재 → 제공된 스플릿 그대로 쓰면 test 성능이 부풀려짐. **제공 스플릿 사용 불가, 소스 단위 재분할 필수**
- **데이터셋 간 공통 소스 203개** (final0 train의 2,707장이 ikkj와 소스 공유) → 결합 시 소스 단위 중복 제거 필수

## 4. 라벨 샘플 감사 (train 각 6장 무작위)

- **final0**: 6/6 타당 — spatter/crack/geometric 박스 위치·범위 합리적 (`audit_samples/final0_*.jpg`)
- **ikkj**: 문제 다수 —
  - `ikkj_1`: 명백한 crack에 "Spatters" 라벨 (오라벨)
  - `ikkj_3`: 용접부 전체 대비 과소한 미세 박스, "IRON" 워터마크 텍스트 포함
  - `ikkj_5`: 모자이크 증강(4분할 합성) + 강한 노이즈가 데이터에 박혀 있음
  - Porosity 사실상 부재(41장)

## 5. 결론 및 권고

1. **final0 = 주력 데이터셋으로 적합** (라벨 품질 양호). 단, 소스 단위 재분할 필요
2. **ikkj v2는 현재 형태로는 결합 부적합**: 증강 박힘 + 오라벨 + 폴리곤 혼재 + Porosity 부재
   - **권고**: Roboflow에서 ikkj **v1(원본, 증강 전)** 존재 여부 확인 후 재다운로드 → 원본만으로 재평가
   - v1이 없거나 여전히 품질 낮으면: ikkj는 **교차 데이터셋 일반화 평가(외부 테스트셋)** 용도로 축소 사용
3. 결합 시 절차(확정): 소스 스템 그룹핑 → 데이터셋 간 공통 소스 제거 → 클래스 매핑 통합 → **그룹 단위(소스 단위) 70/20/10 재분할** → 증강은 학습 시점(on-the-fly)에만 적용
4. **논문 관점**: 이 발견 자체가 유의미 — 선행 논문이 지적한 "공개 데이터셋 품질 문제"를 정량 확인(누수 소스 167/472개, 오라벨 사례). 방법론 절에서 '데이터 정제 절차'로 기술하면 검증력 있는 기여가 됨 (라벨 감사 최소 방침 유지하되, 누수 제거는 타당성의 필수 요건)

## 5-1. 추가 검증 (2026-08-07, 브라우저로 버전 페이지 직접 확인)

| 후보 | 버전 확인 결과 | 판정 |
|---|---|---|
| ikkj v1 | v2와 동일: Grayscale + Adaptive Equalization + Mosaic 박힘 (7,243장) | ❌ 제외 (양 버전 모두) |
| Delvin v1 (909 원본) | **224×224 Stretch** + Adaptive Equalization + 증강 박힘 (2,180장) | ❌ 제외 (해상도 파괴) |
| weld-qwzyj v7 (560 원본) | **Auto-Orient만 적용** — 원본 해상도·컬러 보존. 증강(flip/rotate/crop 등) 3배 박힘 (1,462장) | ✅ 보조 후보로 사용 가능 |
| weld-qwzyj v1 | 256×256 리사이즈 | ❌ (v7 사용) |

**확정 권고 전략**:
1. **주력**: final0 정제본 (소스 그룹핑 → 중복·누수 제거 → 소스 단위 재분할) — 5개 결함 유형 본 실험
2. **외부 일반화 평가**: weld-qwzyj v7 정제본 (공통 3클래스 crack/porosity/spatter만) — RQ3 담당
3. **ikkj 제외**: 파괴적 전처리(회색조·평활화·모자이크)와 오라벨 → 논문의 데이터 감사 절에서 제외 사유로 기술 (공개 데이터셋 품질 문제의 정량 사례)

## 5-2. weld-qwzyj v7 실측 검증 (2026-08-07) — ❌ 최종 제외

다운로드 후 픽셀·파일명 검사 결과 (`data/qwzyj_report.json`):
- 실제 이미지 크기 240~256px (버전 페이지 표기와 달리 원본 자체가 저해상도)
- 샘플 40장 중 22장 사실상 흑백 (평균 채도 18.6/255)
- 파일명 `denoised_RRT-..._fake_png` → **X-ray 방사선 영상의 denoise/합성(GAN 추정) 패치**
- 결론: RGB 현장 사진(final0)과 도메인 상이 → 외부 평가용으로도 부적합. **final0 단독 진행 확정**

## 5-3. final0 정제본(final0_clean) 생성 (2026-08-07) — ✅ 완료

파일명 분석에서 **업로드 전 증강**(`_270_flip` 등 회전/플립, `_block_r_c` 타일 분할)이 추가 발견되어
2단계 그룹핑 적용: content_id(회전/플립 제거) → parent_id(타일 → 부모 사진).

- 원자료 4,958파일 = **고유 콘텐츠 708개** = **부모 사진 686장** (실질 데이터 규모)
- 부모 사진 단위 층화(stratified) 70/20/10 분할, seed 42 → **누수 0건 검증 완료**
- train 3,567장(증강 사본 유지) / valid 142장 / test 71장(회전·플립 중복 제거, 콘텐츠당 1장)
  - test 71장은 선행 논문의 test 74장과 유사 규모
- 5개 클래스가 세 스플릿 모두에 존재 (test 인스턴스: Geometric 58, Non-fusion 22, crack 22, porosity 12, spatters 40)
- 산출물: `data/final0_clean/` (249MB, 짧은 파일명으로 리네이밍, data.yaml 포함)
  - 추적성: `manifest.csv` (새 ID ↔ 원본 파일명 ↔ content/parent ID)
  - 통계: `data/final0_clean_summary.json`
- 한계(논문에 명시): porosity가 소수 부모 사진에 집중되어 평가셋 내 이미지 수 적음 (valid 9, test 5장)

## 6. 기술 메모 (재현용)
- Windows 경로 260자 제한으로 긴 파일명 124건 추출 실패 → `\\?\` 확장 경로로 복구 완료. 통합 데이터셋 생성 시 파일명을 짧게 리네이밍할 것
- 압축 해제 시 Dropbox 동기화와 충돌 가능 → 파일별 재시도 로직 사용
