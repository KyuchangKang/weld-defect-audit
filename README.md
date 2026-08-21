# weld-defect-audit

Audit, cleaning and leakage-quantification materials for

> K. Kang, *Type-Specific Weld Defect Detection on an Audited Public Dataset: A Leakage-Aware
> Feasibility Study Using YOLO*, Asia-pacific Journal of Convergent Research Interchange (submitted).

## What is here

| Path | Contents |
|---|---|
| `scripts/` | Audit, re-splitting, training, evaluation and analysis scripts (30 files) |
| `manifest/final0_clean_manifest.csv` | Per-file record: new ID, split, original split, original filename, stem, content ID, parent ID |
| `manifest/*.json` | Integrity-check outputs for the cleaned split and the screening audits |
| `scripts/review5/` | Follow-up analyses added in revision: leakage-mechanism evidence, audit of the preceding study's dataset, cross-dataset overlap, the bisection's domain/class composition control, and the post-hoc medium-model test evaluation |
| `results/review5/` | JSON outputs of the above (measured values quoted in Sections 3.1, 4.1, 4.3 and 4.4) |
| `label_fixes/` | Original labels, corrected labels, evidence renderings and changelog for the documented test-label correction |
| `results/` | Result files: gate-D test evaluation, five parent-level re-splits, controlled splits, perceptual-hash validation, bootstrap uncertainty, re-analyses |
| `test71_audit.md` | Full audit report of the 71 test images |
| `data_validation.md` | Dataset screening and exclusion report |
| `evaluation_plan.md` | Pre-registered metrics, protocol and success criteria |

## What is not here

The image files themselves are not redistributed. The source distribution is

* Roboflow Universe, user `kaid-al`, dataset **welding defect final0** (v1, YOLOv8 export), CC BY 4.0
* archive MD5 `76fcb391b1c7a5ec1b5ef88830ba5ff2`

Download that archive and run `scripts/build_clean2.py` with the manifest to reconstruct the cleaned
split (3,567 / 142 / 71 files) exactly as used in the paper.

## Reproducing the audit

1. `scripts/analyze_datasets.py` — MD5 duplicates and filename-lineage grouping
2. `scripts/parent_stats.py` — content and parent grouping, split-leakage counts
3. `scripts/phash_validate.py` — perceptual-hash cross-validation of the grouping (thresholds 0/4/8)
4. `scripts/build_clean2.py` — parent-level stratified re-split
5. `scripts/train_runs.py`, `scripts/eval_gate_d.py` — training and the sealed test evaluation

Paths in the scripts are absolute and reflect the original workstation; adjust `BASE` at the top of
each file before running.

## Licence

Code: MIT. Derived split definitions and manifests: CC BY 4.0, inherited from the source dataset.

## Regenerating the visual audit sheets

The field / out-of-domain judgement of the 230 original-test parent photographs, and the
30-image unseen-parent sheet, are not shipped as images. They are regenerated deterministically
from the source archive by `scripts/review5/make_parent_sheets.py` and
`scripts/review5/make_unseen_sheet.py`; the judgement itself is recorded as parent indices in
`scripts/review5/m2_domain_control.py`.
