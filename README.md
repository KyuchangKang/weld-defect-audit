# weld-defect-audit

Audit, cleaning and leakage-quantification materials for

> K. Kang, *Type-Specific Weld Defect Detection on an Audited Public Dataset: A Leakage-Aware
> Feasibility Study Using YOLO*, Asia-pacific Journal of Convergent Research Interchange (submitted).

## What is here

| Path | Contents |
|---|---|
| `scripts/` | Audit, re-splitting, training, evaluation and analysis scripts (31 files) |
| `manifest/final0_clean_manifest.csv` | Per-file record: new ID, split, original split, original filename, stem, content ID, parent ID |
| `manifest/source_archive_manifest.csv` | Per-file integrity reference for the **source** distribution: path, size and MD5 for all 9,919 archive entries |
| `manifest/source_archive_manifest.json` | Archive MD5, entry statistics and the verification procedure for the above |
| `manifest/*.json` | Integrity-check outputs for the cleaned split and the screening audits |
| `scripts/review5/` | Follow-up analyses added in revision: leakage-mechanism evidence, audit of the preceding study's dataset, cross-dataset overlap, the bisection's domain/class composition control, and the post-hoc medium-model test evaluation |
| `results/review5/` | JSON outputs of the above (measured values quoted in Sections 3.1, 4.1, 4.3 and 4.4) |
| `label_fixes/` | Original labels, corrected labels, evidence renderings and changelog for the documented test-label correction |
| `results/` | Result files: gate-D test evaluation, five parent-level re-splits, controlled splits, perceptual-hash validation, bootstrap uncertainty, re-analyses |

## What is not here

The image files themselves are not redistributed. The source distribution is

* Roboflow Universe, user `kaid-al`, dataset **welding defect final0** (v1, YOLOv8 export), CC BY 4.0
* archive MD5 `76fcb391b1c7a5ec1b5ef88830ba5ff2`

Download that archive and run `scripts/build_clean2.py` with the manifest to reconstruct the cleaned
split (3,567 / 142 / 71 files) exactly as used in the paper.

Because a community distribution can be updated or withdrawn, `manifest/source_archive_manifest.csv`
records the size and MD5 of every file in the archive we audited. To confirm that a copy obtained
later is byte-identical to ours:

```
python scripts/make_source_manifest.py --verify
```

It recomputes every per-file hash and reports any path that is missing, extra or different. The
manifest carries hashes only, not image data, so it redistributes nothing.

For reference, the archive holds 4,958 images and 4,958 labels, split 4,027 / 621 / 310 by the
distributor. Only two image pairs are byte-identical: the 7.2-fold redundancy the paper reports is
carried by pre-generated rotations, flips and tiles, which MD5 comparison alone cannot detect.

## Reproducing the audit

1. `scripts/make_source_manifest.py --verify` — confirm the archive matches the one audited
2. `scripts/analyze_datasets.py` — MD5 duplicates and filename-lineage grouping
3. `scripts/parent_stats.py` — content and parent grouping, split-leakage counts
4. `scripts/phash_validate.py` — perceptual-hash cross-validation of the grouping (thresholds 0/4/8)
5. `scripts/build_clean2.py` — parent-level stratified re-split
6. `scripts/train_runs.py`, `scripts/eval_gate_d.py` — training and the sealed test evaluation

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

## Scope of this release

This repository holds the artefacts the paper's data-availability statement names: the audit and
re-splitting scripts, the per-file provenance manifest, the original and corrected test labels with
their evidence renderings, and the re-split configurations. Internal working notes from the study
(screening reports, the frozen evaluation plan, the label-audit worksheet) are not included: they are
written in Korean, describe intermediate states rather than the reported protocol, and the method as
reported is specified in Sections 3.1 and 3.2 of the paper. The label audit itself ran as a
machine-assisted first pass over rendered overlays, with corrections and borderline decisions taken
by the author; `label_fixes/label_fixes_20260813/CHANGELOG.md` records what was changed and why.
