# weld-defect-audit

Audit, cleaning and leakage-quantification materials for

> K. Kang, *Type-Specific Weld Defect Detection on an Audited Public Dataset: A Leakage-Aware
> Feasibility Study Using YOLO*, Asia-pacific Journal of Convergent Research Interchange (submitted).

## What is here

| Path | Contents |
|---|---|
| `scripts/` | Audit, re-splitting, training, evaluation and analysis scripts of the original submission (31 files), plus `weldpaths.py`, which sets the folders they read (see "Reproducing the audit") |
| `requirements.txt` | Pinned software environment used for every reported run |
| `configs/runs_config.csv` | Configuration of all 34 training runs, extracted from each run's own `args.yaml` (model, data, epochs, batch, image size, learning rate, seed, workers, augmentation settings) |
| `scripts/revision1/`, `results/revision1/` | First-revision analyses: grouping-rule sensitivity, leaking-parent recount (407 of 686), per-split class and parent distribution, the reproducible OpenAlex literature search |
| `scripts/revision2/`, `results/revision2/` | Second-revision analyses: budget-matched comparison of retained copies with online 90° rotation and vertical flip (five runs), the script that builds `configs/runs_config.csv`, and the blind visual audit of the grouping (`t3_*.py`, `results/revision2/T3/`; see "Grouping audit") |
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
3. `scripts/revision1/leak_recount.py` — content and parent grouping and split-leakage counts (708 content groups,
   686 parents, 407 leaking); run from the repository root
4. `scripts/phash_validate.py` — perceptual-hash cross-validation of the grouping (thresholds 0/4/8)
5. `scripts/build_clean2.py` — parent-level stratified re-split
6. `scripts/train_runs.py`, `scripts/eval_gate_d.py` — training and the sealed test evaluation

`scripts/parent_stats.py` is a preliminary grouping pass from before the rules of Section 3.1 were fixed (it
yields 950 parents). It is kept for the record; no reported number comes from it.

The scripts keep the original workstation's paths as defaults, so with nothing set they run exactly as
they did for the paper. To run them elsewhere, point them at your copies on the command line or in the
environment (the command line wins); `scripts/weldpaths.py` resolves both:

| Option | Environment variable | Folder |
|---|---|---|
| `--root PATH` | `WELD_AUDIT_ROOT` | project folder holding `04_experiments/data/` (`final0/`, `final0_clean/`) and `05_results/` |
| `--runs PATH` | `WELD_AUDIT_RUNS` | training-run folder holding the run directories (`E2_v5n/`, `upgrade/`, `review4/`, ...) |

For example, from the repository root:

```
python scripts/revision1/leak_recount.py --root /path/to/project
```

prints the 708 content groups, 686 parents and 407 leaking parents. Only the path lines differ from the
scripts as run; every rewritten path was checked to resolve to its original value when no option is given.
The one exception is the grouping-audit scripts (`scripts/revision2/t3_*.py`), which read and write
`results/revision2/T3/` in the repository instead of the working folder they ran in; run in this layout they
regenerate every file there byte for byte.

## Grouping audit (second revision)

The grouping of Section 3.1 was checked by a blind visual audit. Candidate pairs were drawn from the perceptual-hash
distances of all 4,958 images (`phash_final0.npy`), rendered side by side with filenames, splits and distances hidden,
and shuffled together with positive and negative control pairs.

| Step | Script | Output in `results/revision2/T3/` |
|---|---|---|
| Sampling frames | `t3_population.py` | `population.json`, `pairs.npz` |
| Sheets and hidden key | `t3_make_sheets.py` (needs the source images) | `key.json`; the sheets themselves are regenerated, not shipped |
| First reading (machine-assisted) | — | `judge_claude.jsonl` (416 pairs), `judge_claude_groups.jsonl` (60 groups) |
| Tally per stratum | `t3_tally.py` | `tally_claude.json` |
| Independent-photograph estimate | `t3_photo_estimate.py` | `photo_estimate.json` |
| Tile rule as specified | `t3_tile_fix.py` | `tile_fix.json` (see "Known issue") |
| Test re-scoring without shared photographs | `t3_rescore.py` (needs the gate-D weights) | `rescore.json` |
| Second reading and agreement | `t3_agreement.py` | `agreement.json` |

The second reader (the author) read a stratified subsample of 60 pairs and 5 groups without seeing the first reader's
judgments. Both readers applied the written decision rules in `판독규칙_v3_저자확정.md` (in Korean; the author's
rules). Because those rules quote 12 sheet pairs as worked examples, the machine-assisted reader re-read the
subsample under them before the author's sheet was opened (`판독표_Claude_v3.csv`), and the agreement figures exclude
the 12 examples. On the remaining 48 pairs the two readers agreed on all 33 that both decided, same scene versus
different scene (Cohen's κ = 1.00); in the other 15, at least one reader left the pair undecided. The author's
sheet is `판독표_저자.csv`.

Excluding the 31 clean test images that share a photograph or scene with the training split raises overall mAP50 from
0.576 to 0.617 with the frozen gate-D weights; the 31 images alone score 0.597.

## Known issue: tile suffix in `build_clean2.py`

Section 3.1 specifies that a parent ID strips the tile suffix `_block_<r>_<c>`. The content rule in
`scripts/build_clean2.py` (and the identical rule in `phash_validate.py`) first removes trailing rotation suffixes
`_0`, repeatedly, and in doing so also removes the final `0` of a tile suffix: `img_3629_block_0_0` becomes
`img_3629_block`, and `img_3629_block_2_0` becomes `img_3629_block_2`. Tiles of one photograph can therefore receive
different parent IDs.

The released manifest and split are the ones used in the paper and are not changed. `scripts/revision2/t3_tile_fix.py`
applies the rule as specified: 667 parents instead of 686, and 395 leaking (59.2%) instead of 407 (59.3%). In
`final0_clean`, five parents then span two splits, affecting eight validation and two test images (`test_00057`,
`test_00059`); `tile_fix.json` lists them. The paper reports this in Section 3.1 and the footnote to [Table 1].

## Software and configuration

All runs used Python 3.14.6, PyTorch 2.11.0 (CUDA 12.8 build), Ultralytics 8.4.118 and ImageHash 4.3.2 on one
NVIDIA RTX PRO 5000 Blackwell (driver 595.97) under Windows 11; `requirements.txt` pins the rest. Do not install
`albumentations`: when it is present, Ultralytics adds blur, median blur, grayscale and CLAHE (each with probability 0.01) to the default training
pipeline, which changes the training conditions reported in the paper.

Every run is fine-tuned from COCO weights with the Ultralytics defaults except where `configs/runs_config.csv` shows
otherwise: 100 epochs (720 for the 495-image budget-matched runs), batch 16, 640 px, lr0 0.01, `cache=False`,
`deterministic=True`, training seeds 42 to 44, split seeds 101 to 104 for the robustness re-splits. The number of
data-loader workers was not held fixed: it was 8 in 20 runs and 4 in 14, including within one model's seed
repetitions, and the table records it per run. The 2nd-revision rotation is applied by `scripts/revision2/rot90_patch.py`, which wraps the dataset
loader so that each image is rotated by a uniformly drawn multiple of 90° before mosaic; vertical flip uses the
standard `flipud=0.5` setting.

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
