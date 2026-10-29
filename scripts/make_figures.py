"""논문 그림 제작 (게이트 3).

Fig1: 감사·재분할 파이프라인 다이어그램 (신규 제작)
Fig4: 예측 시각화 몽타주 2x2 — (a) 현장 다중클래스 (b) 현장 spatters
      (c) 웹 콜라주 porosity 부분 검출 (d) SEM 도메인 밖
Fig2(학습곡선)·Fig3(혼동행렬)은 기존 산출물 복사.
출력: 06_manuscript/figures/
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from weldpaths import project_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
FIG = BASE / "06_manuscript" / "figures"
PRED = BASE / "05_results" / "test_eval" / "pred_vis_s42"


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#2b5aa0", fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.2, color="#333"))

    y, h = 0.42, 0.44
    box(0.005, y, 0.125, h, "Distribution\n4,958 files\n(official split)", fc="#fdeaea", ec="#a03030")
    box(0.165, y, 0.125, h, "Step 1–2\nMD5 dedup +\nfilename lineage")
    box(0.325, y, 0.125, h, "Step 3 grouping\n708 contents\n686 parents")
    box(0.485, y, 0.125, h, "Step 4 leakage\n391/686 parents\n(57%) leak", fc="#fdf3e0", ec="#b07818")
    box(0.645, y, 0.125, h, "Parent-level\nstratified\nre-split (seed 42)")
    box(0.805, y, 0.19, h,
        "final0_clean\n3,567 / 142 / 71\n0 leaks + manifest", fc="#e8f5e9", ec="#2e7d32", bold=True)
    for x1, x2 in [(0.13, 0.165), (0.29, 0.325), (0.45, 0.485), (0.61, 0.645), (0.77, 0.805)]:
        arrow(x1, y + h / 2, x2, y + h / 2)
    ax.text(0.5, 0.13, "Label audit: all 71 test images inspected; documented correction (+4 porosity); "
                       "14 out-of-domain images catalogued; grouping cross-validated by perceptual hashing",
            ha="center", fontsize=8.8, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0.02, 0.98)
    fig.savefig(FIG / "fig1_audit_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_montage():
    panels = [
        ("test_00006.jpg", "(a) field image: Geometric + Non-fusion"),
        ("test_00055.jpg", "(b) field image: spatters"),
        ("test_00050.jpg", "(c) web collage: partial porosity detection"),
        ("test_00049.jpg", "(d) out-of-domain: SEM micrograph"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.6))
    for ax, (fname, title) in zip(axes.flat, panels):
        ax.imshow(Image.open(PRED / fname))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_predictions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_pipeline()
    shutil.copy2(BASE / "05_results" / "runs" / "E2_v5n" / "results.png",
                 FIG / "fig2_training_curves.png")
    shutil.copy2(BASE / "05_results" / "test_eval" / "E2_v5n_test" / "confusion_matrix_normalized.png",
                 FIG / "fig3_confusion_matrix.png")
    fig4_montage()
    for f in sorted(FIG.iterdir()):
        print(f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
