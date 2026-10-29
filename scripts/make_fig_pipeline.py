"""Fig. 1 (감사·재분할 파이프라인) 재제작 — 2026-08-17.

기존 fig1_audit_pipeline.png의 문제: boxstyle의 pad(0.02)가 데이터 좌표로 더해져
박스가 서로 맞닿고 화살표가 가려짐 + 마지막 박스가 축 밖으로 나감.

재설계: 2행 x 3열 사행(serpentine) 배치, 박스/화살표 간격을 명시적으로 계산해
겹침이 발생할 수 없도록 함. 출력 = 06_manuscript/figures/fig1_audit_pipeline.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from weldpaths import project_dir  # noqa: E402  (path defaults; see weldpaths.py)
BASE = project_dir(r"D:\Sync\Dropbox\paper_wedling2")
FIG = BASE / "06_manuscript" / "figures"

BLUE_F, BLUE_E = "#eef3fb", "#2b5aa0"
RED_F, RED_E = "#fdeaea", "#a03030"
AMB_F, AMB_E = "#fdf3e0", "#b07818"
GRN_F, GRN_E = "#e8f5e9", "#2e7d32"

# 격자 (축 좌표 0~1) — 겹침이 불가능하도록 폭/간격을 먼저 고정
W, H = 0.285, 0.30          # 박스 크기
GX = 0.0725                 # 열 간격 (화살표 자리)
GY = 0.20                   # 행 간격 (하강 화살표 자리)
X = [i * (W + GX) for i in range(3)]        # 0, 0.3575, 0.715  -> 오른쪽 끝 1.0
Y1 = 0.60                                   # 1행 하단
Y0 = Y1 - H - GY                            # 2행 하단

STAGES = [
    # (열, 행, 텍스트, 색, 굵게)
    (0, 1, "Distribution as published\n4,958 files, official split\n(train / valid / test)", RED_F, RED_E, False),
    (1, 1, "Steps 1-2\nMD5 duplicate detection +\nfilename-lineage analysis", BLUE_F, BLUE_E, False),
    (2, 1, "Step 3  grouping\n708 content groups\n686 parent photographs", BLUE_F, BLUE_E, False),
    (2, 0, "Step 4  leakage check\n391 / 686 parents (57%)\nspan the official split", AMB_F, AMB_E, False),
    (1, 0, "Parent-level, class-stratified\nre-split (seed 42);\ntrain retains all copies", BLUE_F, BLUE_E, False),
    (0, 0, "final0_clean\n3,567 / 142 / 71 files\n0 file-lineage leaks", GRN_F, GRN_E, True),
]


def main():
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.axis("off")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.06, 0.97)

    def box(col, row, text, fc, ec, bold):
        x, y = X[col], (Y1 if row else Y0)
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0",
                                    fc=fc, ec=ec, lw=1.4, mutation_aspect=2.2))
        ax.text(x + W / 2, y + H / 2, text, ha="center", va="center",
                fontsize=9.8, fontweight="bold" if bold else "normal", linespacing=1.55)

    def arrow(p1, p2, rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                                     lw=1.4, color="#333333",
                                     connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=0, shrinkB=0))

    for col, row, text, fc, ec, bold in STAGES:
        box(col, row, text, fc, ec, bold)

    m = 0.012                       # 박스 테두리와 화살표 사이 여백
    yc1, yc0 = Y1 + H / 2, Y0 + H / 2
    # 1행: 좌 -> 우
    for c in (0, 1):
        arrow((X[c] + W + m, yc1), (X[c + 1] - m, yc1))
    # 열 전환: 3열에서 아래로
    arrow((X[2] + W / 2, Y1 - m), (X[2] + W / 2, Y0 + H + m))
    # 2행: 우 -> 좌
    for c in (2, 1):
        arrow((X[c] - m, yc0), (X[c - 1] + W + m, yc0))

    ax.text(0.5, Y0 - 0.115,
            "Cross-checks: all 71 test images label-audited (documented +4 porosity correction; "
            "14 out-of-domain images catalogued);\ngrouping cross-validated by rotation/flip-invariant "
            "perceptual hashing (228 adjacent-shot pairs still cross splits, bounded at 0.007 mAP50)",
            ha="center", va="top", fontsize=9.2, style="italic", linespacing=1.6)

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig1_audit_pipeline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
