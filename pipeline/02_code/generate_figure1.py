import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Shadow
from matplotlib.lines import Line2D

FIG_DIR = _paths.RERUN_FIGS
OUT_PNG = os.path.join(FIG_DIR, "Figure1_FlowDiagram.png")
OUT_PDF = os.path.join(FIG_DIR, "Figure1_FlowDiagram.pdf")

FIG_W, FIG_H = 13.2, 11.0
DPI = 300

X_MID = 34
X_LEFT = 18
X_RIGHT = 58
X_EXCL = 86

MAIN_W = 50
EXCL_W = 26

Y_TOP = 132
Y_MID = 96
Y_FINAL = 60
Y_SPLIT_LABEL = 38
Y_DEVTEST = 16

PALETTE = {
    "cohort":     {"header": "#3B5BDB", "body": "#FFFFFF", "border": "#1E3A8A"},
    "processing": {"header": "#0CA678", "body": "#FFFFFF", "border": "#087F5B"},
    "final":      {"header": "#10B981", "body": "#FFFFFF", "border": "#047857"},
    "dev":        {"header": "#4263EB", "body": "#FFFFFF", "border": "#1E3A8A"},
    "test":       {"header": "#F08C00", "body": "#FFFFFF", "border": "#B45309"},
    "excluded":   {"header": "#6B7280", "body": "#FEF2F2", "border": "#9CA3AF"},
}

EDGE_MAIN = "#1F2937"
ARROW_COL = "#1F2937"
SHADOW_OFFSET = (0.45, -0.45)
SHADOW_ALPHA = 0.18
LW = 1.4

FS_TITLE = 11
FS_BODY = 9.2
FS_EXCL_TITLE = 9.2
FS_EXCL_BODY = 8.5
FONT = "DejaVu Sans"

HEADER_FRAC = 0.32

def draw_two_tone_box(ax, x_center, y_center, w, h, title, body_lines, palette,
                      title_color="white", body_color="#1F2937", lw=LW,
                      title_fs=FS_TITLE, body_fs=FS_BODY):
    x0 = x_center - w / 2
    y0 = y_center - h / 2

    shadow = FancyBboxPatch(
        (x0 + SHADOW_OFFSET[0], y0 + SHADOW_OFFSET[1]), w, h,
        boxstyle="round,pad=0.16,rounding_size=0.55",
        linewidth=0, edgecolor="none",
        facecolor="black", alpha=SHADOW_ALPHA, zorder=1,
    )
    ax.add_patch(shadow)

    outer = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.16,rounding_size=0.55",
        linewidth=lw, edgecolor=palette["border"],
        facecolor=palette["body"], zorder=2,
    )
    ax.add_patch(outer)

    header_h = h * HEADER_FRAC
    header = FancyBboxPatch(
        (x0, y0 + h - header_h), w, header_h,
        boxstyle="round,pad=0.16,rounding_size=0.55",
        linewidth=0, edgecolor="none",
        facecolor=palette["header"], zorder=3,
    )
    ax.add_patch(header)
    mask_h = header_h * 0.6
    mask = Rectangle(
        (x0, y0 + h - header_h), w, mask_h,
        linewidth=0, edgecolor="none",
        facecolor=palette["header"], zorder=3,
    )
    ax.add_patch(mask)

    ax.text(
        x_center, y_center + h / 2 - header_h / 2,
        title, ha="center", va="center",
        fontsize=title_fs, fontweight="bold", family=FONT,
        color=title_color, zorder=4,
    )

    body_top = y_center + h / 2 - header_h - 1.2
    body_text = "\n".join(body_lines)
    ax.text(
        x_center, body_top,
        body_text, ha="center", va="top",
        fontsize=body_fs, family=FONT, color=body_color,
        linespacing=1.4, zorder=4,
    )

def draw_arrow(ax, x1, y1, x2, y2, lw=1.6, color=ARROW_COL):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=15,
        linewidth=lw, color=color, zorder=1,
    ))

def draw_dashed_excl_connector(ax, y_center, x_from, x_to, color="#9CA3AF"):
    ax.add_patch(FancyArrowPatch(
        (x_from, y_center), (x_to, y_center),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.2, color=color,
        linestyle=(0, (4, 2)), zorder=1,
    ))

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 150)
ax.set_aspect("equal")
ax.axis("off")

TOP_H = 24
draw_two_tone_box(
    ax, X_MID, Y_TOP, MAIN_W, TOP_H,
    "KCYPS 2018 — pooled m1 + e4 panels",
    [
        "m1 (middle-school Grade-1) cohort baseline: n = 2,590",
        "e4 (elementary Grade-4) cohort baseline: n = 2,607",
        "Total enrolled: n = 5,197 adolescents",
        "Pooled and grade-harmonized to the G7–G10 window",
    ],
    PALETTE["cohort"],
)
draw_two_tone_box(
    ax, X_EXCL, Y_TOP, EXCL_W, 11,
    "Excluded",
    ["Records outside the",
     "G7–G10 grade window:",
     "n = 15,591"],
    PALETTE["excluded"],
    title_color="white",
    title_fs=FS_EXCL_TITLE,
    body_fs=FS_EXCL_BODY,
)
draw_dashed_excl_connector(
    ax, Y_TOP,
    x_from=X_MID + MAIN_W / 2,
    x_to=X_EXCL - EXCL_W / 2,
)

draw_arrow(ax, X_MID, Y_TOP - TOP_H / 2, X_MID, Y_MID + 13)

MID_H = 27
draw_two_tone_box(
    ax, X_MID, Y_MID, MAIN_W, MID_H,
    "Outcome-wave processing",
    [
        "Construction of t → t+1 transitions",
        "(G7→G8, G8→G9, G9→G10)",
        "SAPS classification at the outcome wave",
        "under National Information Society Agency",
        "2017 cut-offs (general / potential-risk / high-risk)",
    ],
    PALETTE["processing"],
)
draw_two_tone_box(
    ax, X_EXCL, Y_MID, EXCL_W, 16,
    "Excluded at outcome wave",
    ["• SAPS potential-risk at t+1:",
     "  n = 2,658 transitions",
     "• Missing outcome SAPS:",
     "  n = 1,961 transitions"],
    PALETTE["excluded"],
    title_color="white",
    title_fs=FS_EXCL_TITLE,
    body_fs=FS_EXCL_BODY,
)
draw_dashed_excl_connector(
    ax, Y_MID,
    x_from=X_MID + MAIN_W / 2,
    x_to=X_EXCL - EXCL_W / 2,
)

draw_arrow(ax, X_MID, Y_MID - MID_H / 2, X_MID, Y_FINAL + 12)

FINAL_H = 23
draw_two_tone_box(
    ax, X_MID, Y_FINAL, MAIN_W + 4, FINAL_H,
    "Final analytic file",
    [
        "Transitions: n = 10,972",
        "Unique adolescents: n = 4,634",
        "High-risk prevalence at t+1: 4.71%",
        "G7→G8: 3,778  |  G8→G9: 3,638  |  G9→G10: 3,556",
    ],
    PALETTE["final"],
)

draw_arrow(ax, X_MID, Y_FINAL - FINAL_H / 2, X_MID, Y_SPLIT_LABEL + 3)
ax.text(
    X_MID + 2.0, (Y_FINAL - FINAL_H / 2 + Y_SPLIT_LABEL) / 2 + 0.5,
    "GroupShuffleSplit by adolescent ID\n(random_state = 42, test_size = 0.20)",
    ha="left", va="center", fontsize=9, style="italic",
    family=FONT, color="#374151", zorder=3,
)

ax.add_line(Line2D(
    [X_MID, X_LEFT], [Y_SPLIT_LABEL, Y_SPLIT_LABEL],
    color=ARROW_COL, linewidth=1.4, zorder=1,
))
ax.add_line(Line2D(
    [X_MID, X_RIGHT], [Y_SPLIT_LABEL, Y_SPLIT_LABEL],
    color=ARROW_COL, linewidth=1.4, zorder=1,
))
DEVTEST_H = 26
draw_arrow(ax, X_LEFT, Y_SPLIT_LABEL, X_LEFT, Y_DEVTEST + DEVTEST_H / 2)
draw_arrow(ax, X_RIGHT, Y_SPLIT_LABEL, X_RIGHT, Y_DEVTEST + DEVTEST_H / 2)

draw_two_tone_box(
    ax, X_LEFT, Y_DEVTEST, 34, DEVTEST_H,
    "Development set",
    [
        "Transitions: n = 8,775",
        "Unique adolescents: n = 3,707",
        "High-risk at t+1: n = 424 (4.83%)",
        "General use at t+1: n = 8,351 (95.17%)",
        "G7→G8: 3,013  |  G8→G9: 2,905",
        "G9→G10: 2,857",
    ],
    PALETTE["dev"],
)
draw_two_tone_box(
    ax, X_RIGHT, Y_DEVTEST, 34, DEVTEST_H,
    "Held-out test set",
    [
        "Transitions: n = 2,197",
        "Unique adolescents: n = 927",
        "High-risk at t+1: n = 93 (4.23%)",
        "General use at t+1: n = 2,104 (95.77%)",
        "G7→G8: 765  |  G8→G9: 733",
        "G9→G10: 699",
    ],
    PALETTE["test"],
)

plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
print(f"[OK] {OUT_PNG}")
print(f"[OK] {OUT_PDF}")
