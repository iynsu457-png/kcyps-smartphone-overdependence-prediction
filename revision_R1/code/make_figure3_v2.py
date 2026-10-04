import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "pipeline", "02_code"))
import _paths

import os
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(_HERE), "figures")
OUT_PNG = os.path.join(FIG_DIR, "Figure3_PerTransition_v2.png")
OUT_PDF = os.path.join(FIG_DIR, "Figure3_PerTransition_v2.pdf")

df = pd.read_csv(_paths.TRANSITION_PERF)
TRANSITION_ORDER = ["G7->G8", "G8->G9", "G9->G10"]
TRANSITION_LABELS = ["G7→G8", "G8→G9", "G9→G10"]
MODEL_ORDER = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "LogRegEN"]

COLORS = {
    "XGBoost":      "#D62728",
    "LightGBM":     "#1F77B4",
    "CatBoost":     "#2CA02C",
    "RandomForest": "#FF7F0E",
    "ExtraTrees":   "#9467BD",
    "LogRegEN":     "#7F7F7F",
}
MARKERS = {
    "XGBoost":      "o",
    "LightGBM":     "s",
    "CatBoost":     "D",
    "RandomForest": "^",
    "ExtraTrees":   "v",
    "LogRegEN":     "X",
}
NICE_NAME = {
    "XGBoost":      "XGBoost",
    "LightGBM":     "LightGBM",
    "CatBoost":     "CatBoost",
    "RandomForest": "Random Forest",
    "ExtraTrees":   "Extra Trees",
    "LogRegEN":     "Elastic-net logistic\nregression",
}

fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
x_pos = list(range(len(TRANSITION_ORDER)))

for model in MODEL_ORDER:
    sub = df[df["Model"] == model].set_index("Transition").loc[TRANSITION_ORDER]
    auc = sub["AUC"].values
    lo = sub["AUC_lo"].values
    hi = sub["AUC_hi"].values
    color = COLORS[model]

    ax.fill_between(x_pos, lo, hi, color=color, alpha=0.10, linewidth=0)

    ax.plot(x_pos, auc, "-", color=color, linewidth=2.0, alpha=0.9, zorder=3)
    ax.plot(
        x_pos, auc,
        marker=MARKERS[model], markersize=8,
        markeredgecolor="white", markeredgewidth=1.0,
        markerfacecolor=color, linestyle="",
        label=NICE_NAME[model], zorder=4,
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(TRANSITION_LABELS, fontsize=11)
ax.set_xlabel("Grade transition", fontsize=12, labelpad=8)
ax.set_ylabel("AUROC (95% CI)", fontsize=12, labelpad=8)
ax.set_ylim(0.50, 0.97)
ax.set_yticks([0.50, 0.60, 0.70, 0.80, 0.90])
ax.tick_params(axis="both", which="major", labelsize=10, length=4, width=0.8)

for y in [0.60, 0.70, 0.80, 0.90]:
    ax.axhline(y, color="0.92", linewidth=0.6, zorder=0)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("0.4")
    ax.spines[spine].set_linewidth(0.8)

leg = ax.legend(
    loc="center left", bbox_to_anchor=(1.01, 0.5),
    frameon=False, fontsize=10, handletextpad=0.4,
    labelspacing=0.7, borderaxespad=0,
)

plt.tight_layout()
fig.savefig(OUT_PNG, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
print(f"[OK] {OUT_PNG}")
print(f"[OK] {OUT_PDF}")
