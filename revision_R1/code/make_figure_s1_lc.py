import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revnumbers as NB

C8 = NB.load_comment8()
LC = C8["lc"]
OUT = os.path.join(NB.RES, "FigureS1_LearningCurves.png")

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
STY = {"455 item-level columns, tuned XGBoost": ("#1f4e79", "o",
                                                 "Primary model, 455 item-level columns"),
       "42 scale-score columns, neutral XGBoost": ("#c55a11", "s",
                                                   "Parsimonious model, 42 scale scores")}
for rep, (col, mk, lab) in STY.items():
    g = LC[LC.representation == rep].sort_values("n_pos")
    for ax, m in zip(axes, ("AUROC", "PR_AUC")):
        ax.plot(g.n_pos, g[m], marker=mk, color=col, label=lab, lw=1.6, ms=5)
        if m == "AUROC":
            ax.fill_between(g.n_pos, g.AUROC_min, g.AUROC_max, color=col, alpha=0.15,
                            lw=0)
for ax, lab, tag in zip(axes, ("Held-out AUROC", "Held-out PR-AUC"), ("A", "B")):
    ax.set_xlabel("High-risk transitions in the training subsample")
    ax.set_ylabel(lab)
    ax.set_title(tag, loc="left", fontweight="bold")
    ax.grid(alpha=0.3)
    top = ax.secondary_xaxis("top", functions=(lambda e: e / 424 * 8775,
                                               lambda n: n / 8775 * 424))
    top.set_xlabel("Training transitions")
axes[1].axhline(93 / 2197, color="grey", ls="--", lw=1)
axes[1].text(430, 93 / 2197 + 0.004, "prevalence", ha="right", color="grey", fontsize=8)
axes[0].legend(loc="lower right", fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print(f"written: {OUT}")
