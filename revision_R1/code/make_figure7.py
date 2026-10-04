import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revnumbers as NB

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
FIGS = os.path.join(REV, "figures")
os.makedirs(FIGS, exist_ok=True)

A = pd.read_csv(os.path.join(RES, "grouped_shap_audit.csv"))
A = A.dropna(subset=["summed_shap", "perm_importance"])
TOP = A.sort_values("perm_importance", ascending=False).head(12).scale.tolist()
D = A[A.scale.isin(TOP)].copy()

SAPS = ["SAPS mean score", "SAPS sum score", "SAPS tolerance", "SAPS withdrawal",
        "SAPS virtual-life", "Daily smartphone use", "Baseline SAPS",
        "SAPS change from baseline", "SAPS slope", "Prior high-risk"]
DESIGN = ["Wave", "Cohort"]

def colour(name):
    if name in SAPS:
        return "#b03a2e"
    if name in DESIGN:
        return "#909497"
    return "#1f4e79"

PANELS = [("perm_importance", "A  Grouped permutation importance\n(AUROC drop, whole "
                              "scale permuted)", "AUROC drop"),
          ("summed_shap", "B  Summed mean |SHAP|\n(as in the submitted figures)",
           "Sum of mean |SHAP| across items"),
          ("per_item_shap", "C  Mean |SHAP| per item\n(summed value / number of items)",
           "Mean |SHAP| per item")]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4))
for ax, (col, title, xlab) in zip(axes, PANELS):
    d = D.sort_values(col, ascending=True)
    labels = [f"{NB.lab(s)}  ({int(n)})" for s, n in zip(d.scale, d.n_items)]
    bars = ax.barh(range(len(d)), d[col], color=[colour(s) for s in d.scale],
                   height=0.72)
    if col == "perm_importance":
        ax.errorbar(d[col], range(len(d)), xerr=d.perm_sd, fmt="none",
                    ecolor="#444444", elinewidth=1, capsize=2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

handles = [plt.Rectangle((0, 0), 1, 1, color="#1f4e79"),
           plt.Rectangle((0, 0), 1, 1, color="#b03a2e"),
           plt.Rectangle((0, 0), 1, 1, color="#909497")]
fig.legend(handles, ["Psychosocial, behavioral, family, or demographic",
                     "SAPS-derived", "Study design"],
           loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = os.path.join(FIGS, "Figure7_GroupedImportance_Methods.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"written: {out}")
print("scales shown (number of items in parentheses):")
for _, r in D.sort_values("perm_importance", ascending=False).iterrows():
    print(f"  {r.scale:<32} ({int(r.n_items):>2})  perm {r.perm_importance:+.4f}  "
          f"summed rank {int(r.rank_summed):>2}  per-item rank {int(r.rank_per_item):>2}")
