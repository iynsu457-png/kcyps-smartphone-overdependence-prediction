import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

LABELS = {
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
    "RandomForest": "Random Forest",
    "ExtraTrees": "Extra Trees",
    "LogRegEN": "Elastic-net Logistic Regression",
}

df = pd.read_csv(_paths.TEST_PREDICTIONS)
y = df["y_true"].values

rows = [(c, roc_auc_score(y, df[c].values)) for c in LABELS]
rows.sort(key=lambda r: -r[1])

fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
for clf, auc in rows:
    fpr, tpr, _ = roc_curve(y, df[clf].values)
    ax.plot(fpr, tpr, lw=1.8, label=f"{LABELS[clf]} (AUROC = {auc:.3f})")
ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="0.5", label="Chance")

ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("1 - Specificity (false-positive rate)")
ax.set_ylabel("Sensitivity (true-positive rate)")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

png = os.path.join(_paths.RERUN_FIGS, "Figure2_ROC.png")
pdf = os.path.join(_paths.RERUN_FIGS, "Figure2_ROC.pdf")
fig.savefig(png, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
print(f"[OK] {png}")
print(f"[OK] {pdf}")
for clf, auc in rows:
    print(f"    {LABELS[clf]:<32} AUROC = {auc:.4f}")
