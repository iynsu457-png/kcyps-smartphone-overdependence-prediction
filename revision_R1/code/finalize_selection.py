import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
CLFS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "LogRegEN"]
PRETTY = {"XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost",
          "RandomForest": "Random Forest", "ExtraTrees": "Extra Trees",
          "LogRegEN": "Elastic-net Logistic Regression"}

F = pd.read_csv(os.path.join(RES, "selection_repeated_cv_folds.csv"))
prev = pd.read_csv(os.path.join(RES, "selection_repeated_cv.csv"))
thr_prev = prev.set_index("classifier")["prespecified_threshold_rep1"].to_dict()
FROZEN = json.load(open(os.path.join(BUNDLE, "03_results", "thresholds.json")))

wide = F.pivot_table(index=["repeat", "fold"], columns="classifier", values="AUROC")
best_per_fold = wide.idxmax(axis=1).value_counts()
ranks = wide.rank(axis=1, ascending=False).mean()

rows = []
for c in CLFS:
    f = F[F.classifier == c]
    rows.append(dict(
        classifier=c, pretty=PRETTY[c],
        mean_AUROC=f.AUROC.mean(), sd_AUROC=f.AUROC.std(ddof=1),
        se_AUROC=f.AUROC.std(ddof=1) / np.sqrt(len(f)),
        min_AUROC=f.AUROC.min(), max_AUROC=f.AUROC.max(),
        mean_PR_AUC=f.PR_AUC.mean(),
        mean_fold_F1=f.F1_at_fold_optimal_threshold.mean(),
        folds_ranked_first=int(best_per_fold.get(c, 0)),
        mean_rank=float(ranks[c]),
    ))
S = pd.DataFrame(rows).sort_values("mean_AUROC", ascending=False).reset_index(drop=True)

selected = S.classifier.iloc[0]
runner = S.classifier.iloc[1]
margin = S.mean_AUROC.iloc[0] - S.mean_AUROC.iloc[1]
se_sel = S.se_AUROC.iloc[0]

for c in CLFS:
    d = wide[selected] - wide[c]
    S.loc[S.classifier == c, "paired_diff_vs_selected"] = d.mean()
    S.loc[S.classifier == c, "paired_diff_lo"] = d.mean() - 1.96 * d.sem()
    S.loc[S.classifier == c, "paired_diff_hi"] = d.mean() + 1.96 * d.sem()

S["selected"] = S.classifier == selected
thr_sel = FROZEN[selected]
thr_src = ("F1-optimal on the development out-of-fold predictions, as frozen in the "
           "primary analysis")
S["prespecified_threshold_rep1"] = np.where(S.classifier == selected, thr_sel, np.nan)
S["threshold_source"] = np.where(S.classifier == selected, thr_src, "")
S["margin_over_runner_up"] = np.where(S.classifier == selected, margin, np.nan)

print("=" * 78)
print("FINAL SELECTION RULE: highest mean fold AUROC, no tie-break")
print("=" * 78)
print(S[["pretty", "mean_AUROC", "sd_AUROC", "se_AUROC", "mean_PR_AUC",
         "mean_fold_F1", "folds_ranked_first", "mean_rank"]]
      .to_string(index=False, float_format="%.4f"))
print(f"\nselected: {PRETTY[selected]}")
print(f"margin over {PRETTY[runner]}: {margin:.4f}   "
      f"standard error of the mean fold AUROC: {se_sel:.4f}")
print("the margin is "
      + ("smaller" if margin < se_sel else "larger")
      + " than one standard error, so the designation among the leading classifiers "
        "is reported as weakly supported")
n_within = int((S.mean_AUROC >= S.mean_AUROC.iloc[0] - se_sel).sum())
print(f"classifiers within one standard error of the best mean AUROC: {n_within}")
print("\nfor reference, the threshold-dependent tie-break of the first draft would "
      "have chosen: "
      + PRETTY[S[S.mean_AUROC >= S.mean_AUROC.iloc[0] - 0.005]
               .sort_values("mean_fold_F1", ascending=False).classifier.iloc[0]])

S.to_csv(os.path.join(RES, "selection_repeated_cv.csv"), index=False, encoding="utf-8")
print(f"\nwritten: {os.path.join(RES, 'selection_repeated_cv.csv')}")
