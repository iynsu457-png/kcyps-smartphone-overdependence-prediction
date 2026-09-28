import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from pandas.core.arrays.string_ import StringDtype, StringArray

_orig_strdtype_init = StringDtype.__init__
def _patched_strdtype_init(self, storage=None, *args, **kwargs):
    _orig_strdtype_init(self, storage)
StringDtype.__init__ = _patched_strdtype_init

def _patched_setstate(self, state):
    if isinstance(state, tuple) and len(state) == 2 and not isinstance(state[0], np.ndarray):
        arr = state[1]
    elif isinstance(state, tuple):
        arr = state[0]
    else:
        arr = state
    arr = np.asarray(arr, dtype=object)
    self.__init__(arr, copy=False)
StringArray.__setstate__ = _patched_setstate

ROOT_FINAL = _paths.RERUN_DIR
ROOT_OUT = _paths.RERUN_DIR
FIG_DIR = _paths.RERUN_FIGS

SEED = 42
TEST_SIZE = 0.20
SHAP_SUBSET = 1000

print("[1] Loading transitions.pkl ...")
df = _paths.load_transitions()
print(f"    n_rows = {len(df)}, n_cols = {len(df.columns)}")

META_COLS = {"ID", "WAVE", "TRANSITION", "y"}
feat_cols = [c for c in df.columns if c not in META_COLS]
print(f"    n_features = {len(feat_cols)}")

X = df[feat_cols].copy()
y = df["y"].astype(int).values
ids = df["ID"].astype(int).values

print("[2] Reproducing 80/20 GroupShuffleSplit (seed=42) ...")
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
tr_idx, te_idx = next(gss.split(X, y, groups=ids))
X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]
print(f"    development = {len(X_tr)} rows, test = {len(X_te)} rows")

print("[3] Loading XGBoost hyperparameters ...")
with open(_paths.BEST_PARAMS_JSON) as f:
    best = json.load(f)["XGBoost"]
print("    " + ", ".join(f"{k}={v}" for k, v in best.items()))

clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    use_label_encoder=False,
    random_state=SEED,
    n_jobs=-1,
    **best,
)

pipe = ImbPipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("smote", SMOTEENN(random_state=SEED)),
    ("clf", clf),
])

print("[4] Refitting XGBoost pipeline on the development set ...")
pipe.fit(X_tr, y_tr)
print("    done")

from sklearn.metrics import roc_auc_score
prob_te = pipe.predict_proba(X_te)[:, 1]
auc_check = roc_auc_score(y_te, prob_te)
print(f"[5] Held-out test AUROC = {auc_check:.6f}")
print(f"    (model_performance.csv reports 0.819103 for XGBoost)")

print(f"[6] Computing SHAP on {SHAP_SUBSET}-record test subset ...")
rng = np.random.default_rng(SEED)
sub_idx = rng.choice(len(X_te), size=min(SHAP_SUBSET, len(X_te)), replace=False)
X_sub = X_te.iloc[sub_idx]

X_sub_imp = pd.DataFrame(
    pipe.named_steps["imp"].transform(X_sub),
    columns=feat_cols,
    index=X_sub.index,
)

best_clf = pipe.named_steps["clf"]
explainer = shap.TreeExplainer(best_clf)
shap_vals = explainer.shap_values(X_sub_imp)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
elif shap_vals.ndim == 3:
    shap_vals = shap_vals[..., 1]

mean_abs = np.mean(np.abs(shap_vals), axis=0)
imp = pd.Series(mean_abs, index=feat_cols).sort_values(ascending=False)

FEATURE_LABELS = {
    "SP_MEAN":          "SAPS mean score",
    "SP_SUM":           "SAPS sum score",
    "SP_TOLERANCE":     "SAPS tolerance",
    "SP_WITHDRAWAL":    "SAPS withdrawal",
    "SP_VIRTUAL":       "SAPS virtual-life",
    "SP_DAILY":         "Daily smartphone use",
    "SP_SUM_BASELINE":  "Baseline SAPS",
    "SP_SUM_DELTA":     "SAPS change from baseline",
    "SP_SUM_SLOPE":     "SAPS slope",

    "PREV_HIGH_RISK":   "Prior high-risk",
    "WAVE_IDX":         "Wave",
    "COHORT_SRC_e4":    "Cohort",

    "YGENDER":          "Sex",
    "YBRT1A":           "Birth year",
    "YBRT1B":           "Birth month",
    "YTWIN":            "Twin",

    "P_PFBRTA1":        "Father age",

    "P_PHOMPOP":        "Household size",
    "P_PINCOME":        "Household income",
}

def _domain_of(code):
    if code.startswith("YPSY1"):    return "Self-esteem"
    if code.startswith("YPSY2"):    return "Depression"
    if code.startswith("YPSY3"):    return "Social anxiety"
    if code.startswith("YPSY4"):    return "Psychological well-being"
    if code.startswith("YPSY5"):    return "Peer relationship"
    if code.startswith("YPSY6"):    return "Self-control"
    if code.startswith("YPSY7"):    return "Aggression"
    if code.startswith("YINT"):     return "Smartphone use"
    if code.startswith("YTIM"):     return "Time use"
    if code.startswith("YMDA"):     return "Device access"
    if code.startswith("YEDU"):     return "School engagement"
    if code.startswith("YDLQ"):     return "Delinquency"
    if code.startswith("YFAM"):     return "Family relationship"
    if code.startswith("YACT"):     return "After-school activity"
    if code.startswith("YFUR"):     return "Career orientation"
    if code.startswith("YPHY"):     return "Physical health"
    if code.startswith("P_PMDA"):   return "Parental mediation"
    if code.startswith("P_PPSY"):   return "Parenting stress"
    if code.startswith("P_PFAM"):   return "Parent-reported family"
    if code.startswith("P_PEDU"):   return "Parent education involvement"
    if code.startswith("P_PSCHOOL"):return "Parent school involvement"
    if code.startswith("P_PWORK"):  return "Parent work hours"
    if code.startswith("P_PPHY"):   return "Parent health"
    if code.startswith("P_PFBRT"):  return "Father info"
    if code.startswith("P_PYBRT"):  return "Mother info"
    if code.startswith("P_PHOM"):   return "Household structure"
    if code.startswith("P_PINC"):   return "Household income"
    return None

from collections import defaultdict
from sklearn.preprocessing import StandardScaler

def _group_for_aggregation(code):
    if code in FEATURE_LABELS:
        return FEATURE_LABELS[code]
    return _domain_of(code)

group_for_code = {}
group_codes = defaultdict(list)
for code in feat_cols:
    g = _group_for_aggregation(code)
    if g is None:
        continue
    group_for_code[code] = g
    group_codes[g].append(code)

agg_mean_abs = defaultdict(float)
for code, val in zip(feat_cols, mean_abs):
    g = group_for_code.get(code)
    if g is not None:
        agg_mean_abs[g] += float(val)
agg_imp = pd.Series(agg_mean_abs).sort_values(ascending=False)

n_samples = X_sub_imp.shape[0]
agg_shap_cols = {}
agg_feat_cols_data = {}
for group, codes in group_codes.items():
    col_idx = [feat_cols.index(c) for c in codes]
    if len(codes) == 1:
        agg_shap_cols[group] = shap_vals[:, col_idx[0]]
        agg_feat_cols_data[group] = X_sub_imp.iloc[:, col_idx[0]].to_numpy()
    else:
        agg_shap_cols[group] = shap_vals[:, col_idx].sum(axis=1)
        block = X_sub_imp.iloc[:, col_idx].to_numpy()
        block_std = StandardScaler().fit_transform(block)
        agg_feat_cols_data[group] = block_std.mean(axis=1)

EXCLUDE_GROUPS = {
    "SAPS mean score",
    "SAPS sum score",
    "SAPS tolerance",
    "SAPS withdrawal",
    "SAPS virtual-life",
    "Baseline SAPS",
    "SAPS change from baseline",
    "SAPS slope",
    "Daily smartphone use",
    "Prior high-risk",
    "Wave",
    "Cohort",
}

agg_imp_filtered = agg_imp[~agg_imp.index.isin(EXCLUDE_GROUPS)].sort_values(
    ascending=False
)

out_csv_root = os.path.join(ROOT_FINAL, "shap_top25.csv")
imp.head(25).to_csv(out_csv_root, encoding="utf-8-sig")
print(f"[7] {out_csv_root}")

out_csv_local = os.path.join(ROOT_OUT, "shap_top25_xgboost.csv")
imp.head(25).to_csv(out_csv_local, encoding="utf-8-sig")
print(f"    {out_csv_local}")

out_csv_agg = os.path.join(ROOT_OUT, "shap_top25_xgboost_aggregated.csv")
agg_imp.to_csv(out_csv_agg, encoding="utf-8-sig", header=["mean_abs_SHAP"])
print(f"    {out_csv_agg}")

out_csv_filt = os.path.join(ROOT_OUT, "shap_top25_xgboost_aggregated_filtered.csv")
agg_imp_filtered.to_csv(out_csv_filt, encoding="utf-8-sig", header=["mean_abs_SHAP"])
print(f"    {out_csv_filt}")

TOP_N = 15
top_groups = agg_imp_filtered.head(TOP_N).index.tolist()
top_values = agg_imp_filtered.head(TOP_N).values

print("[8] Saving SHAP bar and beeswarm figures (interpretable scales only) ...")
plt.figure(figsize=(10, 9.5))
y_pos = list(range(len(top_groups)))[::-1]
plt.barh(y_pos, top_values, color="#3B75AF")
plt.yticks(y_pos, top_groups, fontsize=10)
plt.xlabel("Mean |SHAP value| (aggregated within scale)")
plt.title("SHAP feature importance — XGBoost (interpretable scales)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Figure4_SHAP_bar.png"), dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close()

agg_shap_matrix = np.column_stack([agg_shap_cols[g] for g in top_groups])
agg_feat_matrix = np.column_stack([agg_feat_cols_data[g] for g in top_groups])

plt.figure(figsize=(10, 10))
shap.summary_plot(
    agg_shap_matrix,
    agg_feat_matrix,
    feature_names=top_groups,
    show=False,
    max_display=TOP_N,
    plot_type="dot",
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Figure5_SHAP_beeswarm.png"), dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close()
print(f"    {FIG_DIR}/Figure4_SHAP_bar.png")
print(f"    {FIG_DIR}/Figure5_SHAP_beeswarm.png")
print(f"\n[9] Top-{TOP_N} interpretable scales (after excluding SAPS-derived"
      f" autocorrelation features and engineered design indicators):")
for i, (group, val) in enumerate(agg_imp_filtered.head(TOP_N).items(), start=1):
    n_items = len(group_codes[group])
    suffix = f"  ({n_items} items aggregated)" if n_items > 1 else ""
    print(f"  {i:>2}. {group:<35s} {val:.4f}{suffix}")
print(f"\n  Excluded from figure (still in model): {sorted(EXCLUDE_GROUPS)}")

print("\n[OK] SHAP recomputed for XGBoost.")
print("\nTop-15 features by mean |SHAP value| (XGBoost):")
for i, (name, val) in enumerate(imp.head(15).items(), start=1):
    print(f"  {i:>2}. {name:<25s} {val:.4f}")
