import io
import os
import sys
import json
import time
import types
import warnings

if "numba" not in sys.modules:
    numba = types.ModuleType("numba")
    numba.__path__ = []

    def _njit(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]

        def deco(f):
            return f
        return deco

    numba.njit = _njit
    numba.jit = _njit
    numba.prange = range
    typed = types.ModuleType("numba.typed")

    class _List(list):
        @classmethod
        def empty_list(cls, *a, **k):
            return cls()

    typed.List = _List
    typed.Dict = dict
    core = types.ModuleType("numba.core")
    core.__path__ = []
    ct = types.ModuleType("numba.core.types")
    ct.int64 = int
    ct.float64 = float
    sys.modules.update({"numba": numba, "numba.typed": typed,
                        "numba.core": core, "numba.core.types": ct})
    numba.typed = typed
    numba.core = core

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", write_through=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, confusion_matrix, brier_score_loss)
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

SEED = 42
EPS = 1e-6
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))

print("=" * 78)
print("I1  FULL FEATURE-IMPORTANCE HIERARCHY (frozen, unfiltered)")
print("=" * 78)
AGG = pd.read_csv(os.path.join(BUNDLE, "03_results",
                               "shap_top25_xgboost_aggregated.csv"))
AGG.columns = ["scale", "mean_abs_SHAP"]
RELABEL = {"Daily smartphone use": "SAPS disturbance of adaptive functions",
           "SAPS mean score": "SAPS mean score (wave t)",
           "SAPS sum score": "SAPS sum score (wave t)",
           "Prior high-risk": "Prior-wave high-risk classification",
           "Wave": "Transition index (study design)",
           "Cohort": "Cohort source (study design)"}
AGG["scale"] = AGG["scale"].replace(RELABEL)
SAPS_LABELS = [v for k, v in RELABEL.items() if "SAPS" in v or "Prior" in v] + \
              ["SAPS tolerance", "Baseline SAPS", "SAPS withdrawal", "SAPS virtual-life",
               "SAPS change from baseline", "SAPS slope"]
DESIGN_LABELS = ["Transition index (study design)", "Cohort source (study design)"]
AGG["group"] = np.where(AGG.scale.isin(SAPS_LABELS), "SAPS-derived",
                        np.where(AGG.scale.isin(DESIGN_LABELS), "Study design",
                                 "Psychosocial, behavioral, family, or demographic"))
AGG["rank"] = range(1, len(AGG) + 1)
AGG["share_pct"] = 100 * AGG.mean_abs_SHAP / AGG.mean_abs_SHAP.sum()
AGG.to_csv(os.path.join(RES, "shap_full_hierarchy.csv"), index=False, encoding="utf-8")
share = AGG.groupby("group").mean_abs_SHAP.sum()
print(f"{len(AGG)} interpretable scales; total aggregated mean |SHAP| "
      f"{AGG.mean_abs_SHAP.sum():.3f}")
for g, v in share.sort_values(ascending=False).items():
    print(f"  {g:<48} {v:.3f}  ({100 * v / AGG.mean_abs_SHAP.sum():.1f} %)")
print("\ntop 12 unfiltered:")
for _, r in AGG.head(12).iterrows():
    print(f"  {r['rank']:>2}. {r.scale:<42} {r.mean_abs_SHAP:.3f}  [{r.group}]")

df = _paths.load_transitions()
ALL = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
SP = [c for c in ALL if c.startswith("SP_")]
SAPS_ALL = SP + ["PREV_HIGH_RISK"]
DEMOG = ["YGENDER", "P_PYBRP1", "P_PYBRP2", "COHORT_SRC_e4", "WAVE_IDX"]
SETS = [
    ("M1", "Current SAPS status only", ["SP_SUM", "PREV_HIGH_RISK"]),
    ("M2", "All SAPS-derived history", SAPS_ALL),
    ("M3", "SAPS history plus demographics and design", SAPS_ALL + DEMOG),
    ("M4", "Full model (455 features)", ALL),
    ("M5", "Full model without any SAPS-derived feature",
     [c for c in ALL if c not in SAPS_ALL]),
]
for k, lab, cols in SETS:
    missing = [c for c in cols if c not in ALL]
    assert not missing, (k, missing)

X = df[ALL].astype(float)
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
dev, te = next(gss.split(X.values, y, groups=ids))

print("\n" + "=" * 78)
print("I2  NESTED FEATURE SETS INSIDE THE FROZEN PRIMARY SPLIT")
print("=" * 78)
print(f"development {len(dev):,} / held-out {len(te):,}; high risk "
      f"{int(y[te].sum())} ({100 * y[te].mean():.2f} %) in the held-out set")
for k, lab, cols in SETS:
    print(f"  {k}: {lab} ({len(cols)} feature{'s' if len(cols) != 1 else ''})")

NEUTRAL_XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                   colsample_bytree=1.0, min_child_weight=5, gamma=0.0,
                   reg_alpha=0.0, reg_lambda=1.0)
NEUTRAL_LR = dict(C=1.0, l1_ratio=0.5)

def pipe(algo, cols, tuned=False):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if algo == "LogRegEN":
        steps.append(("sc", StandardScaler()))
        params = BEST["LogRegEN"] if tuned else NEUTRAL_LR
        clf = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=3000,
                                 random_state=SEED, n_jobs=-1, **params)
    else:
        params = BEST["XGBoost"] if tuned else NEUTRAL_XGB
        clf = xgb.XGBClassifier(**params, random_state=SEED, eval_metric="auc",
                                tree_method="hist", n_jobs=-1, verbosity=0)
    steps += [("smote_enn", SMOTEENN(random_state=SEED)), ("clf", clf)]
    return ImbPipeline(steps)

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def _groups(g):
    out = {}
    for i, k in enumerate(g):
        out.setdefault(k, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}

GT = _groups(ids[te])
KT = np.array(list(GT.keys()))

def net_benefit(yt, prob, pt):
    pred = prob >= pt
    n = len(yt)
    tp = np.sum(pred & (yt == 1))
    fp = np.sum(pred & (yt == 0))
    return tp / n - (fp / n) * (pt / (1 - pt))

rows = []
store = {}
for algo in ("XGBoost", "LogRegEN"):
    print(f"\n--- {algo} ---")
    runs = [(k, lab, cols, False) for k, lab, cols in SETS]
    runs.append(("M4t", "Full model with the hyperparameters of the primary analysis",
                 ALL, True))
    for k, lab, cols, tuned in runs:
        t0 = time.time()
        Xs = X[cols].values
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(dev))
        for tr, va in skf.split(Xs[dev], y[dev], groups=ids[dev]):
            pp = pipe(algo, cols, tuned)
            pp.fit(Xs[dev][tr], y[dev][tr])
            oof[va] = pp.predict_proba(Xs[dev][va])[:, 1]
        grid = np.linspace(0.01, 0.99, 99)
        f1s = [f1_score(y[dev], (oof >= t).astype(int), zero_division=0) for t in grid]
        thr = float(grid[int(np.argmax(f1s))])
        pp = pipe(algo, cols, tuned)
        pp.fit(Xs[dev], y[dev])
        prob = pp.predict_proba(Xs[te])[:, 1]
        store[(algo, k)] = prob

        platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        platt.fit(logit(oof).reshape(-1, 1), y[dev])
        rc = platt.predict_proba(logit(prob).reshape(-1, 1))[:, 1]

        pred = (prob >= thr).astype(int)
        tn, fp, fn_, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        sl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        sl.fit(logit(prob).reshape(-1, 1), y[te])
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(logit(prob) + mid)))).mean() < y[te].mean():
                lo = mid
            else:
                hi = mid

        rng = np.random.default_rng(SEED)
        ba, bp = [], []
        for _ in range(2000):
            r = np.concatenate([GT[q] for q in rng.choice(KT, len(KT), replace=True)])
            if len(np.unique(y[te][r])) < 2:
                continue
            ba.append(roc_auc_score(y[te][r], prob[r]))
            bp.append(average_precision_score(y[te][r], prob[r]))
        rows.append(dict(
            analysis="I2_nested", algorithm=algo, model=k, label=lab,
            hyperparameters=("tuned on 455 features" if tuned else "neutral, identical "
                             "across feature sets"),
            n_features=len(cols), threshold=thr,
            AUROC=roc_auc_score(y[te], prob),
            AUROC_lo=float(np.percentile(ba, 2.5)),
            AUROC_hi=float(np.percentile(ba, 97.5)),
            PR_AUC=average_precision_score(y[te], prob),
            PR_AUC_lo=float(np.percentile(bp, 2.5)),
            PR_AUC_hi=float(np.percentile(bp, 97.5)),
            sensitivity=tp / (tp + fn_) if tp + fn_ else np.nan,
            specificity=tn / (tn + fp) if tn + fp else np.nan,
            PPV=tp / (tp + fp) if tp + fp else np.nan,
            F1=f1_score(y[te], pred, zero_division=0),
            accuracy=accuracy_score(y[te], pred),
            calibration_slope=float(sl.coef_[0][0]),
            calibration_intercept=float((lo + hi) / 2),
            Brier=brier_score_loss(y[te], prob),
            Brier_recalibrated=brier_score_loss(y[te], rc),
            net_benefit_at_005=net_benefit(y[te], rc, 0.05),
            net_benefit_at_010=net_benefit(y[te], rc, 0.10),
            net_benefit_at_020=net_benefit(y[te], rc, 0.20),
        ))
        pd.DataFrame(rows).to_csv(os.path.join(RES, "incremental_value.csv"),
                                  index=False, encoding="utf-8")
        r = rows[-1]
        print(f"  {k} {lab[:38]:<38} AUROC {r['AUROC']:.3f} "
              f"({r['AUROC_lo']:.3f}-{r['AUROC_hi']:.3f})  PR-AUC {r['PR_AUC']:.3f}  "
              f"slope {r['calibration_slope']:.3f}  Brier {r['Brier']:.4f}  "
              f"NB@0.10 {r['net_benefit_at_010']:+.4f}  [{time.time() - t0:.0f}s]")

print("\n" + "=" * 78)
print("PAIRED DIFFERENCES AGAINST THE FULL MODEL (M4), same held-out adolescents")
print("=" * 78)
for algo in ("XGBoost", "LogRegEN"):
    full = store[(algo, "M4")]
    for k, lab, cols in SETS:
        if k == "M4":
            continue
        other = store[(algo, k)]
        rng = np.random.default_rng(SEED)
        da, dp = [], []
        for _ in range(2000):
            r = np.concatenate([GT[q] for q in rng.choice(KT, len(KT), replace=True)])
            if len(np.unique(y[te][r])) < 2:
                continue
            da.append(roc_auc_score(y[te][r], full[r]) - roc_auc_score(y[te][r], other[r]))
            dp.append(average_precision_score(y[te][r], full[r])
                      - average_precision_score(y[te][r], other[r]))
        rows.append(dict(analysis="I2_delta_vs_full", algorithm=algo, model=k, label=lab,
                         AUROC=float(np.mean(da)),
                         AUROC_lo=float(np.percentile(da, 2.5)),
                         AUROC_hi=float(np.percentile(da, 97.5)),
                         PR_AUC=float(np.mean(dp)),
                         PR_AUC_lo=float(np.percentile(dp, 2.5)),
                         PR_AUC_hi=float(np.percentile(dp, 97.5))))
        r = rows[-1]
        print(f"  {algo:<9} M4 minus {k}: AUROC {r['AUROC']:+.3f} "
              f"({r['AUROC_lo']:+.3f} to {r['AUROC_hi']:+.3f})   PR-AUC "
              f"{r['PR_AUC']:+.3f} ({r['PR_AUC_lo']:+.3f} to {r['PR_AUC_hi']:+.3f})")

pd.DataFrame(rows).to_csv(os.path.join(RES, "incremental_value.csv"),
                          index=False, encoding="utf-8")
print(f"\nwritten: {os.path.join(RES, 'incremental_value.csv')}")
print(f"written: {os.path.join(RES, 'shap_full_hierarchy.csv')}")
