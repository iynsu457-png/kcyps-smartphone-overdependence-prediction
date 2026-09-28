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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

SEED = 42
REPEAT_SEEDS = [42, 43, 44]
N_FOLDS = 5
TIE_MARGIN = 0.005

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.path.join(REV, "results")
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
CLFS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "LogRegEN"]
PRETTY = {"XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost",
          "RandomForest": "Random Forest", "ExtraTrees": "Extra Trees",
          "LogRegEN": "Elastic-net Logistic Regression"}

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
X = df[feat].astype(float).values
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
dev, te = next(gss.split(X, y, groups=ids))
Xd, yd, gd = X[dev], y[dev], ids[dev]
print("=" * 78)
print("DEVELOPMENT-ONLY MODEL SELECTION BY REPEATED GROUPED CROSS-VALIDATION")
print("=" * 78)
print(f"development set n = {len(dev):,} transitions / {pd.Series(gd).nunique():,} "
      f"adolescents; high risk {int(yd.sum()):,} ({100 * yd.mean():.2f} %)")
print(f"held-out test set n = {len(te):,} - NOT touched by this script")
print(f"design: {len(REPEAT_SEEDS)} repeats x {N_FOLDS}-fold StratifiedGroupKFold "
      f"(seeds {REPEAT_SEEDS}), frozen hyperparameters\n")

def mk(name):
    p = BEST[name]
    if name == "XGBoost":
        m = xgb.XGBClassifier(**p, random_state=SEED, eval_metric="auc",
                              tree_method="hist", n_jobs=-1, verbosity=0)
    elif name == "LightGBM":
        m = lgb.LGBMClassifier(**p, random_state=SEED, n_jobs=-1, verbose=-1)
    elif name == "CatBoost":
        m = cb.CatBoostClassifier(**p, random_seed=SEED, verbose=0,
                                  allow_writing_files=False, thread_count=-1)
    elif name == "RandomForest":
        m = RandomForestClassifier(**p, random_state=SEED, n_jobs=-1)
    elif name == "ExtraTrees":
        m = ExtraTreesClassifier(**p, random_state=SEED, n_jobs=-1)
    elif name == "LogRegEN":
        m = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=3000,
                               random_state=SEED, n_jobs=-1, **p)
    else:
        raise ValueError(name)
    steps = [("imp", SimpleImputer(strategy="median"))]
    if name == "LogRegEN":
        steps.append(("sc", StandardScaler()))
    steps += [("smote_enn", SMOTEENN(random_state=SEED)), ("clf", m)]
    return ImbPipeline(steps)

def f1_opt(yt, yp):
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(yt, (yp >= t).astype(int), zero_division=0) for t in grid]
    b = int(np.argmax(f1s))
    return float(grid[b]), float(f1s[b])

fold_rows = []
oof = {name: {} for name in CLFS}
for name in CLFS:
    t0 = time.time()
    for rep, rs in enumerate(REPEAT_SEEDS, start=1):
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=rs)
        pooled = np.zeros(len(yd))
        for k, (tr, va) in enumerate(skf.split(Xd, yd, groups=gd), start=1):
            pipe = mk(name)
            pipe.fit(Xd[tr], yd[tr])
            pv = pipe.predict_proba(Xd[va])[:, 1]
            pooled[va] = pv
            thr, f1 = f1_opt(yd[va], pv)
            fold_rows.append(dict(classifier=name, repeat=rep, fold=k,
                                  AUROC=roc_auc_score(yd[va], pv),
                                  PR_AUC=average_precision_score(yd[va], pv),
                                  F1_at_fold_optimal_threshold=f1,
                                  fold_threshold=thr, n_val=len(va),
                                  n_pos=int(yd[va].sum())))
        oof[name][rep] = pooled
        pd.DataFrame(fold_rows).to_csv(
            os.path.join(RES, "selection_repeated_cv_folds.csv"),
            index=False, encoding="utf-8")
    fr = pd.DataFrame(fold_rows)
    fr = fr[fr.classifier == name]
    print(f"{PRETTY[name]:<32} mean fold AUROC {fr.AUROC.mean():.4f} "
          f"(SD {fr.AUROC.std(ddof=1):.4f}); pooled OOF AUROC per repeat "
          + " / ".join(f"{roc_auc_score(yd, oof[name][r]):.4f}" for r in (1, 2, 3))
          + f"  [{time.time() - t0:.0f}s]")

FR = pd.DataFrame(fold_rows)

summary = []
for name in CLFS:
    f = FR[FR.classifier == name]
    summary.append(dict(
        classifier=name, pretty=PRETTY[name],
        mean_AUROC=f.AUROC.mean(), sd_AUROC=f.AUROC.std(ddof=1),
        min_AUROC=f.AUROC.min(), max_AUROC=f.AUROC.max(),
        mean_PR_AUC=f.PR_AUC.mean(),
        mean_fold_F1=f.F1_at_fold_optimal_threshold.mean(),
        pooled_OOF_AUROC_rep1=roc_auc_score(yd, oof[name][1]),
        pooled_OOF_AUROC_rep2=roc_auc_score(yd, oof[name][2]),
        pooled_OOF_AUROC_rep3=roc_auc_score(yd, oof[name][3]),
    ))
S = pd.DataFrame(summary).sort_values("mean_AUROC", ascending=False)

best_auc = S.mean_AUROC.iloc[0]
contenders = S[S.mean_AUROC >= best_auc - TIE_MARGIN]
selected = (contenders.sort_values("mean_fold_F1", ascending=False).classifier.iloc[0]
            if len(contenders) > 1 else S.classifier.iloc[0])

wide = FR.pivot_table(index=["repeat", "fold"], columns="classifier", values="AUROC")
best_per_fold = wide.idxmax(axis=1)
S["folds_ranked_first"] = S.classifier.map(best_per_fold.value_counts()).fillna(0).astype(int)
S["mean_rank"] = S.classifier.map(wide.rank(axis=1, ascending=False).mean())

for name in CLFS:
    d = wide[selected] - wide[name]
    S.loc[S.classifier == name, "paired_diff_vs_selected"] = d.mean()
    S.loc[S.classifier == name, "paired_diff_lo"] = d.mean() - 1.96 * d.sem()
    S.loc[S.classifier == name, "paired_diff_hi"] = d.mean() + 1.96 * d.sem()

print("\n" + "=" * 78)
print("PRESPECIFIED SELECTION (development data only)")
print("=" * 78)
print(S[["pretty", "mean_AUROC", "sd_AUROC", "mean_fold_F1", "folds_ranked_first",
         "mean_rank", "paired_diff_vs_selected"]].to_string(index=False,
                                                            float_format="%.4f"))
print(f"\nclassifiers within {TIE_MARGIN} of the best mean AUROC: "
      f"{', '.join(PRETTY[c] for c in contenders.classifier)}")
print(f"SELECTED PRIMARY MODEL: {PRETTY[selected]}")

thr1, f1_1 = f1_opt(yd, oof[selected][1])
print(f"F1-optimal threshold on the repeat-1 pooled out-of-fold predictions of "
      f"{PRETTY[selected]}: {thr1:.2f} (OOF F1 {f1_1:.3f})")
frozen = json.load(open(os.path.join(BUNDLE, "03_results", "thresholds.json")))
print(f"threshold frozen in the submitted analysis: {frozen[selected]:.2f}")

S["selected"] = S.classifier == selected
S["prespecified_threshold_rep1"] = np.where(S.classifier == selected, thr1, np.nan)
S.to_csv(os.path.join(RES, "selection_repeated_cv.csv"), index=False, encoding="utf-8")
print(f"\nwritten: {os.path.join(RES, 'selection_repeated_cv.csv')}")
