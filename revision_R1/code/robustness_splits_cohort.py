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
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, confusion_matrix)
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

SEED = 42
SEEDS_PRIMARY = [42] + list(range(1, 20))
SEEDS_ALL = [42, 1, 2, 3, 4]
PRIMARY = "XGBoost"

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.path.join(REV, "results")
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
THR = json.load(open(os.path.join(BUNDLE, "03_results", "thresholds.json")))
CLFS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "LogRegEN"]
PRETTY = {"XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost",
          "RandomForest": "Random Forest", "ExtraTrees": "Extra Trees",
          "LogRegEN": "Elastic-net Logistic Regression"}
OUTCOME_YEAR = {("m1", "G7->G8"): 2019, ("m1", "G8->G9"): 2020, ("m1", "G9->G10"): 2021,
                ("e4", "G7->G8"): 2022, ("e4", "G8->G9"): 2023, ("e4", "G9->G10"): 2024}
CSV = os.path.join(RES, "robustness_results.csv")

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
X = df[feat].astype(float).values
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
trans = df["TRANSITION"].astype(str).values
cohort = np.where(df["COHORT_SRC_e4"].astype(int).values == 1, "e4", "m1")

print("=" * 78)
print("SPLIT STABILITY AND INTERNAL-EXTERNAL VALIDATION")
print("=" * 78)
print(f"primary analytic file n = {len(df):,} transitions / {df['ID'].nunique():,} "
      f"adolescents; high risk {int(y.sum()):,} ({100 * y.mean():.2f} %)")
for c in ("m1", "e4"):
    m = cohort == c
    yrs = sorted({OUTCOME_YEAR[(c, t)] for t in set(trans[m])})
    print(f"  {c}: n = {m.sum():,} / {pd.Series(ids[m]).nunique():,} adolescents; "
          f"high risk {int(y[m].sum()):,} ({100 * y[m].mean():.2f} %); "
          f"outcome waves {yrs[0]}-{yrs[-1]}")
print("  the two panels share no adolescent, and their outcome years do not "
      "overlap, so a cohort holdout is also a calendar-period holdout\n")

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

def metrics(yt, yp, thr):
    pred = (yp >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return dict(AUROC=roc_auc_score(yt, yp),
                PR_AUC=average_precision_score(yt, yp),
                F1=f1_score(yt, pred, zero_division=0),
                sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                specificity=tn / (tn + fp) if tn + fp else np.nan,
                accuracy=accuracy_score(yt, pred),
                PPV=tp / (tp + fp) if tp + fp else np.nan,
                NPV=tn / (tn + fn) if tn + fn else np.nan)

def _groups(gid):
    out = {}
    for i, g in enumerate(gid):
        out.setdefault(g, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}

def cluster_ci(yt, yp, gid, thr, n_boot=2000, seed=SEED):
    point = metrics(yt, yp, thr)
    rng = np.random.default_rng(seed)
    grouped = _groups(gid)
    keys = np.array(list(grouped.keys()))
    boot = {k: [] for k in point}
    for _ in range(n_boot):
        rows = np.concatenate([grouped[k] for k in rng.choice(keys, len(keys), replace=True)])
        if len(np.unique(yt[rows])) < 2:
            continue
        for k, v in metrics(yt[rows], yp[rows], thr).items():
            boot[k].append(v)
    return {k: (point[k], float(np.percentile(boot[k], 2.5)),
                float(np.percentile(boot[k], 97.5))) for k in point}

def oof_threshold(name, idx):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(idx))
    for tr, va in skf.split(X[idx], y[idx], groups=ids[idx]):
        pipe = mk(name)
        pipe.fit(X[idx][tr], y[idx][tr])
        oof[va] = pipe.predict_proba(X[idx][va])[:, 1]
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y[idx], (oof >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])

rows = []

def save():
    pd.DataFrame(rows).to_csv(CSV, index=False, encoding="utf-8")

print("=" * 78)
print("R1  REPEATED GROUPED 80/20 SPLITS (frozen threshold, no re-tuning)")
print("=" * 78)
for name in CLFS:
    seeds = SEEDS_PRIMARY if name == PRIMARY else SEEDS_ALL
    t0 = time.time()
    aucs = []
    for sd in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=sd)
        dv, te = next(gss.split(X, y, groups=ids))
        pipe = mk(name)
        pipe.fit(X[dv], y[dv])
        pv = pipe.predict_proba(X[te])[:, 1]
        m = metrics(y[te], pv, THR[name])
        aucs.append(m["AUROC"])
        rows.append(dict(analysis="R1_seed_variation", classifier=name, seed=sd,
                         n_test=len(te), n_pos_test=int(y[te].sum()),
                         threshold=THR[name], **m))
        save()
    a = np.array(aucs)
    print(f"{PRETTY[name]:<32} {len(seeds):>2} seeds  AUROC mean {a.mean():.3f} "
          f"SD {a.std(ddof=1):.3f}  range {a.min():.3f}-{a.max():.3f}  "
          f"IQR {np.percentile(a, 25):.3f}-{np.percentile(a, 75):.3f}  "
          f"[{time.time() - t0:.0f}s]")
    rows.append(dict(analysis="R1_seed_summary", classifier=name, n_seeds=len(seeds),
                     AUROC=a.mean(), AUROC_sd=a.std(ddof=1), AUROC_min=a.min(),
                     AUROC_max=a.max(), AUROC_q25=np.percentile(a, 25),
                     AUROC_q75=np.percentile(a, 75)))
    save()

print("\n" + "=" * 78)
print("R2  INTERNAL-EXTERNAL VALIDATION BY COHORT (= CALENDAR PERIOD)")
print("=" * 78)
probs = {}
for train_c, test_c in (("m1", "e4"), ("e4", "m1")):
    tr_idx = np.where(cohort == train_c)[0]
    te_idx = np.where(cohort == test_c)[0]
    print(f"\ntrain {train_c} (n = {len(tr_idx):,}) -> test {test_c} "
          f"(n = {len(te_idx):,}, high risk {int(y[te_idx].sum())})")
    for name in CLFS:
        t0 = time.time()
        pipe = mk(name)
        pipe.fit(X[tr_idx], y[tr_idx])
        pv = pipe.predict_proba(X[te_idx])[:, 1]
        probs[(train_c, name)] = (te_idx, pv)
        m = cluster_ci(y[te_idx], pv, ids[te_idx], THR[name])
        rows.append(dict(analysis="R2_cohort_holdout", classifier=name,
                         train_cohort=train_c, test_cohort=test_c,
                         threshold_source="frozen (submitted analysis)",
                         threshold=THR[name], n_test=len(te_idx),
                         n_pos_test=int(y[te_idx].sum()),
                         **{k: v[0] for k, v in m.items()},
                         **{f"{k}_lo": v[1] for k, v in m.items()},
                         **{f"{k}_hi": v[2] for k, v in m.items()}))
        save()
        print(f"  {PRETTY[name]:<32} AUROC {m['AUROC'][0]:.3f} "
              f"({m['AUROC'][1]:.3f}-{m['AUROC'][2]:.3f})  "
              f"sens {m['sensitivity'][0]:.3f}  spec {m['specificity'][0]:.3f}  "
              f"PPV {m['PPV'][0]:.3f}  [{time.time() - t0:.0f}s]")

    t0 = time.time()
    thr_in = oof_threshold(PRIMARY, tr_idx)
    te_idx, pv = probs[(train_c, PRIMARY)]
    m = cluster_ci(y[te_idx], pv, ids[te_idx], thr_in)
    rows.append(dict(analysis="R2_cohort_holdout", classifier=PRIMARY,
                     train_cohort=train_c, test_cohort=test_c,
                     threshold_source="training-cohort out-of-fold",
                     threshold=thr_in, n_test=len(te_idx),
                     n_pos_test=int(y[te_idx].sum()),
                     **{k: v[0] for k, v in m.items()},
                     **{f"{k}_lo": v[1] for k, v in m.items()},
                     **{f"{k}_hi": v[2] for k, v in m.items()}))
    save()
    print(f"  {PRETTY[PRIMARY]} with a threshold re-derived inside {train_c} only: "
          f"thr {thr_in:.2f}  sens {m['sensitivity'][0]:.3f}  "
          f"spec {m['specificity'][0]:.3f}  F1 {m['F1'][0]:.3f}  "
          f"[{time.time() - t0:.0f}s]")

print("\n" + "=" * 78)
print("R3  PER-OUTCOME-YEAR DISCRIMINATION INSIDE EACH COHORT HOLDOUT (primary model)")
print("=" * 78)
for train_c, test_c in (("m1", "e4"), ("e4", "m1")):
    te_idx, pv = probs[(train_c, PRIMARY)]
    for t in ["G7->G8", "G8->G9", "G9->G10"]:
        m = trans[te_idx] == t
        if len(np.unique(y[te_idx][m])) < 2:
            continue
        a = roc_auc_score(y[te_idx][m], pv[m])
        yr = OUTCOME_YEAR[(test_c, t)]
        print(f"  train {train_c} -> test {test_c} {t} (outcome {yr}): "
              f"AUROC {a:.3f}  n {int(m.sum()):,}  high risk {int(y[te_idx][m].sum())}")
        rows.append(dict(analysis="R3_per_year", classifier=PRIMARY,
                         train_cohort=train_c, test_cohort=test_c, stratum=t,
                         outcome_year=yr, AUROC=a, n_test=int(m.sum()),
                         n_pos_test=int(y[te_idx][m].sum())))
        save()

save()
print(f"\nwritten: {CSV}")
