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
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, confusion_matrix, brier_score_loss,
                             mean_absolute_error, r2_score)
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.path.join(REV, "results")
FS_PKL = os.path.join(RES, "transitions_fullspectrum.pkl")
BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
CSV_OUT = os.path.join(RES, "sensitivity_results.csv")

F = pd.read_pickle(FS_PKL)
FEATS = [c for c in F.columns if c not in ("y_bin3", "y_ord", "sp_t1", "ID", "TRANSITION")]
assert len(FEATS) == 455, len(FEATS)
X = F[FEATS].astype(float).values
ids = F["ID"].astype("int64").values
y_bin = F["y_bin3"].astype(int).values
y_ord = F["y_ord"].astype(int).values
sp_t1 = F["sp_t1"].astype(float).values

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
dev, te = next(gss.split(X, y_bin, groups=ids))
extreme_dev = dev[y_ord[dev] != 1]
extreme_te = te[y_ord[te] != 1]
keep = np.isin(te, extreme_te)

print("=" * 78)
print("FULL-SPECTRUM FILE AND SHARED 80/20 ADOLESCENT-LEVEL SPLIT")
print("=" * 78)
print(f"full spectrum        n = {len(F):,} transitions / {F['ID'].nunique():,} adolescents")
print(f"  general use          {int((y_ord == 0).sum()):>7,}  {100 * (y_ord == 0).mean():5.2f}%")
print(f"  potential risk       {int((y_ord == 1).sum()):>7,}  {100 * (y_ord == 1).mean():5.2f}%")
print(f"  high risk            {int((y_ord == 2).sum()):>7,}  {100 * (y_ord == 2).mean():5.2f}%")
for nm, ix in (("development", dev), ("held-out test", te)):
    sub = y_bin[ix][y_ord[ix] != 1]
    print(f"{nm:<20} n = {len(ix):,} / {pd.Series(ids[ix]).nunique():,} adolescents; "
          f"high risk {int(y_bin[ix].sum()):,} ({100 * y_bin[ix].mean():.2f}%); "
          f"extreme-group subset n = {len(sub):,} (high risk {100 * sub.mean():.2f}%)")

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

CLFS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "LogRegEN"]

def tune_thr(yt, yp):
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(yt, (yp >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])

def fit_predict(name, train_idx):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_idx))
    for tr, va in skf.split(X[train_idx], y_bin[train_idx], groups=ids[train_idx]):
        pipe = mk(name)
        pipe.fit(X[train_idx][tr], y_bin[train_idx][tr])
        oof[va] = pipe.predict_proba(X[train_idx][va])[:, 1]
    thr = tune_thr(y_bin[train_idx], oof)
    pipe = mk(name)
    pipe.fit(X[train_idx], y_bin[train_idx])
    return thr, pipe.predict_proba(X[te])[:, 1]

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
                NPV=tn / (tn + fn) if tn + fn else np.nan,
                Brier=brier_score_loss(yt, yp))

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

def delta_ci(yt_a, yp_a, yt_b, yp_b, gid_a, gid_b, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    ga, gb = _groups(gid_a), _groups(gid_b)
    keys = np.array(sorted(ga.keys()))
    d = []
    for _ in range(n_boot):
        s = rng.choice(keys, len(keys), replace=True)
        ra = np.concatenate([ga[k] for k in s])
        rb = [gb[k] for k in s if k in gb]
        if len(rb) == 0:
            continue
        rb = np.concatenate(rb)
        if len(np.unique(yt_a[ra])) < 2 or len(np.unique(yt_b[rb])) < 2:
            continue
        d.append(roc_auc_score(yt_b[rb], yp_b[rb]) - roc_auc_score(yt_a[ra], yp_a[ra]))
    d = np.asarray(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

rows, store = [], {}

def record(analysis, name, scope, thr, m):
    rows.append(dict(analysis=analysis, classifier=name, scope=scope, threshold=thr,
                     **{k: v[0] for k, v in m.items()},
                     **{f"{k}_lo": v[1] for k, v in m.items()},
                     **{f"{k}_hi": v[2] for k, v in m.items()}))
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8")

print("\n" + "=" * 78)
print("S1/S2  TRAINED ON THE FULL SPECTRUM (potential risk recoded as y = 0)")
print("=" * 78)
for name in CLFS:
    t0 = time.time()
    thr, prob = fit_predict(name, dev)
    fs = cluster_ci(y_bin[te], prob, ids[te], thr)
    ex = cluster_ci(y_bin[extreme_te], prob[keep], ids[extreme_te], thr)
    dlt = delta_ci(y_bin[te], prob, y_bin[extreme_te], prob[keep], ids[te], ids[extreme_te])
    store[f"FS|{name}"] = dict(thr=thr, prob=prob.tolist())
    record("S1_fullspectrum_training", name, "full spectrum test", thr, fs)
    record("S2_fullspectrum_training", name, "extreme-group test", thr, ex)
    rows.append(dict(analysis="S2_spectrum_delta", classifier=name,
                     scope="AUROC(extreme) - AUROC(full)", AUROC=dlt[0],
                     AUROC_lo=dlt[1], AUROC_hi=dlt[2]))
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8")
    print(f"{name:<14} thr={thr:.2f}  "
          f"full {fs['AUROC'][0]:.3f} ({fs['AUROC'][1]:.3f}-{fs['AUROC'][2]:.3f})  |  "
          f"extreme {ex['AUROC'][0]:.3f} ({ex['AUROC'][1]:.3f}-{ex['AUROC'][2]:.3f})  |  "
          f"gain {dlt[0]:+.3f} ({dlt[1]:+.3f} to {dlt[2]:+.3f})  [{time.time() - t0:.0f}s]")

print("\n" + "=" * 78)
print("S3  TRAINED ON THE EXTREME GROUPS ONLY (submitted design, same split)")
print("=" * 78)
for name in CLFS:
    t0 = time.time()
    thr, prob = fit_predict(name, extreme_dev)
    ex = cluster_ci(y_bin[extreme_te], prob[keep], ids[extreme_te], thr)
    fs = cluster_ci(y_bin[te], prob, ids[te], thr)
    dlt = delta_ci(y_bin[te], prob, y_bin[extreme_te], prob[keep], ids[te], ids[extreme_te])
    store[f"EX|{name}"] = dict(thr=thr, prob=prob.tolist())
    record("S3_restricted_training", name, "extreme-group test", thr, ex)
    record("S3_restricted_training", name, "full spectrum test", thr, fs)
    rows.append(dict(analysis="S3_spectrum_delta", classifier=name,
                     scope="AUROC(extreme) - AUROC(full)", AUROC=dlt[0],
                     AUROC_lo=dlt[1], AUROC_hi=dlt[2]))
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8")
    print(f"{name:<14} thr={thr:.2f}  "
          f"extreme {ex['AUROC'][0]:.3f} ({ex['AUROC'][1]:.3f}-{ex['AUROC'][2]:.3f})  |  "
          f"full {fs['AUROC'][0]:.3f} ({fs['AUROC'][1]:.3f}-{fs['AUROC'][2]:.3f})  |  "
          f"gain {dlt[0]:+.3f} ({dlt[1]:+.3f} to {dlt[2]:+.3f})  [{time.time() - t0:.0f}s]")

print("\n" + "=" * 78)
print("S4  THREE-CATEGORY OUTCOME (general / potential / high), XGBoost")
print("=" * 78)
p = dict(BEST["XGBoost"])
t0 = time.time()
m3 = ImbPipeline([("imp", SimpleImputer(strategy="median")),
                  ("smote_enn", SMOTEENN(random_state=SEED)),
                  ("clf", xgb.XGBClassifier(**p, objective="multi:softprob", num_class=3,
                                            random_state=SEED, tree_method="hist",
                                            n_jobs=-1, verbosity=0))])
m3.fit(X[dev], y_ord[dev])
P3 = m3.predict_proba(X[te])
lab = {0: "general use", 1: "potential risk", 2: "high risk"}
for k in (0, 1, 2):
    a = roc_auc_score((y_ord[te] == k).astype(int), P3[:, k])
    print(f"  one-vs-rest AUROC, {lab[k]:<16} {a:.3f}")
    rows.append(dict(analysis="S4_three_category", classifier="XGBoost",
                     scope=f"one-vs-rest {lab[k]}", AUROC=a))
mac = roc_auc_score(y_ord[te], P3, multi_class="ovr", average="macro")
mask = y_ord[te] != 1
hg = roc_auc_score((y_ord[te][mask] == 2).astype(int),
                   P3[mask, 2] / (P3[mask, 0] + P3[mask, 2]))
print(f"  macro one-vs-rest AUROC          {mac:.3f}")
print(f"  high risk vs general use only    {hg:.3f}   [{time.time() - t0:.0f}s]")
rows.append(dict(analysis="S4_three_category", classifier="XGBoost",
                 scope="macro one-vs-rest", AUROC=mac))
rows.append(dict(analysis="S4_three_category", classifier="XGBoost",
                 scope="high risk vs general use (extreme subset)", AUROC=hg))

print("\n" + "=" * 78)
print("S5  CONTINUOUS OUTCOME-WAVE SAPS SUM, XGBoost regression")
print("=" * 78)
t0 = time.time()
reg = SkPipeline([("imp", SimpleImputer(strategy="median")),
                  ("clf", xgb.XGBRegressor(**p, random_state=SEED, tree_method="hist",
                                           n_jobs=-1, verbosity=0))])
reg.fit(X[dev], sp_t1[dev])
pr = reg.predict(X[te])
r2, mae = r2_score(sp_t1[te], pr), mean_absolute_error(sp_t1[te], pr)
rho = float(spearmanr(sp_t1[te], pr).statistic)
a_hi = roc_auc_score(y_bin[te], pr)
a_hi_ex = roc_auc_score(y_bin[extreme_te], pr[keep])
print(f"  R2 {r2:.3f}   MAE {mae:.2f} points   Spearman rho {rho:.3f}")
print("  predicted score as a ranking of high-risk status:")
print(f"     full spectrum test  AUROC {a_hi:.3f}")
print(f"     extreme-group test  AUROC {a_hi_ex:.3f}   [{time.time() - t0:.0f}s]")
for nm, v in (("R2", r2), ("MAE", mae), ("Spearman_rho", rho)):
    rows.append(dict(analysis="S5_continuous", classifier="XGBoost regression",
                     scope=nm, value=v))
rows.append(dict(analysis="S5_continuous", classifier="XGBoost regression",
                 scope="AUROC for high risk, full spectrum test", AUROC=a_hi))
rows.append(dict(analysis="S5_continuous", classifier="XGBoost regression",
                 scope="AUROC for high risk, extreme-group test", AUROC=a_hi_ex))

pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8")
np.save(os.path.join(RES, "fs_test_index.npy"), te)
json.dump(store, open(os.path.join(RES, "fs_test_probabilities.json"), "w"))
print(f"\nwritten: {CSV_OUT}")
