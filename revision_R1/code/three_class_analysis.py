import io
import json
import os
import sys
import time
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", write_through=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, cohen_kappa_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
SEED = 42
NBOOT = 2000
BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))

F = pd.read_pickle(os.path.join(RES, "transitions_fullspectrum.pkl"))
FEATS = [c for c in F.columns if c not in ("y_bin3", "y_ord", "sp_t1", "ID", "TRANSITION")]
X = F[FEATS].astype(float).values
ids = F["ID"].astype("int64").values
y_ord = F["y_ord"].astype(int).values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
               .split(X, F["y_bin3"].astype(int).values, groups=ids))
assert (np.load(os.path.join(RES, "fs_test_index.npy")) == te).all()
s_t, d_t = F["SP_SUM"].astype(float).values, F["SP_DAILY"].astype(float).values
band_t = np.where(np.isnan(s_t), -1,
                  np.where(s_t >= 45, 2, np.where((s_t >= 42) | (d_t >= 14), 1, 0)))
saps_imp = np.where(np.isnan(s_t), np.nanmedian(s_t[dev]), s_t)
LAB = {0: "general use", 1: "potential risk", 2: "high risk"}
rows = []

def fauc(y, s):
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def groups(g):
    out = {}
    for i, k in enumerate(g):
        out.setdefault(k, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}

def boot(stat_fns, gid, n=NBOOT):
    grp = groups(gid)
    keys = np.array(list(grp.keys()))
    rng = np.random.default_rng(SEED)
    allidx = np.arange(len(gid))
    pt = {k: f(allidx) for k, f in stat_fns.items()}
    bs = {k: [] for k in stat_fns}
    for _ in range(n):
        r = np.concatenate([grp[k] for k in rng.choice(keys, len(keys), replace=True)])
        for k, f in stat_fns.items():
            bs[k].append(f(r))
    return {k: (pt[k], np.nanpercentile(bs[k], 2.5), np.nanpercentile(bs[k], 97.5))
            for k in stat_fns}

def pipe(multi=False):
    p = dict(BEST["XGBoost"])
    extra = dict(objective="multi:softprob", num_class=3) if multi else \
        dict(eval_metric="auc")
    return ImbPipeline([("imp", SimpleImputer(strategy="median")),
                        ("smote_enn", SMOTEENN(random_state=SEED)),
                        ("clf", xgb.XGBClassifier(**p, **extra, random_state=SEED,
                                                  tree_method="hist", n_jobs=-1,
                                                  verbosity=0))])

def add(**k):
    rows.append(k)

print("=" * 78)
print("C1  THREE-CLASS MODEL")
print("=" * 78)
t0 = time.time()
m3 = pipe(multi=True)
m3.fit(X[dev], y_ord[dev])
P3 = m3.predict_proba(X[te])
yt = y_ord[te]
pred = P3.argmax(axis=1)
fns = {}
for k in (0, 1, 2):
    fns[f"AUROC {LAB[k]}"] = (lambda r, k=k: fauc((yt[r] == k).astype(int), P3[r, k]))
    fns[f"PR-AUC {LAB[k]}"] = (lambda r, k=k: average_precision_score(
        (yt[r] == k).astype(int), P3[r, k]) if (yt[r] == k).any() else np.nan)
    fns[f"recall {LAB[k]}"] = (lambda r, k=k: np.mean(pred[r][yt[r] == k] == k))
    fns[f"precision {LAB[k]}"] = (lambda r, k=k: np.mean(yt[r][pred[r] == k] == k)
                                  if (pred[r] == k).any() else np.nan)
fns["macro AUROC"] = lambda r: np.nanmean([fauc((yt[r] == k).astype(int), P3[r, k])
                                           for k in (0, 1, 2)])
fns["macro F1"] = lambda r: f1_score(yt[r], pred[r], average="macro")
fns["quadratic-weighted kappa"] = lambda r: cohen_kappa_score(yt[r], pred[r],
                                                              weights="quadratic")
fns["AUROC elevated (potential or high) vs general, P(potential)+P(high)"] = \
    lambda r: fauc((yt[r] >= 1).astype(int), P3[r, 1] + P3[r, 2])
res = boot(fns, ids[te])
for k, (v, lo, hi) in res.items():
    add(analysis="C1_three_class", metric=k, value=v, lo=lo, hi=hi)
    print(f"  {k:<72} {v:.3f} ({lo:.3f}-{hi:.3f})")
cm = confusion_matrix(yt, pred, labels=[0, 1, 2])
for i in range(3):
    for j in range(3):
        add(analysis="C1_confusion", true=LAB[i], predicted=LAB[j], n=int(cm[i, j]))
print("  confusion (rows true, cols predicted):\n", cm, f"[{time.time() - t0:.0f}s]")
for k in (0, 1, 2):
    add(analysis="C1_counts", cls=LAB[k], n_test=int((yt == k).sum()),
        n_dev=int((y_ord[dev] == k).sum()))

print("\n" + "=" * 78)
print("C2  BINARY ELEVATED RISK (potential or high vs general use)")
print("=" * 78)
t0 = time.time()
ye = (y_ord >= 1).astype(int)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(dev))
for tr, va in skf.split(X[dev], ye[dev], groups=ids[dev]):
    m = pipe()
    m.fit(X[dev][tr], ye[dev][tr])
    oof[va] = m.predict_proba(X[dev][va])[:, 1]
grid = np.linspace(0.01, 0.99, 99)
thr = float(grid[int(np.argmax([f1_score(ye[dev], (oof >= t).astype(int), zero_division=0)
                                 for t in grid]))])
me = pipe()
me.fit(X[dev], ye[dev])
pe = me.predict_proba(X[te])[:, 1]
yte = ye[te]
fl = (pe >= thr).astype(int)

def _thrm(r):
    y_, f_ = yte[r], fl[r]
    tp, fn = ((f_ == 1) & (y_ == 1)).sum(), ((f_ == 0) & (y_ == 1)).sum()
    fp, tn = ((f_ == 1) & (y_ == 0)).sum(), ((f_ == 0) & (y_ == 0)).sum()
    return dict(sens=tp / (tp + fn), spec=tn / (tn + fp),
                ppv=tp / (tp + fp) if tp + fp else np.nan, npv=tn / (tn + fn),
                f1=2 * tp / (2 * tp + fp + fn))

fns = {"AUROC": lambda r: fauc(yte[r], pe[r]),
       "PR-AUC": lambda r: average_precision_score(yte[r], pe[r]),
       "SAPS sum alone AUROC": lambda r: fauc(yte[r], saps_imp[te][r]),
       "AUROC minus SAPS sum alone": lambda r: fauc(yte[r], pe[r]) - fauc(yte[r],
                                                                        saps_imp[te][r])}
for m_ in ("sens", "spec", "ppv", "npv", "f1"):
    fns[m_] = (lambda r, m_=m_: _thrm(r)[m_])
res = boot(fns, ids[te])
add(analysis="C2_elevated", metric="threshold", value=thr)
add(analysis="C2_elevated", metric="prevalence", value=float(yte.mean()))
for k, (v, lo, hi) in res.items():
    add(analysis="C2_elevated", metric=k, value=v, lo=lo, hi=hi)
    print(f"  {k:<30} {v:.3f} ({lo:.3f}-{hi:.3f})")
print(f"  prevalence {yte.mean():.3f}; threshold {thr:.2f}  [{time.time() - t0:.0f}s]")

print("\n" + "=" * 78)
print("C3  ONSET AMONG TRANSITIONS FROM GENERAL USE AT WAVE t")
print("=" * 78)
sub = band_t[te] == 0
ys, ps_, ss = yte[sub], pe[sub], saps_imp[te][sub]
fns = {"AUROC, elevated-risk model": lambda r: fauc(ys[r], ps_[r]),
       "PR-AUC, elevated-risk model": lambda r: average_precision_score(ys[r], ps_[r]),
       "AUROC, SAPS sum alone": lambda r: fauc(ys[r], ss[r]),
       "AUROC difference": lambda r: fauc(ys[r], ps_[r]) - fauc(ys[r], ss[r])}
res = boot(fns, ids[te][sub])
add(analysis="C3_onset", metric="n", value=int(sub.sum()))
add(analysis="C3_onset", metric="events", value=int(ys.sum()))
for k, (v, lo, hi) in res.items():
    add(analysis="C3_onset", metric=k, value=v, lo=lo, hi=hi)
    print(f"  {k:<30} {v:.3f} ({lo:.3f}-{hi:.3f})")
print(f"  n {int(sub.sum())}, onset {int(ys.sum())} ({100 * ys.mean():.1f}%)")

print("\n" + "=" * 78)
print("C4  POTENTIAL-RISK TRANSITIONS UNDER THE PRIMARY DESIGN")
print("=" * 78)
store = json.load(open(os.path.join(RES, "fs_test_probabilities.json")))
px = np.asarray(store["EX|XGBoost"]["prob"])
thx = float(store["EX|XGBoost"]["thr"])
for k in (0, 1, 2):
    v = px[yt == k]
    add(analysis="C4_primary_design", cls=LAB[k], median=float(np.median(v)),
        q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
        flagged=float(np.mean(v >= thx)), n=int(len(v)))
    print(f"  {LAB[k]:<16} n {len(v):>5}  median {np.median(v):.3f} "
          f"(IQR {np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f})  flagged at "
          f"{thx:.2f}: {100 * np.mean(v >= thx):.1f}%")
m01 = yt <= 1
m12 = yt >= 1
fns = {"AUROC potential vs general": lambda r: fauc((yt[r][m01[r]] == 1).astype(int),
                                                    px[r][m01[r]]),
       "AUROC high vs potential": lambda r: fauc((yt[r][m12[r]] == 2).astype(int),
                                                 px[r][m12[r]]),
       "AUROC elevated vs general": lambda r: fauc((yt[r] >= 1).astype(int), px[r])}
res = boot(fns, ids[te])
for k, (v, lo, hi) in res.items():
    add(analysis="C4_primary_design", metric=k, value=v, lo=lo, hi=hi)
    print(f"  {k:<30} {v:.3f} ({lo:.3f}-{hi:.3f})")

pd.DataFrame(rows).to_csv(os.path.join(RES, "three_class.csv"), index=False,
                          encoding="utf-8")
print("\nwritten: three_class.csv")
