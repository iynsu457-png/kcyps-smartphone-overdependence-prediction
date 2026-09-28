import io
import json
import os
import sys
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", write_through=True)
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline as SkPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
FIG = os.path.join(REV, "figures")
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
sys.path.insert(0, HERE)
import _paths
import revnumbers as NB

SEED = 42
BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
X = df[feat].astype(float).values
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
               .split(X, y, groups=ids))
yt = y[te]
TP = pd.read_csv(_paths.TEST_PREDICTIONS)
TP.columns = [c.lstrip("﻿") for c in TP.columns]
assert (TP["ID"].astype("int64").values == ids[te]).all()
saps = df["SP_SUM"].astype(float).values
saps = np.where(np.isnan(saps), np.nanmedian(saps[dev]), saps)

nos = SkPipeline([("imp", SimpleImputer(strategy="median")),
                  ("clf", xgb.XGBClassifier(**BEST["XGBoost"], random_state=SEED,
                                            eval_metric="auc", tree_method="hist",
                                            n_jobs=-1, verbosity=0))])
nos.fit(X[dev], y[dev])
p_nos = nos.predict_proba(X[te])[:, 1]

SC = {c: TP[c].values for c in NB.CLFS}
SC["SAPS sum (no model)"] = saps[te]
SC["XGBoost without SMOTE+ENN"] = p_nos

grp = {}
for i, g in enumerate(ids[te]):
    grp.setdefault(g, []).append(i)
grp = {k: np.asarray(v) for k, v in grp.items()}
keys = np.array(list(grp.keys()))

def prec_at_recall(yy, s, r):
    p, rc, _ = precision_recall_curve(yy, s)
    ok = rc >= r
    return float(p[ok].max()) if ok.any() else np.nan

rows = []
rng = np.random.default_rng(SEED)
B = {k: [] for k in SC}
PR = {k: {r: [] for r in (0.25, 0.5, 0.75)} for k in ("XGBoost", "SAPS sum (no model)")}
D = {k: [] for k in ("SAPS sum (no model)", "XGBoost without SMOTE+ENN")}
for _ in range(2000):
    r = np.concatenate([grp[k] for k in rng.choice(keys, len(keys), replace=True)])
    yr = yt[r]
    if yr.sum() == 0:
        continue
    ap = {k: average_precision_score(yr, s[r]) for k, s in SC.items()}
    for k in SC:
        B[k].append(ap[k])
    for k in D:
        D[k].append(ap[k] - ap["XGBoost"])
    for k in PR:
        for rr in PR[k]:
            PR[k][rr].append(prec_at_recall(yr, SC[k][r], rr))

prev = yt.mean()
print(f"held-out prevalence {prev:.4f}")
for k, s in SC.items():
    a = average_precision_score(yt, s)
    rows.append(dict(analysis="P1_pr_auc", score=k, value=a,
                     lo=np.percentile(B[k], 2.5), hi=np.percentile(B[k], 97.5),
                     lift=a / prev))
    print(f"  {k:<30} PR-AUC {a:.3f} ({np.percentile(B[k], 2.5):.3f}-"
          f"{np.percentile(B[k], 97.5):.3f})  lift {a / prev:.1f}")
for k, d in D.items():
    v = average_precision_score(yt, SC[k]) - average_precision_score(yt, SC["XGBoost"])
    rows.append(dict(analysis="P1_pr_auc_diff_vs_primary", score=k, value=v,
                     lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    print(f"  {k} minus primary: {v:+.3f} ({np.percentile(d, 2.5):+.3f} to "
          f"{np.percentile(d, 97.5):+.3f})")
for k in PR:
    for rr, v in PR[k].items():
        pt = prec_at_recall(yt, SC[k], rr)
        rows.append(dict(analysis="P2_precision_at_recall", score=k, recall=rr, value=pt,
                         lo=np.nanpercentile(v, 2.5), hi=np.nanpercentile(v, 97.5)))
        print(f"  {k:<22} precision at recall {rr:.2f}: {pt:.3f} "
              f"({np.nanpercentile(v, 2.5):.3f}-{np.nanpercentile(v, 97.5):.3f})")

FS = pd.read_pickle(os.path.join(RES, "transitions_fullspectrum.pkl"))
FEATS = [c for c in FS.columns if c not in ("y_bin3", "y_ord", "sp_t1", "ID", "TRANSITION")]
fid = FS["ID"].astype("int64").values
fy = FS["y_bin3"].astype(int).values
fdev, fte = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
                 .split(FS[FEATS].values, fy, groups=fid))
assert (np.load(os.path.join(RES, "fs_test_index.npy")) == fte).all()
store = json.load(open(os.path.join(RES, "fs_test_probabilities.json")))
pf = np.asarray(store["FS|XGBoost"]["prob"])
fyt = fy[fte]
g2 = {}
for i, g in enumerate(fid[fte]):
    g2.setdefault(g, []).append(i)
g2 = {k: np.asarray(v) for k, v in g2.items()}
k2 = np.array(list(g2.keys()))
rng = np.random.default_rng(SEED)
bb = []
for _ in range(2000):
    r = np.concatenate([g2[k] for k in rng.choice(k2, len(k2), replace=True)])
    if fyt[r].sum():
        bb.append(average_precision_score(fyt[r], pf[r]))
a_fs = average_precision_score(fyt, pf)
rows.append(dict(analysis="P3_full_spectrum", score="XGBoost, full-spectrum model",
                 value=a_fs, lo=np.percentile(bb, 2.5), hi=np.percentile(bb, 97.5),
                 prevalence=fyt.mean(), lift=a_fs / fyt.mean()))
print(f"  full spectrum (prevalence {fyt.mean():.4f}): PR-AUC {a_fs:.3f} "
      f"({np.percentile(bb, 2.5):.3f}-{np.percentile(bb, 97.5):.3f}), lift "
      f"{a_fs / fyt.mean():.1f}")

PERF = NB.load_performance()
T = PERF["thr_table"]
for lab in ("F1-optimal (frozen, submitted analysis)", "sensitivity of at least 0.50",
            "sensitivity of at least 0.70", "sensitivity of at least 0.80"):
    r = T[T.label == lab].iloc[0]
    hr = 1000 * prev
    tp, fn = r.sensitivity * hr, (1 - r.sensitivity) * hr
    fp = (1 - r.specificity) * (1000 - hr)
    rows.append(dict(analysis="P4_per_1000", score=lab, threshold=r.threshold,
                     high_risk=hr, detected=tp, missed=fn, false_positive=fp,
                     flagged=tp + fp, ppv=r.PPV, sensitivity=r.sensitivity))
    print(f"  {lab:<42} per 1,000: high risk {hr:.0f}, detected {tp:.0f}, missed "
          f"{fn:.0f}, flagged {tp + fp:.0f} of whom {fp:.0f} not high risk")

pd.DataFrame(rows).to_csv(os.path.join(RES, "pr_analysis.csv"), index=False,
                          encoding="utf-8")

COL = {"XGBoost": "#D62728", "LightGBM": "#1F77B4", "CatBoost": "#2CA02C",
       "RandomForest": "#FF7F0E", "ExtraTrees": "#9467BD", "LogRegEN": "#7F7F7F"}
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
for c in NB.CLFS:
    p, rc, _ = precision_recall_curve(yt, SC[c])
    a = average_precision_score(yt, SC[c])
    ax[0].plot(rc, p, color=COL[c], lw=2.2 if c == "XGBoost" else 1.3,
               label=f"{NB.PRETTY[c].replace('Logistic Regression', 'logistic regression')}"
                     f" ({a:.3f})", drawstyle="steps-post")
p, rc, _ = precision_recall_curve(yt, SC["SAPS sum (no model)"])
ax[0].plot(rc, p, color="black", ls="--", lw=1.3,
           label=f"SAPS sum, no model ({average_precision_score(yt, saps[te]):.3f})",
           drawstyle="steps-post")
for c, lab, st in (("XGBoost", "Primary model, with SMOTE+ENN", "-"),
                   ("XGBoost without SMOTE+ENN", "Refitted without SMOTE+ENN", "--")):
    p, rc, _ = precision_recall_curve(yt, SC[c])
    ax[1].plot(rc, p, color="#D62728", ls=st, lw=2.0, drawstyle="steps-post",
               label=f"{lab} ({average_precision_score(yt, SC[c]):.3f})")
for lab, mk in (("F1-optimal (frozen, submitted analysis)", "o"),
                ("sensitivity of at least 0.50", "s"), ("sensitivity of at least 0.80", "^")):
    r = T[T.label == lab].iloc[0]
    ax[1].plot(r.sensitivity, r.PPV, marker=mk, color="black", ms=8, ls="",
               label=f"Operating point: {lab.replace(' (frozen, submitted analysis)', '')}"
                     f" (threshold {r.threshold:.2f})")
for a_, t_ in zip(ax, ("A  Six classifiers and the SAPS score alone",
                       "B  Primary model: resampling and operating points")):
    a_.axhline(prev, color="grey", ls=":", lw=1)
    a_.text(0.01, prev - 0.035, f"prevalence {prev:.3f}", ha="left", fontsize=8,
            color="grey")
    a_.set_xlim(0, 1)
    a_.set_ylim(0, 1)
    a_.set_xlabel("Recall (sensitivity)")
    a_.set_ylabel("Precision (positive predictive value)")
    a_.set_title(t_, loc="left", fontsize=11)
    a_.legend(fontsize=8, frameon=False, loc="upper right")
    a_.grid(alpha=0.25)
fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, "Figure8_PrecisionRecall.png"), dpi=300, facecolor="white")
print("written: pr_analysis.csv, Figure8_PrecisionRecall.png")
