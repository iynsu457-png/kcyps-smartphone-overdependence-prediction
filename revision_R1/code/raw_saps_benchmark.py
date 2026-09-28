import io
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score)
from sklearn.model_selection import GroupShuffleSplit

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", write_through=True)

SEED = 42
EPS = 1e-6
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
y_all = df["y"].astype(int).values
ids_all = df["ID"].astype("int64").values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                random_state=SEED).split(df[feat].values, y_all,
                                                         groups=ids_all))

TP = pd.read_csv(_paths.TEST_PREDICTIONS)
assert len(TP) == len(te), (len(TP), len(te))
assert (TP["ID"].astype("int64").values == ids_all[te]).all(), "row order mismatch"
y = TP["y_true"].astype(int).values
gid = TP["ID"].astype("int64").values
full = TP["XGBoost"].values
PREV = y.mean()

saps_dev = df["SP_SUM"].values[dev]
saps_te = df["SP_SUM"].values[te]
prev_hr_te = df["PREV_HIGH_RISK"].values[te].astype(float)

print("=" * 78)
print("THE SAPS SCORE ITSELF AS A COMPARATOR (no model, frozen split)")
print("=" * 78)
print(f"held-out n = {len(y):,}; high risk {int(y.sum())} ({100 * PREV:.2f} %)")
print(f"missing SAPS sum at wave t in the held-out set: {int(np.isnan(saps_te).sum())}")

med = np.nanmedian(saps_dev)
xd = np.where(np.isnan(saps_dev), med, saps_dev).reshape(-1, 1)
xt = np.where(np.isnan(saps_te), med, saps_te).reshape(-1, 1)
platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
platt.fit(xd, y_all[dev])
saps_prob = platt.predict_proba(xt)[:, 1]
score = np.where(np.isnan(saps_te), med, saps_te)

def _groups(g):
    out = {}
    for i, k in enumerate(g):
        out.setdefault(k, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}

G = _groups(gid)
K = np.array(list(G.keys()))

def boot(fn, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r = np.concatenate([G[k] for k in rng.choice(K, len(K), replace=True)])
        if len(np.unique(y[r])) < 2:
            continue
        out.append(fn(r))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def net_benefit(yt, prob, pt):
    pred = prob >= pt
    tp = np.sum(pred & (yt == 1))
    fp = np.sum(pred & (yt == 0))
    return tp / len(yt) - (fp / len(yt)) * (pt / (1 - pt))

rows = []
COMPARATORS = [
    ("FULL", "Primary 455-feature model (frozen held-out predictions)", full),
    ("M0", "SAPS 15-item sum at wave t, used directly as the score", score),
    ("M0b", "Prior-wave high-risk classification, used directly", prev_hr_te),
]
for k, lab, sc in COMPARATORS:
    a = roc_auc_score(y, sc)
    p = average_precision_score(y, sc)
    a_lo, a_hi = boot(lambda r, sc=sc: roc_auc_score(y[r], sc[r]))
    p_lo, p_hi = boot(lambda r, sc=sc: average_precision_score(y[r], sc[r]))
    rows.append(dict(analysis="M0_benchmark", model=k, label=lab,
                     AUROC=a, AUROC_lo=a_lo, AUROC_hi=a_hi,
                     PR_AUC=p, PR_AUC_lo=p_lo, PR_AUC_hi=p_hi))
    print(f"  {k:<5} AUROC {a:.3f} ({a_lo:.3f}-{a_hi:.3f})   "
          f"PR-AUC {p:.3f} ({p_lo:.3f}-{p_hi:.3f})   {lab}")

print("\npaired differences, full model minus comparator (same resamples):")
for k, lab, sc in COMPARATORS[1:]:
    da = roc_auc_score(y, full) - roc_auc_score(y, sc)
    dp = average_precision_score(y, full) - average_precision_score(y, sc)
    da_lo, da_hi = boot(lambda r, sc=sc: roc_auc_score(y[r], full[r])
                        - roc_auc_score(y[r], sc[r]))
    dp_lo, dp_hi = boot(lambda r, sc=sc: average_precision_score(y[r], full[r])
                        - average_precision_score(y[r], sc[r]))
    rows.append(dict(analysis="M0_delta_full_minus", model=k, label=lab,
                     AUROC=da, AUROC_lo=da_lo, AUROC_hi=da_hi,
                     PR_AUC=dp, PR_AUC_lo=dp_lo, PR_AUC_hi=dp_hi))
    print(f"  vs {k:<4} AUROC {da:+.3f} ({da_lo:+.3f} to {da_hi:+.3f})   "
          f"PR-AUC {dp:+.3f} ({dp_lo:+.3f} to {dp_hi:+.3f})")

print("\nPlatt-scaled SAPS sum (map fitted on the development set only):")
lp = np.log(np.clip(saps_prob, EPS, 1 - EPS) / (1 - np.clip(saps_prob, EPS, 1 - EPS)))
sl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
sl.fit(lp.reshape(-1, 1), y)
lo, hi = -10.0, 10.0
for _ in range(200):
    mid = (lo + hi) / 2
    if (1 / (1 + np.exp(-(lp + mid)))).mean() < y.mean():
        lo = mid
    else:
        hi = mid
br = brier_score_loss(y, saps_prob)
print(f"  calibration slope {float(sl.coef_[0][0]):.3f}   intercept "
      f"{(lo + hi) / 2:+.3f}   Brier {br:.4f}   "
      f"Brier skill {1 - br / (PREV * (1 - PREV)):+.3f}")
rows.append(dict(analysis="M0_calibration", model="M0",
                 label="Platt-scaled SAPS sum at wave t",
                 calibration_slope=float(sl.coef_[0][0]),
                 calibration_intercept=(lo + hi) / 2, Brier=br,
                 Brier_skill=1 - br / (PREV * (1 - PREV))))

print("\nnet benefit of the Platt-scaled SAPS sum:")
for pt in (0.05, 0.10, 0.20, 0.30):
    nb = net_benefit(y, saps_prob, pt)
    nb_all = PREV - (1 - PREV) * (pt / (1 - pt))
    print(f"  threshold probability {pt:.2f}: SAPS score {nb:+.4f}   "
          f"screen everyone {nb_all:+.4f}")
    rows.append(dict(analysis="M0_decision_curve", model="M0",
                     label=f"threshold probability {pt:.2f}",
                     net_benefit=nb, net_benefit_treat_all=nb_all))

thr_full = 0.30
flagged = (full >= thr_full).mean()
cut = float(np.quantile(score, 1 - flagged))
print(f"\nat a matched flagged fraction of {100 * flagged:.1f} % "
      f"(SAPS sum cut-off {cut:.0f}):")
for k, sc, t in (("FULL", full, thr_full), ("M0", score, cut)):
    pred = (sc >= t).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn_) if tp + fn_ else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    print(f"  {k:<5} sensitivity {sens:.3f}  specificity "
          f"{tn / (tn + fp):.3f}  PPV {ppv:.3f}  F1 "
          f"{f1_score(y, pred, zero_division=0):.3f}")
    rows.append(dict(analysis="M0_matched_flag", model=k,
                     label=f"flagged {100 * flagged:.1f} %", threshold=t,
                     sensitivity=sens, specificity=tn / (tn + fp), PPV=ppv,
                     F1=f1_score(y, pred, zero_division=0)))

pd.DataFrame(rows).to_csv(os.path.join(RES, "raw_saps_benchmark.csv"),
                          index=False, encoding="utf-8")
print(f"\nwritten: {os.path.join(RES, 'raw_saps_benchmark.csv')}")
