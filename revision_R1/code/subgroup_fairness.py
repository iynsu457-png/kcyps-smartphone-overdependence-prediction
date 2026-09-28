import io
import os
import sys
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", write_through=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
NBOOT = 2000
EPS = 1e-6
THR = 0.30
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
               .split(df[feat].values, y, groups=ids))
TP = pd.read_csv(_paths.TEST_PREDICTIONS)
TP.columns = [c.lstrip("﻿") for c in TP.columns]
assert (TP["ID"].astype("int64").values == ids[te]).all()
OOF = np.load(os.path.join(RES, "oof_dev_xgboost.npy"))
LOC = np.load(os.path.join(RES, "local_test_xgboost.npy"))

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
platt.fit(logit(OOF).reshape(-1, 1), y[dev])
REC_TE = platt.predict_proba(logit(LOC).reshape(-1, 1))[:, 1]
REC_DEV = platt.predict_proba(logit(OOF).reshape(-1, 1))[:, 1]

sex = pd.Series(df["YGENDER"].values).map({1.0: "Male", 2.0: "Female"}).fillna("")
sex = np.asarray(sex, dtype=str)
coh = np.where(df["COHORT_SRC_e4"].astype(int).values == 1, "e4", "m1")
inc_code = df["P_PINCOME"].replace({9999: np.nan}).astype(float).values
inc = np.select([inc_code <= 6, inc_code <= 9, inc_code <= 12],
                ["< 5 million KRW", "5 to < 8 million KRW", ">= 8 million KRW"], "")
inc[np.isnan(inc_code)] = ""
ses_code = df["P_PINCOME1"].replace({9999: np.nan}).astype(float).values
ses = np.select([ses_code <= 2, ses_code == 3, ses_code >= 4],
                ["Low", "Middle", "High"], "")
ses[np.isnan(ses_code)] = ""
rel = df[["P_PFRELATION2", "P_PFRELATION3", "P_PFRELATION4"]].values
fam = np.where((rel == 1).any(axis=1) & (rel == 2).any(axis=1), "Two-parent household",
               "Other household")
fam[np.isnan(rel[:, 0])] = ""
trn = df["TRANSITION"].astype(str).str.replace("->", " to ").values
inc, ses, fam, trn = (np.asarray(v, dtype=str) for v in (inc, ses, fam, trn))
GROUPS = [("Sex", sex, ["Male", "Female"]),
          ("Cohort", coh, ["m1", "e4"]),
          ("Household income", inc, ["< 5 million KRW", "5 to < 8 million KRW",
                                     ">= 8 million KRW"]),
          ("Subjective economic status", ses, ["Low", "Middle", "High"]),
          ("Family structure", fam, ["Two-parent household", "Other household"]),
          ("Grade transition", trn, ["G7 to G8", "G8 to G9", "G9 to G10"])]

def fast_auc(yt, s):
    pos = yt == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def ap(yt, s):
    if yt.sum() == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    yy = yt[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())

def thr_metrics(yt, s):
    pr = s >= THR
    tp = int((pr & (yt == 1)).sum())
    fn = int((~pr & (yt == 1)).sum())
    fp = int((pr & (yt == 0)).sum())
    tn = int((~pr & (yt == 0)).sum())
    return dict(sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                specificity=tn / (tn + fp) if tn + fp else np.nan,
                PPV=tp / (tp + fp) if tp + fp else np.nan,
                flagged=100 * (tp + fp) / len(yt))

def cal_slope(yt, p):
    if yt.sum() < 2:
        return np.nan
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    m.fit(logit(p).reshape(-1, 1), yt)
    return float(m.coef_[0][0])

def all_metrics(yt, s, rec):
    d = dict(AUROC=fast_auc(yt, s), PR_AUC=ap(yt, s), **thr_metrics(yt, s),
             OE=yt.sum() / rec.sum(), mean_pred=rec.mean(), observed=yt.mean())
    return d

def groups_of(gid):
    out = {}
    for i, g in enumerate(gid):
        out.setdefault(g, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}

rows = []

def run(part, idx, score, rec):
    yv, gid = y[idx], ids[idx]
    grp = groups_of(gid)
    keys = np.array(list(grp.keys()))
    for gname, gvec, levels in GROUPS:
        gv = gvec[idx]
        pt = {lv: all_metrics(yv[gv == lv], score[gv == lv], rec[gv == lv])
              for lv in levels}
        slope = {lv: cal_slope(yv[gv == lv], rec[gv == lv]) for lv in levels}
        rng = np.random.default_rng(SEED)
        bs = {lv: {k: [] for k in pt[lv]} for lv in levels}
        for _ in range(NBOOT):
            r = np.concatenate([grp[k] for k in rng.choice(keys, len(keys), replace=True)])
            yr, sr, rr, gr = yv[r], score[r], rec[r], gv[r]
            for lv in levels:
                m = gr == lv
                if yr[m].sum() == 0:
                    for k in bs[lv]:
                        bs[lv][k].append(np.nan)
                    continue
                for k, v in all_metrics(yr[m], sr[m], rr[m]).items():
                    bs[lv][k].append(v)
        for lv in levels:
            m = gv == lv
            rec_ = dict(analysis=part, group=gname, level=lv, n=int(m.sum()),
                        events=int(yv[m].sum()), calibration_slope=slope[lv])
            for k, v in pt[lv].items():
                b = np.asarray(bs[lv][k], dtype=float)
                rec_[k] = v
                rec_[k + "_lo"] = np.nanpercentile(b, 2.5)
                rec_[k + "_hi"] = np.nanpercentile(b, 97.5)
            rows.append(rec_)
            print(f"  {part:<9} {gname:<27} {lv:<22} n {int(m.sum()):>5,} ev "
                  f"{int(yv[m].sum()):>3}  AUROC {pt[lv]['AUROC']:.3f} "
                  f"({rec_['AUROC_lo']:.3f}-{rec_['AUROC_hi']:.3f})  sens "
                  f"{pt[lv]['sensitivity']:.3f}  PPV {pt[lv]['PPV']:.3f}  O/E "
                  f"{pt[lv]['OE']:.2f} ({rec_['OE_lo']:.2f}-{rec_['OE_hi']:.2f})  "
                  f"slope {slope[lv]:.2f}")
        ref = levels[0]
        for lv in levels[1:]:
            for k in ("AUROC", "sensitivity", "OE"):
                a = np.asarray(bs[lv][k], dtype=float)
                b = np.asarray(bs[ref][k], dtype=float)
                d = a - b
                p = 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))
                rows.append(dict(analysis=part + "_diff", group=gname, level=lv,
                                 reference=ref, metric=k,
                                 value=pt[lv][k] - pt[ref][k],
                                 lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                                 p=min(1.0, p)))
                if k == "AUROC":
                    print(f"      {lv} minus {ref}: AUROC {rows[-1]['value']:+.3f} "
                          f"({rows[-1]['lo']:+.3f} to {rows[-1]['hi']:+.3f}), "
                          f"p {rows[-1]['p']:.2f}")

print("=" * 78)
print("S1  HELD-OUT SET (frozen predictions; calibration after Platt recalibration)")
print("=" * 78)
run("held-out", te, TP["XGBoost"].values, REC_TE)
print("\n" + "=" * 78)
print("S2  DEVELOPMENT SET, 5-FOLD OUT-OF-FOLD")
print("=" * 78)
run("dev OOF", dev, OOF, REC_DEV)

OUT = os.path.join(RES, "subgroup_fairness.csv")
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
print(f"\nwritten: {OUT}")
