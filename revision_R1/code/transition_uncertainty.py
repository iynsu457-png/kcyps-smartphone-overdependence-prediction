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
from scipy.stats import rankdata, chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

SEED = 42
NBOOT = 2000
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
CLFS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "LogRegEN"]
TR = ["G7->G8", "G8->G9", "G9->G10"]
PAIRS = [("G8->G9", "G7->G8"), ("G9->G10", "G7->G8"), ("G8->G9", "G9->G10")]
OUTCOME_YEAR = {("m1", "G7->G8"): 2019, ("m1", "G8->G9"): 2020, ("m1", "G9->G10"): 2021,
                ("e4", "G7->G8"): 2022, ("e4", "G8->G9"): 2023, ("e4", "G9->G10"): 2024}
TRAJ = ["SP_SUM_BASELINE", "SP_SUM_DELTA", "SP_SUM_SLOPE"]

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
X = df[feat].astype(float).values
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
trans = df["TRANSITION"].astype(str).values
cohort = np.where(df["COHORT_SRC_e4"].astype(int).values == 1, "e4", "m1")
saps = df["SP_SUM"].astype(float).values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                random_state=SEED).split(X, y, groups=ids))
saps_raw = saps.copy()
saps = np.where(np.isnan(saps), np.nanmedian(saps[dev]), saps)
TP = pd.read_csv(_paths.TEST_PREDICTIONS)
TP.columns = [c.lstrip("﻿") for c in TP.columns]
assert (TP["ID"].astype("int64").values == ids[te]).all()
assert (TP["y_true"].astype(int).values == y[te]).all()
OOF = np.load(os.path.join(RES, "oof_dev_xgboost.npy"))
LOC = np.load(os.path.join(RES, "local_test_xgboost.npy"))
rows = []

def fast_auc(yt, s):
    pos = yt == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def groups(gid):
    out = {}
    for i, g in enumerate(gid):
        out.setdefault(g, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}

def strat_boot(yv, scores, gid, strata, levels, tag, extra=None):
    extra = extra or {}
    grp = groups(gid)
    keys = np.array(list(grp.keys()))
    rng = np.random.default_rng(SEED)
    pt = {(sn, lv): fast_auc(yv[strata == lv], s[strata == lv])
          for sn, s in scores.items() for lv in levels}
    bs = {k: [] for k in pt}
    for _ in range(NBOOT):
        r = np.concatenate([grp[k] for k in rng.choice(keys, len(keys), replace=True)])
        yr, sr = yv[r], strata[r]
        for sn, s in scores.items():
            ss = s[r]
            for lv in levels:
                m = sr == lv
                bs[(sn, lv)].append(fast_auc(yr[m], ss[m]))
    bs = {k: np.asarray(v) for k, v in bs.items()}
    for sn in scores:
        for lv in levels:
            m = strata == lv
            b = bs[(sn, lv)]
            rows.append(dict(analysis=tag, score=sn, stratum=lv, contrast="",
                             n=int(m.sum()), events=int(yv[m].sum()),
                             AUROC=pt[(sn, lv)], lo=np.nanpercentile(b, 2.5),
                             hi=np.nanpercentile(b, 97.5), se=np.nanstd(b, ddof=1),
                             **extra))
        D = []
        for a, c in PAIRS:
            d = bs[(sn, a)] - bs[(sn, c)]
            D.append(d)
            p = 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))
            se = np.nanstd(d, ddof=1)
            rows.append(dict(analysis=tag + "_diff", score=sn, stratum="",
                             contrast=f"{a} minus {c}", AUROC=pt[(sn, a)] - pt[(sn, c)],
                             lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                             se=se, p=min(1.0, p), mdd80=2.80 * se, **extra))
        M = np.vstack(D[:2]).T
        M = M[~np.isnan(M).any(axis=1)]
        dv = np.array([pt[(sn, PAIRS[0][0])] - pt[(sn, PAIRS[0][1])],
                       pt[(sn, PAIRS[1][0])] - pt[(sn, PAIRS[1][1])]])
        Q = float(dv @ np.linalg.solve(np.cov(M.T), dv))
        rows.append(dict(analysis=tag + "_wald", score=sn, stratum="",
                         contrast="equal AUROC across the three transitions",
                         AUROC=Q, p=float(chi2.sf(Q, 2)), **extra))
    return pt

def show(tag, sn, **flt):
    R = pd.DataFrame(rows)
    sel = (R.score == sn)
    for k, v in flt.items():
        sel &= (R[k] == v)
    for r in R[(R.analysis == tag) & sel].itertuples():
        print(f"    {r.stratum:<8} n {r.n:>5,}  events {r.events:>3}  AUROC {r.AUROC:.3f} "
              f"({r.lo:.3f}-{r.hi:.3f})")
    for r in R[(R.analysis == tag + "_diff") & sel].itertuples():
        print(f"    {r.contrast:<22} {r.AUROC:+.3f} ({r.lo:+.3f} to {r.hi:+.3f})  "
              f"SE {r.se:.3f}  p {r.p:.2f}  detectable (80 %) {r.mdd80:.3f}")
    for r in R[(R.analysis == tag + "_wald") & sel].itertuples():
        print(f"    Wald test of equal AUROC across transitions: chi2(2) {r.AUROC:.2f}, "
              f"p {r.p:.2f}")

def pipe():
    return ImbPipeline([("imp", SimpleImputer(strategy="median")),
                        ("smote_enn", SMOTEENN(random_state=SEED)),
                        ("clf", xgb.XGBClassifier(**BEST["XGBoost"],
                                                  random_state=SEED, eval_metric="auc",
                                                  tree_method="hist", n_jobs=-1,
                                                  verbosity=0))])

print("=" * 78)
print("T1  EVENT COUNTS BY TRANSITION AND BY COHORT x TRANSITION")
print("=" * 78)
for part, idx in (("analytic", np.arange(len(y))), ("development", dev), ("test", te)):
    for t in TR:
        for c in ("all", "m1", "e4"):
            m = trans[idx] == t
            if c != "all":
                m = m & (cohort[idx] == c)
            n, e = int(m.sum()), int(y[idx][m].sum())
            rows.append(dict(analysis="T1_counts", partition=part, stratum=t, cohort=c,
                             outcome_year=OUTCOME_YEAR.get((c, t), np.nan), n=n,
                             events=e, prevalence=100 * e / n,
                             saps_mean=float(np.nanmean(saps_raw[idx][m])),
                             saps_sd=float(np.nanstd(saps_raw[idx][m], ddof=1))))
            print(f"  {part:<12} {t:<8} {c:<4} n {n:>5,}  events {e:>4}  "
                  f"({100 * e / n:.2f} %)")

print("\n" + "=" * 78)
print("T2  HELD-OUT SET: PER-TRANSITION AUROC AND BETWEEN-TRANSITION DIFFERENCES")
print("=" * 78)
sc = {c: TP[c].values for c in CLFS}
sc["SAPS sum (no model)"] = saps[te]
strat_boot(y[te], sc, ids[te], trans[te], TR, "T2_test")
for sn in ("XGBoost", "SAPS sum (no model)"):
    print(f"  {sn}")
    show("T2_test", sn)

print("\n" + "=" * 78)
print("T3  DEVELOPMENT SET, 5-FOLD OUT-OF-FOLD PREDICTIONS (about 140 events each)")
print("=" * 78)
strat_boot(y[dev], {"XGBoost": OOF, "SAPS sum (no model)": saps[dev]}, ids[dev],
           trans[dev], TR, "T3_oof")
for sn in ("XGBoost", "SAPS sum (no model)"):
    print(f"  {sn}")
    show("T3_oof", sn)

print("\n" + "=" * 78)
print("T4  COHORT x TRANSITION")
print("=" * 78)
print("  (a) development out-of-fold, within each cohort")
for c in ("m1", "e4"):
    m = cohort[dev] == c
    strat_boot(y[dev][m], {"XGBoost": OOF[m], "SAPS sum (no model)": saps[dev][m]},
               ids[dev][m], trans[dev][m], TR, "T4a_oof_cohort", dict(cohort=c))
    print(f"  cohort {c}")
    show("T4a_oof_cohort", "XGBoost", cohort=c)

print("  (b) cohort holdout: trained on one whole cohort, scored on the other")
for tr_c, te_c in (("m1", "e4"), ("e4", "m1")):
    t0 = time.time()
    tri, tei = np.where(cohort == tr_c)[0], np.where(cohort == te_c)[0]
    p = pipe()
    p.fit(X[tri], y[tri])
    pv = p.predict_proba(X[tei])[:, 1]
    strat_boot(y[tei], {"XGBoost": pv, "SAPS sum (no model)": saps[tei]}, ids[tei],
               trans[tei], TR, "T4b_holdout", dict(cohort=te_c, train_cohort=tr_c))
    print(f"  train {tr_c} -> test {te_c}  [{time.time() - t0:.0f}s]")
    show("T4b_holdout", "XGBoost", cohort=te_c)
    print("   SAPS sum")
    show("T4b_holdout", "SAPS sum (no model)", cohort=te_c)

print("  (c) held-out test set, frozen predictions, by cohort (very few events)")
for c in ("m1", "e4"):
    m = cohort[te] == c
    strat_boot(y[te][m], {"XGBoost": TP["XGBoost"].values[m]}, ids[te][m],
               trans[te][m], TR, "T4c_test_cohort", dict(cohort=c))
    print(f"  cohort {c}")
    show("T4c_test_cohort", "XGBoost", cohort=c)

print("\n" + "=" * 78)
print("T5  TRAJECTORY FEATURES AND THE FIRST TRANSITION")
print("=" * 78)
for t in TR:
    m = trans == t
    rows.append(dict(analysis="T5_trajectory_values", stratum=t,
                     delta_nonzero=float(pd.Series(df["SP_SUM_DELTA"].values[m])
                                         .dropna().ne(0).mean()),
                     slope_nonzero=float(pd.Series(df["SP_SUM_SLOPE"].values[m])
                                         .dropna().ne(0).mean()),
                     baseline_equals_current=float(
                         (pd.Series(df["SP_SUM_BASELINE"].values[m])
                          == pd.Series(saps_raw[m]))[~np.isnan(saps_raw[m])].mean())))
    print(f"  {t:<8} SAPS change non-zero {100 * rows[-1]['delta_nonzero']:.1f} %, "
          f"baseline equal to current score "
          f"{100 * rows[-1]['baseline_equals_current']:.1f} %")
keep = [i for i, c in enumerate(feat) if c not in TRAJ]
t0 = time.time()
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_nt = np.zeros(len(dev))
for tr, va in skf.split(X[dev], y[dev], groups=ids[dev]):
    p = pipe()
    p.fit(X[dev][tr][:, keep], y[dev][tr])
    oof_nt[va] = p.predict_proba(X[dev][va][:, keep])[:, 1]
p = pipe()
p.fit(X[dev][:, keep], y[dev])
te_nt = p.predict_proba(X[te][:, keep])[:, 1]
print(f"  refitted without {', '.join(TRAJ)}  [{time.time() - t0:.0f}s]")
for part, yv, full, nt, gid, st in (("development OOF", y[dev], OOF, oof_nt, ids[dev],
                                     trans[dev]),
                                    ("held-out", y[te], LOC, te_nt, ids[te], trans[te])):
    grp = groups(gid)
    keys = np.array(list(grp.keys()))
    rng = np.random.default_rng(SEED)
    bs = {t: [] for t in TR}
    for _ in range(NBOOT):
        r = np.concatenate([grp[k] for k in rng.choice(keys, len(keys), replace=True)])
        for t in TR:
            m = st[r] == t
            bs[t].append(fast_auc(yv[r][m], full[r][m]) - fast_auc(yv[r][m], nt[r][m]))
    for t in TR:
        m = st == t
        a_f, a_n = fast_auc(yv[m], full[m]), fast_auc(yv[m], nt[m])
        b = np.asarray(bs[t])
        rows.append(dict(analysis="T5_ablation", partition=part, stratum=t,
                         events=int(yv[m].sum()), AUROC_full=a_f, AUROC_no_traj=a_n,
                         AUROC=a_f - a_n, lo=np.nanpercentile(b, 2.5),
                         hi=np.nanpercentile(b, 97.5)))
        print(f"  {part:<16} {t:<8} full {a_f:.3f}  without trajectory {a_n:.3f}  "
              f"loss {a_f - a_n:+.3f} ({rows[-1]['lo']:+.3f} to {rows[-1]['hi']:+.3f})")

OUT = os.path.join(RES, "transition_uncertainty.csv")
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
print(f"\nwritten: {OUT}")
