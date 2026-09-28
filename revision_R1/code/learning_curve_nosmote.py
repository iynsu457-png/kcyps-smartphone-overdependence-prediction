import io
import os
import sys
import json
import time
import types
import warnings
from collections import defaultdict

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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, confusion_matrix, brier_score_loss)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

SEED = 42
EPS = 1e-6
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
NEUTRAL = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
               colsample_bytree=1.0, min_child_weight=5, gamma=0.0, reg_alpha=0.0,
               reg_lambda=1.0)
CLFS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "LogRegEN"]

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
X = df[feat].values
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                random_state=SEED).split(X, y, groups=ids))
TP = pd.read_csv(_paths.TEST_PREDICTIONS)
assert (TP["ID"].astype("int64").values == ids[te]).all()

_DOM = [("YPSY1", "Self-esteem"), ("YPSY2", "Depression"), ("YPSY3", "Social anxiety"),
        ("YPSY4", "Psychological well-being"), ("YPSY5", "Peer relationship"),
        ("YPSY6", "Self-control"), ("YPSY7", "Aggression"), ("YINT", "Smartphone use"),
        ("YTIM", "Time use"), ("YMDA", "Device access"), ("YEDU", "School engagement"),
        ("YDLQ", "Delinquency"), ("YFAM", "Family relationship"),
        ("YACT", "After-school activity"), ("YFUR", "Career orientation"),
        ("YPHY", "Physical health"), ("P_PMDA", "Parental mediation"),
        ("P_PPSY", "Parenting stress"), ("P_PFAM", "Parent-reported family"),
        ("P_PEDU", "Parent education involvement"),
        ("P_PSCHOOL", "Parent school involvement"), ("P_PWORK", "Parent work hours"),
        ("P_PPHY", "Parent health"), ("P_PFBRT", "Father info"),
        ("P_PYBRT", "Mother info"), ("P_PHOM", "Household structure"),
        ("P_PINC", "Household income")]
SINGLE = ["SP_MEAN", "SP_SUM", "SP_TOLERANCE", "SP_WITHDRAWAL", "SP_VIRTUAL", "SP_DAILY",
          "SP_SUM_BASELINE", "SP_SUM_DELTA", "SP_SUM_SLOPE", "PREV_HIGH_RISK", "WAVE_IDX",
          "COHORT_SRC_e4", "YGENDER", "YTWIN", "P_PFBRTA1", "P_PHOMPOP"]
gc = defaultdict(list)
for c in feat:
    if c in SINGLE:
        gc[c].append(c)
        continue
    for pre, lab in _DOM:
        if c.startswith(pre):
            gc[lab].append(c)
            break
XS = pd.DataFrame({g: df[cs].mean(axis=1) if len(cs) > 1 else df[cs[0]]
                   for g, cs in gc.items()}).values
assert XS.shape[1] == 42, XS.shape
SCALE_ONLY = os.environ.get("LC_SCALE_ONLY") == "1"

G = {}
for i, k in enumerate(ids[te]):
    G.setdefault(k, []).append(i)
G = {k: np.asarray(v) for k, v in G.items()}
KEYS = np.array(list(G.keys()))

def resample_pipe(clf, linear=False, resample=True):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if linear:
        steps.append(("sc", StandardScaler()))
    if resample:
        steps.append(("smote_enn", SMOTEENN(random_state=SEED)))
    steps.append(("clf", clf))
    return ImbPipeline(steps) if resample else SkPipeline(steps)

def make(name):
    p = BEST[name]
    if name == "XGBoost":
        return xgb.XGBClassifier(**p, random_state=SEED, eval_metric="auc",
                                 tree_method="hist", n_jobs=-1, verbosity=0), False
    if name == "LightGBM":
        return lgb.LGBMClassifier(**p, random_state=SEED, n_jobs=-1, verbose=-1), False
    if name == "CatBoost":
        return cb.CatBoostClassifier(**p, random_seed=SEED, verbose=0,
                                     allow_writing_files=False, thread_count=-1), False
    if name == "RandomForest":
        return RandomForestClassifier(**p, random_state=SEED, n_jobs=-1), False
    if name == "ExtraTrees":
        return ExtraTreesClassifier(**p, random_state=SEED, n_jobs=-1), False
    return LogisticRegression(penalty="elasticnet", solver="saga", max_iter=3000,
                              random_state=SEED, n_jobs=-1, **p), True

print("=" * 78)
print("L1  LEARNING CURVES (held-out set fixed; development subsampled by adolescent)")
print("=" * 78)
FRACS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
REPS = 3
dev_ids = np.unique(ids[dev])
rows = []
if SCALE_ONLY:
    rows = [r for r in pd.read_csv(os.path.join(RES, "learning_curve.csv"))
            .to_dict("records") if not r["representation"].startswith("42 ")]
for frac in FRACS:
    for rep in range(REPS if frac < 1.0 else 1):
        rng = np.random.default_rng(SEED + 100 * rep + int(frac * 1000))
        keep = rng.choice(dev_ids, size=max(2, int(round(frac * len(dev_ids)))),
                          replace=False) if frac < 1.0 else dev_ids
        sub = dev[np.isin(ids[dev], keep)]
        npos = int(y[sub].sum())
        for rep_name, Xm, clf in (
                ("455 item-level columns, tuned XGBoost", X,
                 xgb.XGBClassifier(**BEST["XGBoost"], random_state=SEED,
                                   eval_metric="auc", tree_method="hist", n_jobs=-1,
                                   verbosity=0)),
                ("42 scale-score columns, neutral XGBoost", XS,
                 xgb.XGBClassifier(**NEUTRAL, random_state=SEED, eval_metric="auc",
                                   tree_method="hist", n_jobs=-1, verbosity=0))):
            if SCALE_ONLY and not rep_name.startswith("42 "):
                continue
            t0 = time.time()
            pipe = resample_pipe(clf)
            pipe.fit(Xm[sub], y[sub])
            pr = pipe.predict_proba(Xm[te])[:, 1]
            rows.append(dict(analysis="L1_learning_curve", representation=rep_name,
                             fraction=frac, rep=rep, n_train=len(sub),
                             n_pos_train=npos, AUROC=roc_auc_score(y[te], pr),
                             PR_AUC=average_precision_score(y[te], pr)))
            print(f"  {frac:4.0%} rep {rep}  n {len(sub):>5} ({npos:>3} events)  "
                  f"{rep_name[:34]:<34} AUROC {rows[-1]['AUROC']:.3f}  PR-AUC "
                  f"{rows[-1]['PR_AUC']:.3f}  [{time.time() - t0:.0f}s]")
        pd.DataFrame(rows).to_csv(os.path.join(RES, "learning_curve.csv"),
                                  index=False, encoding="utf-8")

if SCALE_ONLY:
    sys.exit(0)

print("\n" + "=" * 78)
print("N1  ALL SIX CLASSIFIERS WITHOUT SMOTE+ENN")
print("=" * 78)

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def metrics(yt, yp, thr):
    pred = (yp >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(AUROC=roc_auc_score(yt, yp), PR_AUC=average_precision_score(yt, yp),
                sensitivity=sens, specificity=spec,
                PPV=tp / (tp + fp) if tp + fp else np.nan,
                NPV=tn / (tn + fn) if tn + fn else np.nan,
                balanced_accuracy=(sens + spec) / 2,
                F1=f1_score(yt, pred, zero_division=0),
                accuracy=accuracy_score(yt, pred), Brier=brier_score_loss(yt, yp))

def cal(yt, yp):
    lp = logit(yp)
    sl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    sl.fit(lp.reshape(-1, 1), yt)
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(lp + mid)))).mean() < yt.mean():
            lo = mid
        else:
            hi = mid
    return float(sl.coef_[0][0]), (lo + hi) / 2

yt = y[te]
nrows = []
for name in CLFS:
    t0 = time.time()
    clf, linear = make(name)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(dev))
    for tr, va in skf.split(X[dev], y[dev], groups=ids[dev]):
        c2, _ = make(name)
        pp = resample_pipe(c2, linear, resample=False)
        pp.fit(X[dev][tr], y[dev][tr])
        oof[va] = pp.predict_proba(X[dev][va])[:, 1]
    grid = np.linspace(0.01, 0.99, 99)
    thr = float(grid[int(np.argmax([f1_score(y[dev], (oof >= t).astype(int),
                                              zero_division=0) for t in grid]))])
    pp = resample_pipe(clf, linear, resample=False)
    pp.fit(X[dev], y[dev])
    pr = pp.predict_proba(X[te])[:, 1]
    frozen = TP[name].values

    point = metrics(yt, pr, thr)
    slope, icpt = cal(yt, pr)
    f_slope, f_icpt = cal(yt, frozen)
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in point}
    dA, dP, dB = [], [], []
    for _ in range(2000):
        r = np.concatenate([G[k] for k in rng.choice(KEYS, len(KEYS), replace=True)])
        if len(np.unique(yt[r])) < 2:
            continue
        m = metrics(yt[r], pr[r], thr)
        for k, v in m.items():
            if np.isfinite(v):
                boot[k].append(v)
        dA.append(roc_auc_score(yt[r], pr[r]) - roc_auc_score(yt[r], frozen[r]))
        dP.append(average_precision_score(yt[r], pr[r])
                  - average_precision_score(yt[r], frozen[r]))
        dB.append(brier_score_loss(yt[r], pr[r]) - brier_score_loss(yt[r], frozen[r]))
    rec = dict(analysis="N1_no_resampling", classifier=name, threshold=thr,
               calibration_slope=slope, calibration_intercept=icpt,
               frozen_calibration_slope=f_slope, frozen_calibration_intercept=f_icpt,
               mean_predicted=float(pr.mean()), frozen_mean_predicted=float(frozen.mean()),
               delta_AUROC=float(np.mean(dA)), delta_AUROC_lo=float(np.percentile(dA, 2.5)),
               delta_AUROC_hi=float(np.percentile(dA, 97.5)),
               delta_PR_AUC=float(np.mean(dP)),
               delta_PR_AUC_lo=float(np.percentile(dP, 2.5)),
               delta_PR_AUC_hi=float(np.percentile(dP, 97.5)),
               delta_Brier=float(np.mean(dB)),
               delta_Brier_lo=float(np.percentile(dB, 2.5)),
               delta_Brier_hi=float(np.percentile(dB, 97.5)))
    for k, v in point.items():
        rec[k] = v
        rec[k + "_lo"] = float(np.percentile(boot[k], 2.5))
        rec[k + "_hi"] = float(np.percentile(boot[k], 97.5))
    nrows.append(rec)
    pd.DataFrame(nrows).to_csv(os.path.join(RES, "nosmote_results.csv"),
                               index=False, encoding="utf-8")
    print(f"  {name:<14} AUROC {point['AUROC']:.3f} (paired vs resampled "
          f"{rec['delta_AUROC']:+.3f}, {rec['delta_AUROC_lo']:+.3f} to "
          f"{rec['delta_AUROC_hi']:+.3f})  PR-AUC {point['PR_AUC']:.3f}  slope "
          f"{slope:.2f} (resampled {f_slope:.2f})  Brier {point['Brier']:.4f} "
          f"(delta {rec['delta_Brier']:+.4f})  thr {thr:.2f}  [{time.time() - t0:.0f}s]")

print(f"\nwritten: {os.path.join(RES, 'learning_curve.csv')}")
print(f"written: {os.path.join(RES, 'nosmote_results.csv')}")
