import io
import os
import sys
import json
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, confusion_matrix, brier_score_loss,
                             matthews_corrcoef, balanced_accuracy_score)
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

SEED = 42
N_BOOT = 2000
EPS = 1e-6
PRIMARY = "XGBoost"

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
FIGS = os.path.join(REV, "figures")
os.makedirs(FIGS, exist_ok=True)
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
THR = json.load(open(os.path.join(BUNDLE, "03_results", "thresholds.json")))
CLFS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "LogRegEN"]
PRETTY = {"XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost",
          "RandomForest": "Random Forest", "ExtraTrees": "Extra Trees",
          "LogRegEN": "Elastic-net Logistic Regression"}

TP = pd.read_csv(_paths.TEST_PREDICTIONS)
y = TP["y_true"].astype(int).values
gid = TP["ID"].astype("int64").values
PREV = y.mean()

print("=" * 78)
print("EXTENDED PERFORMANCE, CALIBRATION, AND DECISION-CURVE ANALYSIS")
print("=" * 78)
print(f"held-out test set n = {len(y):,} transitions / {pd.Series(gid).nunique():,} "
      f"adolescents; high risk {int(y.sum())} ({100 * PREV:.2f} %)")
print("all primary results are computed from the frozen held-out predictions\n")

def _groups(g):
    out = {}
    for i, k in enumerate(g):
        out.setdefault(k, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}

GROUPS = _groups(gid)
KEYS = np.array(list(GROUPS.keys()))

def boot_rows(rng):
    return np.concatenate([GROUPS[k] for k in rng.choice(KEYS, len(KEYS), replace=True)])

def cluster_ci(fn, n_boot=N_BOOT, seed=SEED):
    point = fn(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in point}
    for _ in range(n_boot):
        rows = boot_rows(rng)
        if len(np.unique(y[rows])) < 2:
            continue
        v = fn(rows)
        for k in point:
            if v.get(k) is not None and np.isfinite(v[k]):
                acc[k].append(v[k])
    return {k: (point[k],
                float(np.percentile(acc[k], 2.5)) if acc[k] else np.nan,
                float(np.percentile(acc[k], 97.5)) if acc[k] else np.nan)
            for k in point}

def full_metrics(prob, thr):
    def fn(rows):
        yt, yp = y[rows], prob[rows]
        pred = (yp >= thr).astype(int)
        tn, fp, fn_, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn_) if tp + fn_ else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        base = yt.mean()
        brier = brier_score_loss(yt, yp)
        return dict(AUROC=roc_auc_score(yt, yp),
                    PR_AUC=average_precision_score(yt, yp),
                    sensitivity=sens, specificity=spec,
                    PPV=tp / (tp + fp) if tp + fp else np.nan,
                    NPV=tn / (tn + fn_) if tn + fn_ else np.nan,
                    balanced_accuracy=(sens + spec) / 2,
                    F1=f1_score(yt, pred, zero_division=0),
                    accuracy=accuracy_score(yt, pred),
                    MCC=matthews_corrcoef(yt, pred),
                    Brier=brier,
                    Brier_skill=1 - brier / (base * (1 - base)))
    return fn

print("=" * 78)
print("P1  EXTENDED METRICS AT THE FROZEN THRESHOLD (frozen predictions)")
print("=" * 78)
rows = []
for c in CLFS:
    m = cluster_ci(full_metrics(TP[c].values, THR[c]))
    rows.append(dict(analysis="P1_extended", classifier=c, threshold=THR[c],
                     **{k: v[0] for k, v in m.items()},
                     **{f"{k}_lo": v[1] for k, v in m.items()},
                     **{f"{k}_hi": v[2] for k, v in m.items()}))
    print(f"{PRETTY[c]:<32} PR-AUC {m['PR_AUC'][0]:.3f} "
          f"({m['PR_AUC'][1]:.3f}-{m['PR_AUC'][2]:.3f})  "
          f"PPV {m['PPV'][0]:.3f}  NPV {m['NPV'][0]:.3f}  "
          f"bal.acc {m['balanced_accuracy'][0]:.3f}  Brier {m['Brier'][0]:.4f}  "
          f"skill {m['Brier_skill'][0]:+.3f}")
pd.DataFrame(rows).to_csv(os.path.join(RES, "performance_extended.csv"),
                          index=False, encoding="utf-8")

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def cal_measures(prob):
    lp = logit(prob)

    def fn(rows):
        yt, x = y[rows], lp[rows]
        slope = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        slope.fit(x.reshape(-1, 1), yt)
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(x + mid)))).mean() < yt.mean():
                lo = mid
            else:
                hi = mid
        return dict(calibration_slope=float(slope.coef_[0][0]),
                    calibration_intercept=float((lo + hi) / 2),
                    mean_predicted=float(prob[rows].mean()),
                    observed=float(yt.mean()))
    return fn

print("\n" + "=" * 78)
print("P2  CALIBRATION OF THE PRIMARY MODEL ON THE UNTOUCHED TEST SET")
print("=" * 78)
cal = cluster_ci(cal_measures(TP[PRIMARY].values))
for k, v in cal.items():
    print(f"  {k:<24} {v[0]:>8.4f}  ({v[1]:.4f} to {v[2]:.4f})")
print(f"  observed-to-expected ratio "
      f"{cal['observed'][0] / cal['mean_predicted'][0]:.3f}")

cal_rows = [dict(analysis="P2_calibration", classifier=PRIMARY, quantity=k,
                 value=v[0], lo=v[1], hi=v[2]) for k, v in cal.items()]

prob = TP[PRIMARY].values
edges = np.quantile(prob, np.linspace(0, 1, 11))
edges[0], edges[-1] = -np.inf, np.inf
binid = np.digitize(prob, edges[1:-1])
rng = np.random.default_rng(SEED)
boot_obs = {b: [] for b in range(10)}
for _ in range(N_BOOT):
    rows = boot_rows(rng)
    for b in range(10):
        m = binid[rows] == b
        if m.sum() >= 5:
            boot_obs[b].append(y[rows][m].mean())
bins = []
for b in range(10):
    m = binid == b
    obs = y[m].mean()
    lo = float(np.percentile(boot_obs[b], 2.5)) if boot_obs[b] else np.nan
    hi = float(np.percentile(boot_obs[b], 97.5)) if boot_obs[b] else np.nan
    bins.append(dict(analysis="P2_reliability", classifier=PRIMARY, bin=b + 1,
                     n=int(m.sum()), mean_predicted=float(prob[m].mean()),
                     observed=float(obs), observed_lo=lo, observed_hi=hi))
    print(f"  bin {b + 1:>2}  n {int(m.sum()):>4}  predicted "
          f"{prob[m].mean():.3f}  observed {obs:.3f} ({lo:.3f}-{hi:.3f})")
cal_rows += bins

print("\n" + "=" * 78)
print("P3  RECALIBRATION FITTED ON DEVELOPMENT OUT-OF-FOLD PREDICTIONS")
print("=" * 78)
df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
Xa = df[feat].astype(float).values
ya = df["y"].astype(int).values
ia = df["ID"].astype("int64").values
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
dev, te = next(gss.split(Xa, ya, groups=ia))

def pipe():
    return ImbPipeline([("imp", SimpleImputer(strategy="median")),
                        ("smote_enn", SMOTEENN(random_state=SEED)),
                        ("clf", xgb.XGBClassifier(**BEST[PRIMARY], random_state=SEED,
                                                  eval_metric="auc",
                                                  tree_method="hist", n_jobs=-1,
                                                  verbosity=0))])

oof = np.zeros(len(dev))
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for tr, va in skf.split(Xa[dev], ya[dev], groups=ia[dev]):
    p = pipe()
    p.fit(Xa[dev][tr], ya[dev][tr])
    oof[va] = p.predict_proba(Xa[dev][va])[:, 1]
p = pipe()
p.fit(Xa[dev], ya[dev])
loc_test = p.predict_proba(Xa[te])[:, 1]
np.save(os.path.join(RES, "oof_dev_xgboost.npy"), oof)
np.save(os.path.join(RES, "local_test_xgboost.npy"), loc_test)

y_te_local = ya[te]
g_te_local = ia[te]
print(f"locally refitted model: development out-of-fold AUROC "
      f"{roc_auc_score(ya[dev], oof):.4f}; held-out AUROC "
      f"{roc_auc_score(y_te_local, loc_test):.4f} "
      f"(the frozen value is {roc_auc_score(y, TP[PRIMARY].values):.4f}; the "
      "difference is the documented library-version effect)")

platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
platt.fit(logit(oof).reshape(-1, 1), ya[dev])
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof, ya[dev])
rc_platt = platt.predict_proba(logit(loc_test).reshape(-1, 1))[:, 1]
rc_iso = iso.predict(loc_test)

Gl = _groups(g_te_local)
Kl = np.array(list(Gl.keys()))

def local_ci(fn, n_boot=500, seed=SEED):
    point = fn(np.arange(len(y_te_local)))
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in point}
    for _ in range(n_boot):
        rows = np.concatenate([Gl[k] for k in rng.choice(Kl, len(Kl), replace=True)])
        if len(np.unique(y_te_local[rows])) < 2:
            continue
        v = fn(rows)
        for k in point:
            if np.isfinite(v[k]):
                acc[k].append(v[k])
    return {k: (point[k], float(np.percentile(acc[k], 2.5)),
                float(np.percentile(acc[k], 97.5))) for k in point}

def local_cal(prob):
    lp = logit(prob)

    def fn(rows):
        yt, x = y_te_local[rows], lp[rows]
        sl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        sl.fit(x.reshape(-1, 1), yt)
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(x + mid)))).mean() < yt.mean():
                lo = mid
            else:
                hi = mid
        return dict(calibration_slope=float(sl.coef_[0][0]),
                    calibration_intercept=float((lo + hi) / 2),
                    Brier=brier_score_loss(yt, prob[rows]),
                    AUROC=roc_auc_score(yt, prob[rows]),
                    mean_predicted=float(prob[rows].mean()))
    return fn

for label, pr in (("uncalibrated (locally refitted)", loc_test),
                  ("Platt scaling on development out-of-fold", rc_platt),
                  ("isotonic on development out-of-fold", rc_iso)):
    m = local_ci(local_cal(pr))
    print(f"  {label:<42} slope {m['calibration_slope'][0]:.3f} "
          f"intercept {m['calibration_intercept'][0]:+.3f} "
          f"Brier {m['Brier'][0]:.4f} AUROC {m['AUROC'][0]:.3f} "
          f"mean predicted {m['mean_predicted'][0]:.3f}")
    for k, v in m.items():
        cal_rows.append(dict(analysis="P3_recalibration", classifier=PRIMARY,
                             variant=label, quantity=k, value=v[0], lo=v[1], hi=v[2]))
pd.DataFrame(cal_rows).to_csv(os.path.join(RES, "calibration.csv"),
                              index=False, encoding="utf-8")

print("\n" + "=" * 78)
print("P4  DECISION-CURVE ANALYSIS")
print("=" * 78)

def net_benefit(yt, prob, pt):
    pred = prob >= pt
    n = len(yt)
    tp = np.sum(pred & (yt == 1))
    fp = np.sum(pred & (yt == 0))
    return tp / n - (fp / n) * (pt / (1 - pt))

pts = np.round(np.arange(0.01, 0.51, 0.01), 2)
dca = []
for pt in pts:
    row = dict(threshold_probability=pt,
               net_benefit_treat_all=PREV - (1 - PREV) * (pt / (1 - pt)),
               net_benefit_treat_none=0.0,
               net_benefit_model_frozen=net_benefit(y, TP[PRIMARY].values, pt),
               net_benefit_model_recalibrated=net_benefit(y_te_local, rc_platt, pt))
    dca.append(row)
D = pd.DataFrame(dca)
D.to_csv(os.path.join(RES, "decision_curve.csv"), index=False, encoding="utf-8")
for pt in (0.05, 0.10, 0.20, 0.30):
    r = D[D.threshold_probability == pt].iloc[0]
    print(f"  threshold probability {pt:.2f}: model (recalibrated) "
          f"{r.net_benefit_model_recalibrated:+.4f}  treat-all "
          f"{r.net_benefit_treat_all:+.4f}  treat-none 0")
best_pt = D.loc[D.net_benefit_model_recalibrated.idxmax()]
print(f"  highest net benefit at a threshold probability of "
      f"{best_pt.threshold_probability:.2f} "
      f"({best_pt.net_benefit_model_recalibrated:+.4f})")
cross = D[(D.net_benefit_model_recalibrated > D.net_benefit_treat_all)
          & (D.net_benefit_model_recalibrated > 0)]
if len(cross):
    print(f"  the model gives the highest net benefit between threshold "
          f"probabilities {cross.threshold_probability.min():.2f} and "
          f"{cross.threshold_probability.max():.2f}")

print("\n" + "=" * 78)
print("P5  PERFORMANCE ACROSS PLAUSIBLE THRESHOLDS (primary model, frozen)")
print("=" * 78)
prob = TP[PRIMARY].values
rows = []

def at(thr, label):
    pred = (prob >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn_) if tp + fn_ else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    return dict(analysis="P5_thresholds", label=label, threshold=thr,
                flagged=int(tp + fp), flagged_pct=100 * (tp + fp) / len(y),
                sensitivity=sens, specificity=spec, PPV=ppv,
                NPV=tn / (tn + fn_) if tn + fn_ else np.nan,
                balanced_accuracy=(sens + spec) / 2,
                F1=f1_score(y, pred, zero_division=0),
                number_needed_to_screen=(tp + fp) / tp if tp else np.nan,
                net_benefit_at_own_threshold=net_benefit(y, prob, max(thr, 0.01)))

rows.append(at(THR[PRIMARY], "F1-optimal (frozen, submitted analysis)"))
for target in (0.50, 0.70, 0.80, 0.90):
    order = np.sort(prob[y == 1])
    thr_t = float(order[max(0, int(np.ceil((1 - target) * len(order))) - 1)])
    rows.append(at(thr_t, f"sensitivity of at least {target:.2f}"))
for thr in (0.05, 0.10, 0.20, 0.40, 0.50):
    rows.append(at(thr, f"fixed threshold {thr:.2f}"))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(RES, "threshold_analysis.csv"), index=False, encoding="utf-8")
print(T[["label", "threshold", "sensitivity", "specificity", "PPV",
         "flagged_pct", "number_needed_to_screen"]].to_string(index=False,
                                                              float_format="%.3f"))

B = pd.DataFrame(bins)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].plot([0, B.mean_predicted.max() * 1.05], [0, B.mean_predicted.max() * 1.05],
           "k--", lw=1, label="Perfect calibration")
ax[0].errorbar(B.mean_predicted, B.observed,
               yerr=[B.observed - B.observed_lo, B.observed_hi - B.observed],
               fmt="o-", color="#1f4e79", capsize=3, lw=1.4, ms=5,
               label="Frozen predictions")
ax[0].set_xlabel("Mean predicted probability")
ax[0].set_ylabel("Observed proportion high risk")
ax[0].set_title("A  Calibration, primary model, held-out test set")
ax[0].legend(frameon=False, fontsize=9)
ax[0].spines[["top", "right"]].set_visible(False)

ax[1].plot(D.threshold_probability, D.net_benefit_model_recalibrated,
           color="#1f4e79", lw=1.8, label="Model, recalibrated")
ax[1].plot(D.threshold_probability, D.net_benefit_model_frozen,
           color="#7fa8d0", lw=1.4, ls=":", label="Model, uncalibrated")
ax[1].plot(D.threshold_probability, D.net_benefit_treat_all,
           color="#999999", lw=1.2, ls="--", label="Screen everyone")
ax[1].axhline(0, color="k", lw=1, label="Screen no one")
ax[1].set_ylim(min(-0.01, D.net_benefit_model_recalibrated.min()),
               max(0.02, D.net_benefit_model_recalibrated.max() * 1.25))
ax[1].set_xlabel("Threshold probability")
ax[1].set_ylabel("Net benefit")
ax[1].set_title("B  Decision curve, primary model")
ax[1].legend(frameon=False, fontsize=9)
ax[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext, dpi in (("png", 300),):
    fig.savefig(os.path.join(FIGS, f"Figure6_Calibration_DecisionCurve.{ext}"),
                dpi=dpi, bbox_inches="tight")
plt.close(fig)
print(f"\nwritten: {os.path.join(FIGS, 'Figure6_Calibration_DecisionCurve.png')}")
print(f"written: {os.path.join(RES, 'performance_extended.csv')}")
