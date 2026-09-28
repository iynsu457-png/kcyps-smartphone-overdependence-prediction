import io
import os
import re
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
RAW = os.path.join(BUNDLE, "01_data", "raw")
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
E = pd.read_csv(os.path.join(RES, "provenance_encoding.csv"))
flagged = E[E.likely_nominal].feature.tolist()

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                random_state=SEED).split(df[feat].values, y, groups=ids))

print("=" * 78)
print("E1  WHICH FLAGGED COLUMNS ARE GENUINELY NOMINAL, FROM THE RAW VALUE LABELS")
print("=" * 78)
ORDINAL_CUES = ("전혀", "별로", "보통", "약간", "매우", "그렇", "않다", "없다", "있다",
                "never", "rarely", "sometimes", "often", "always", "매일", "주",
                "시간", "이상", "미만", "정도", "낮", "높")
labels = {}
try:
    import pyreadstat
    _, meta = pyreadstat.read_sav(
        os.path.join(RAW, "KCYPS2018m1[SPSS]", "KCYPS2018m1Yw1.sav"), metadataonly=True)
    vvl_y = meta.variable_value_labels
    _, metap = pyreadstat.read_sav(
        os.path.join(RAW, "KCYPS2018m1[SPSS]", "KCYPS2018m1Pw1.sav"), metadataonly=True)
    vvl_p = metap.variable_value_labels
    for c in flagged:
        code = (c[2:] if c.startswith("P_") else c) + "w1"
        src = vvl_p if c.startswith("P_") else vvl_y
        labels[c] = src.get(code)
except Exception as exc:
    print("  value labels unavailable:", exc)

rows = []
truly = []
for c in flagged:
    lab = labels.get(c)
    if not lab:
        verdict = "unlabelled, treated as nominal"
        truly.append(c)
    else:
        txt = " ".join(str(v) for v in lab.values())
        ordinal_like = any(k in txt for k in ORDINAL_CUES)
        verdict = ("labelled, ordered wording" if ordinal_like
                   else "labelled, unordered categories")
        if not ordinal_like:
            truly.append(c)
    rows.append(dict(analysis="E1_label_review", feature=c,
                     n_labels=0 if not lab else len(lab),
                     verdict=verdict,
                     example_labels="; ".join(list(map(str, (lab or {}).values()))[:4])))
print(f"  {len(flagged)} columns were flagged by the prefix heuristic")
print(f"  {len(truly)} remain nominal after reading the value labels")
for r in rows:
    print(f"    {r['feature']:16s} {r['n_labels']:>3} labels  {r['verdict']:<32} "
          f"{r['example_labels'][:52]}")

print("\n" + "=" * 78)
print("E2  ONE-HOT EXPERIMENT ON THE GENUINELY NOMINAL COLUMNS")
print("=" * 78)
X = df[feat].copy()
if truly:
    parts = [X.drop(columns=truly)]
    for c in truly:
        d = pd.get_dummies(X[c].astype("category"), prefix=c, dummy_na=True, dtype=float)
        parts.append(d)
    Xoh = pd.concat(parts, axis=1)
else:
    Xoh = X.copy()
print(f"  integer coding: {X.shape[1]} columns; one-hot: {Xoh.shape[1]} columns "
      f"({Xoh.shape[1] - X.shape[1]:+d})")

G = {}
for i, k in enumerate(ids[te]):
    G.setdefault(k, []).append(i)
G = {k: np.asarray(v) for k, v in G.items()}
KEYS = np.array(list(G.keys()))

def fit(Xm, linear):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if linear:
        steps.append(("sc", StandardScaler()))
        clf = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=3000,
                                 random_state=SEED, n_jobs=-1, **BEST["LogRegEN"])
    else:
        clf = xgb.XGBClassifier(**BEST["XGBoost"], random_state=SEED, eval_metric="auc",
                                tree_method="hist", n_jobs=-1, verbosity=0)
    steps += [("smote_enn", SMOTEENN(random_state=SEED)), ("clf", clf)]
    pipe = ImbPipeline(steps)
    A = Xm.values if hasattr(Xm, "values") else Xm
    pipe.fit(A[dev], y[dev])
    return pipe.predict_proba(A[te])[:, 1]

def paired(pa, pb, n=2000):
    rng = np.random.default_rng(SEED)
    da, dp = [], []
    for _ in range(n):
        r = np.concatenate([G[k] for k in rng.choice(KEYS, len(KEYS), replace=True)])
        if len(np.unique(y[te][r])) < 2:
            continue
        da.append(roc_auc_score(y[te][r], pa[r]) - roc_auc_score(y[te][r], pb[r]))
        dp.append(average_precision_score(y[te][r], pa[r])
                  - average_precision_score(y[te][r], pb[r]))
    return (float(np.mean(da)), float(np.percentile(da, 2.5)),
            float(np.percentile(da, 97.5)), float(np.mean(dp)),
            float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5)))

for name, linear in (("XGBoost", False), ("Elastic-net Logistic Regression", True)):
    t0 = time.time()
    p_int = fit(X, linear)
    p_oh = fit(Xoh, linear)
    a_i, a_o = roc_auc_score(y[te], p_int), roc_auc_score(y[te], p_oh)
    pr_i = average_precision_score(y[te], p_int)
    pr_o = average_precision_score(y[te], p_oh)
    d = paired(p_oh, p_int)
    print(f"  {name:<32} integer {a_i:.3f} / PR {pr_i:.3f}   one-hot {a_o:.3f} / PR "
          f"{pr_o:.3f}")
    print(f"      one-hot minus integer: AUROC {d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f})"
          f"   PR-AUC {d[3]:+.3f} ({d[4]:+.3f} to {d[5]:+.3f})  [{time.time()-t0:.0f}s]")
    rows.append(dict(analysis="E2_onehot", classifier=name,
                     n_columns_integer=int(X.shape[1]), n_columns_onehot=int(Xoh.shape[1]),
                     AUROC_integer=a_i, AUROC_onehot=a_o,
                     PR_AUC_integer=pr_i, PR_AUC_onehot=pr_o,
                     delta_AUROC=d[0], delta_AUROC_lo=d[1], delta_AUROC_hi=d[2],
                     delta_PR_AUC=d[3], delta_PR_AUC_lo=d[4], delta_PR_AUC_hi=d[5]))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "encoding_experiment.csv"),
                              index=False, encoding="utf-8")

print(f"\nwritten: {os.path.join(RES, 'encoding_experiment.csv')}")
