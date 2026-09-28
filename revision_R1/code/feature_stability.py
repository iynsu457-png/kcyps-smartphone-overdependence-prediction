import io
import json
import os
import sys
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

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
sys.path.insert(0, HERE)
import _paths
import revnumbers as NB
from grouped_shap_audit_labels import group_of

SEED = 42
NBOOT = 1000
TR = ["G7->G8", "G8->G9", "G9->G10"]
SAPS_G = {"SAPS mean score", "SAPS sum score", "SAPS tolerance", "SAPS withdrawal",
          "SAPS virtual-life", "Daily smartphone use", "Baseline SAPS",
          "SAPS change from baseline", "SAPS slope", "Prior high-risk"}
DESIGN_G = {"Wave", "Cohort"}
BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
trans = df["TRANSITION"].astype(str).values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
               .split(df[feat].values, y, groups=ids))

pipe = ImbPipeline([("imp", SimpleImputer(strategy="median")),
                    ("smote_enn", SMOTEENN(random_state=SEED)),
                    ("clf", xgb.XGBClassifier(**BEST["XGBoost"], random_state=SEED,
                                              eval_metric="auc", tree_method="hist",
                                              n_jobs=-1, verbosity=0))])
pipe.fit(df[feat].iloc[dev], y[dev])
Xte = pd.DataFrame(pipe.named_steps["imp"].transform(df[feat].iloc[te]), columns=feat)
sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xte)
if isinstance(sv, list):
    sv = sv[1]
elif sv.ndim == 3:
    sv = sv[..., 1]
print(f"SHAP matrix {sv.shape}")

cols = defaultdict(list)
for c in feat:
    g = group_of(c)
    if g is not None:
        cols[g].append(feat.index(c))
GR = sorted(cols)
tte = trans[te]
grp = {}
for i, g in enumerate(ids[te]):
    grp.setdefault(g, []).append(i)
grp = {k: np.asarray(v) for k, v in grp.items()}
keys = np.array(list(grp.keys()))
rows = []

def imp_by(idx, drop_design=False):
    m = np.abs(sv[idx]).mean(axis=0)
    s_ = pd.Series({g: m[c].sum() for g, c in cols.items()})
    return s_.drop(labels=[g for g in DESIGN_G if g in s_]) if drop_design else s_

pooled = imp_by(np.arange(len(te)))
per = {t: imp_by(np.where(tte == t)[0]) for t in TR}
perd = {t: imp_by(np.where(tte == t)[0], True) for t in TR}
pooledd = imp_by(np.arange(len(te)), True)
rng = np.random.default_rng(SEED)
bs_rho = {f"{a} vs {b}": [] for a, b in combinations(TR, 2)}
bs_pool = {t: [] for t in TR}
for _ in range(NBOOT):
    r = np.concatenate([grp[k] for k in rng.choice(keys, len(keys), replace=True)])
    pr = {t: imp_by(r[tte[r] == t], True) for t in TR}
    po = imp_by(r, True)
    for a, b in combinations(TR, 2):
        bs_rho[f"{a} vs {b}"].append(spearmanr(pr[a], pr[b]).statistic)
    for t in TR:
        bs_pool[t].append(spearmanr(pr[t], po).statistic)

print("\ngroup-importance rank agreement between transitions (Spearman rho):")
for k, v in bs_rho.items():
    a, b = k.split(" vs ")
    rho = spearmanr(perd[a], perd[b]).statistic
    rows.append(dict(analysis="F2_rank_agreement", contrast=k, rho=rho,
                     lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    print(f"  {k:<22} {rho:+.3f} ({np.percentile(v, 2.5):+.3f} to "
          f"{np.percentile(v, 97.5):+.3f})")
for t in TR:
    rho = spearmanr(perd[t], pooledd).statistic
    rows.append(dict(analysis="F2_vs_pooled", contrast=f"{t} vs pooled", rho=rho,
                     lo=np.percentile(bs_pool[t], 2.5),
                     hi=np.percentile(bs_pool[t], 97.5)))
    print(f"  {t} vs pooled       {rho:+.3f} ({np.percentile(bs_pool[t], 2.5):+.3f} to "
          f"{np.percentile(bs_pool[t], 97.5):+.3f})")

print("\nshare of total attribution by kind of predictor:")
for t in TR + ["pooled"]:
    s = pooled if t == "pooled" else per[t]
    tot = s.sum()
    sh = dict(SAPS=100 * s[[g for g in GR if g in SAPS_G]].sum() / tot,
              design=100 * s[[g for g in GR if g in DESIGN_G]].sum() / tot)
    sh["psychosocial"] = 100 - sh["SAPS"] - sh["design"]
    rows.append(dict(analysis="F3_shares", contrast=t, saps_pct=sh["SAPS"],
                     design_pct=sh["design"], psychosocial_pct=sh["psychosocial"],
                     n=int((tte == t).sum()) if t != "pooled" else len(te)))
    print(f"  {t:<10} SAPS-derived {sh['SAPS']:5.1f}%   design {sh['design']:4.1f}%   "
          f"psychosocial {sh['psychosocial']:5.1f}%")

print("\ntop five groups per transition:")
for t in TR + ["pooled"]:
    s = (pooled if t == "pooled" else per[t]).sort_values(ascending=False)
    print(f"  {t:<10} " + "; ".join(f"{NB.lab(g)} {v:.3f}" for g, v in s.head(5).items()))
    for rnk, (g, v) in enumerate(s.items(), start=1):
        rows.append(dict(analysis="F1_group_importance", contrast=t, group=g,
                         label=NB.lab(g), rank=rnk, value=v))

pd.DataFrame(rows).to_csv(os.path.join(RES, "feature_stability.csv"), index=False,
                          encoding="utf-8")
print("\nwritten: feature_stability.csv")
