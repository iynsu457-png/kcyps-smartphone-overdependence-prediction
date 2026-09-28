import io
import json
import os
import sys
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

from collections import defaultdict

import numpy as np
import pandas as pd
import pyreadstat
import shap
import xgboost as xgb
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

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
KS = [10, 20, 30, 50, 100, 200]
VARS = [0.80, 0.95]
BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
X = df[feat].astype(float).values
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
               .split(X, y, groups=ids))
LOC = np.load(os.path.join(RES, "local_test_xgboost.npy"))
OOF = np.load(os.path.join(RES, "oof_dev_xgboost.npy"))
rows, irows = [], []

def clf():
    return xgb.XGBClassifier(**BEST["XGBoost"], random_state=SEED, eval_metric="auc",
                             tree_method="hist", n_jobs=-1, verbosity=0)

def pipe(pca=None):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if pca is not None:
        steps += [("sc", StandardScaler()), ("pca", PCA(n_components=pca,
                                                        svd_solver="full",
                                                        random_state=SEED))]
    steps += [("smote_enn", SMOTEENN(random_state=SEED)), ("clf", clf())]
    return ImbPipeline(steps)

def fauc(yy, s):
    pos = yy == 1
    r = rankdata(s)
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())

def rank_cols(Xtr, ytr):
    p = pipe()
    p.fit(Xtr, ytr)
    g = p.named_steps["clf"].get_booster().get_score(importance_type="gain")
    imp = np.zeros(Xtr.shape[1])
    for k, v in g.items():
        imp[int(k[1:])] = v
    return np.argsort(-imp)

print("=" * 78)
print("D1/D2  NESTED FEATURE SELECTION AND PCA, DEVELOPMENT OUT-OF-FOLD")
print("=" * 78)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {f"top {k}": np.zeros(len(dev)) for k in KS}
oof.update({f"PCA {int(100 * v)}%": np.zeros(len(dev)) for v in VARS})
ncomp = {f"PCA {int(100 * v)}%": [] for v in VARS}
t0 = time.time()
for fold, (tr, va) in enumerate(skf.split(X[dev], y[dev], groups=ids[dev]), start=1):
    Xtr, ytr, Xva = X[dev][tr], y[dev][tr], X[dev][va]
    order = rank_cols(Xtr, ytr)
    for k in KS:
        p = pipe()
        p.fit(Xtr[:, order[:k]], ytr)
        oof[f"top {k}"][va] = p.predict_proba(Xva[:, order[:k]])[:, 1]
    for v in VARS:
        p = pipe(pca=v)
        p.fit(Xtr, ytr)
        oof[f"PCA {int(100 * v)}%"][va] = p.predict_proba(Xva)[:, 1]
        ncomp[f"PCA {int(100 * v)}%"].append(int(p.named_steps["pca"].n_components_))
    print(f"  fold {fold} done  [{time.time() - t0:.0f}s]")

print("\n" + "=" * 78)
print("D1/D2  HELD-OUT")
print("=" * 78)
order = rank_cols(X[dev], y[dev])
held = {}
for k in KS:
    p = pipe()
    p.fit(X[dev][:, order[:k]], y[dev])
    held[f"top {k}"] = p.predict_proba(X[te][:, order[:k]])[:, 1]
for v in VARS:
    p = pipe(pca=v)
    p.fit(X[dev], y[dev])
    held[f"PCA {int(100 * v)}%"] = p.predict_proba(X[te])[:, 1]
    ncomp[f"PCA {int(100 * v)}% held-out"] = int(p.named_steps["pca"].n_components_)

yt, yd = y[te], y[dev]
grp = {}
for i, g in enumerate(ids[te]):
    grp.setdefault(g, []).append(i)
grp = {k: np.asarray(v) for k, v in grp.items()}
keys = np.array(list(grp.keys()))
rng = np.random.default_rng(SEED)
BS = {k: [] for k in held}
BF = []
for _ in range(2000):
    r = np.concatenate([grp[k] for k in rng.choice(keys, len(keys), replace=True)])
    full = fauc(yt[r], LOC[r])
    BF.append(full)
    for k, s in held.items():
        BS[k].append((fauc(yt[r], s[r]), full, average_precision_score(yt[r], s[r])))
full_a, full_p = fauc(yt, LOC), average_precision_score(yt, LOC)
rows.append(dict(analysis="D_full", variant="all 455 columns", n_inputs=455,
                 epv=424 / 455, oof_auroc=fauc(yd, OOF),
                 oof_prauc=average_precision_score(yd, OOF), auroc=full_a,
                 auroc_lo=np.percentile(BF, 2.5), auroc_hi=np.percentile(BF, 97.5),
                 prauc=full_p))
print(f"  all 455 columns        OOF {fauc(yd, OOF):.3f}  held-out {full_a:.3f} "
      f"PR-AUC {full_p:.3f}")
for k, s in held.items():
    b = np.asarray(BS[k])
    n_in = int(k.split()[1]) if k.startswith("top") else \
        ncomp[f"{k} held-out"]
    a, pa = fauc(yt, s), average_precision_score(yt, s)
    d = b[:, 0] - b[:, 1]
    rows.append(dict(analysis="D_reduced", variant=k, n_inputs=n_in, epv=424 / n_in,
                     n_inputs_folds=(str(ncomp[k]) if k.startswith("PCA") else ""),
                     oof_auroc=fauc(yd, oof[k]),
                     oof_prauc=average_precision_score(yd, oof[k]),
                     auroc=a, auroc_lo=np.percentile(b[:, 0], 2.5),
                     auroc_hi=np.percentile(b[:, 0], 97.5), prauc=pa,
                     prauc_lo=np.percentile(b[:, 2], 2.5),
                     prauc_hi=np.percentile(b[:, 2], 97.5), diff=a - full_a,
                     diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5)))
    print(f"  {k:<22} inputs {n_in:>3} (EPV {424 / n_in:5.1f})  OOF "
          f"{fauc(yd, oof[k]):.3f}  held-out {a:.3f} ({np.percentile(b[:, 0], 2.5):.3f}-"
          f"{np.percentile(b[:, 0], 97.5):.3f})  PR-AUC {pa:.3f}  vs full "
          f"{a - full_a:+.3f} ({np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})")
top_cols = [feat[i] for i in order[:30]]
rows.append(dict(analysis="D_top30_columns", variant=";".join(top_cols)))

print("\n" + "=" * 78)
print("D3  ITEMS WITHIN SCALES")
print("=" * 78)
p = pipe()
p.fit(df[feat].iloc[dev], y[dev])
Xte = df[feat].iloc[te]
sub = np.random.default_rng(SEED).choice(len(Xte), size=1000, replace=False)
Xs = pd.DataFrame(p.named_steps["imp"].transform(Xte.iloc[sub]), columns=feat)
sv = shap.TreeExplainer(p.named_steps["clf"]).shap_values(Xs)
if isinstance(sv, list):
    sv = sv[1]
ma = np.abs(sv).mean(axis=0)
item = pd.Series(ma, index=feat)
frozen = pd.read_csv(os.path.join(BUNDLE, "03_results", "shap_top25_xgboost.csv"),
                     encoding="utf-8-sig")
frozen.columns = ["code", "frozen"]
ov = frozen.merge(item.rename("refit"), left_on="code", right_index=True)
rho = spearmanr(ov.frozen, ov.refit).statistic
top25_refit = set(item.sort_values(ascending=False).head(25).index)
overlap = len(top25_refit & set(frozen.code))
print(f"  frozen top-25 vs refit: Spearman rho {rho:.3f}; {overlap} of 25 shared")
irows.append(dict(kind="agreement", rho=rho, overlap_top25=overlap))

_, my = pyreadstat.read_sav(os.path.join(BUNDLE, "01_data", "raw", "KCYPS2018m1[SPSS]",
                                         "KCYPS2018m1Yw1.sav"), metadataonly=True)
_, mp = pyreadstat.read_sav(os.path.join(BUNDLE, "01_data", "raw", "KCYPS2018m1[SPSS]",
                                         "KCYPS2018m1Pw1.sav"), metadataonly=True)

def label(c):
    k = (c[2:] if c.startswith("P_") else c) + "w1"
    m = mp if c.startswith("P_") else my
    return m.column_names_to_labels.get(k, "(derived)")

groups = defaultdict(list)
for c in feat:
    g = group_of(c)
    if g is not None:
        groups[g].append(c)
for g, cs in groups.items():
    if len(cs) < 2:
        continue
    v = item[cs].sort_values(ascending=False)
    tot = v.sum()
    cum = v.cumsum() / tot
    irows.append(dict(kind="group", group=g, label=NB.lab(g), n_items=len(cs),
                      summed=tot, top1_share=v.iloc[0] / tot,
                      top3_share=v.iloc[:3].sum() / tot,
                      items_for_80pct=int((cum < 0.80).sum() + 1)))
    for rnk, (c, val) in enumerate(v.head(3).items(), start=1):
        irows.append(dict(kind="top_item", group=g, label=NB.lab(g), rank=rnk, code=c,
                          mean_abs_shap=val, share=val / tot, item_label=label(c)))
G = pd.DataFrame([r for r in irows if r["kind"] == "group"]).sort_values("summed",
                                                                         ascending=False)
for r in G.head(10).itertuples():
    print(f"  {r.label:<42} items {r.n_items:>3}  top1 {100 * r.top1_share:4.1f}%  "
          f"top3 {100 * r.top3_share:4.1f}%  items for 80% {r.items_for_80pct}")
for r in frozen.itertuples():
    irows.append(dict(kind="frozen_top25", code=r.code, mean_abs_shap=r.frozen,
                      group=group_of(r.code), label=NB.lab(group_of(r.code) or ""),
                      item_label=label(r.code)))

pd.DataFrame(rows).to_csv(os.path.join(RES, "dim_reduction.csv"), index=False,
                          encoding="utf-8")
pd.DataFrame(irows).to_csv(os.path.join(RES, "item_shap.csv"), index=False,
                           encoding="utf-8")
print("\nwritten: dim_reduction.csv, item_shap.csv")
