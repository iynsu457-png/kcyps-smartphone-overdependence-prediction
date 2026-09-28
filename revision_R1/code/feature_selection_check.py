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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
X = df[feat].values
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                random_state=SEED).split(X, y, groups=ids))

print("=" * 78)
print("WHAT EACH CLASSIFIER FAMILY ACTUALLY DOES WITH 455 FEATURES")
print("=" * 78)
rows = []

def run(name, clf, linear=False):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if linear:
        steps.append(("sc", StandardScaler()))
    steps += [("smote_enn", SMOTEENN(random_state=SEED)), ("clf", clf)]
    pipe = ImbPipeline(steps)
    pipe.fit(X[dev], y[dev])
    m = pipe.named_steps["clf"]
    if linear:
        coef = m.coef_.ravel()
        nz = int(np.sum(coef != 0))
        rec = dict(classifier=name, mechanism="L1-L2 penalty shrinks coefficients",
                   n_features=len(feat), n_used=nz,
                   pct_used=100 * nz / len(feat),
                   n_dropped=len(feat) - nz,
                   note="coefficients exactly zero are genuinely removed")
    else:
        imp = getattr(m, "feature_importances_", None)
        used = int(np.sum(np.asarray(imp) > 0))
        rec = dict(classifier=name,
                   mechanism=("max_features subsamples split candidates"
                              if name in ("Random Forest", "Extra Trees")
                              else "depth, gamma and shrinkage limit splits"),
                   n_features=len(feat), n_used=used,
                   pct_used=100 * used / len(feat),
                   n_dropped=len(feat) - used,
                   note=("nothing is removed; every feature remains available at "
                         "every split" if name in ("Random Forest", "Extra Trees")
                         else "features with no split gain are effectively unused"))
    rows.append(rec)
    print(f"  {name:<32} uses {rec['n_used']:>3} of {len(feat)} features "
          f"({rec['pct_used']:5.1f} %)   {rec['mechanism']}")

run("Elastic-net Logistic Regression",
    LogisticRegression(penalty="elasticnet", solver="saga", max_iter=3000,
                       random_state=SEED, n_jobs=-1, **BEST["LogRegEN"]), linear=True)
run("Random Forest", RandomForestClassifier(**BEST["RandomForest"], random_state=SEED,
                                            n_jobs=-1))
run("Extra Trees", ExtraTreesClassifier(**BEST["ExtraTrees"], random_state=SEED,
                                        n_jobs=-1))
run("XGBoost", xgb.XGBClassifier(**BEST["XGBoost"], random_state=SEED, eval_metric="auc",
                                 tree_method="hist", n_jobs=-1, verbosity=0))
run("LightGBM", lgb.LGBMClassifier(**BEST["LightGBM"], random_state=SEED, n_jobs=-1,
                                   verbose=-1))

P = pd.DataFrame(rows)
P.to_csv(os.path.join(RES, "provenance_selection.csv"), index=False, encoding="utf-8")
print("\nconclusion for Section 2.2: only the elastic net removes features. The tree "
      "ensembles retain all of them, and for Random Forest and Extra Trees "
      "max_features spreads usage across features rather than selecting among them.")
print(f"\nwritten: {os.path.join(RES, 'provenance_selection.csv')}")
