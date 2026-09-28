import io
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

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
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                random_state=SEED).split(df[feat].values, y, groups=ids))
phi = y[dev].mean()
n_dev, e_dev = len(dev), int(y[dev].sum())

print("=" * 78)
print("S1  MINIMUM SAMPLE SIZE (Riley et al., 2020)")
print("=" * 78)
ln_null_per = phi * np.log(phi) + (1 - phi) * np.log(1 - phi)
max_r2 = 1 - np.exp(2 * ln_null_per)

oof = np.load(os.path.join(RES, "oof_dev_xgboost.npy"))
loc = np.load(os.path.join(RES, "local_test_xgboost.npy"))

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
platt.fit(logit(oof).reshape(-1, 1), y[dev])
p = np.clip(platt.predict_proba(logit(loc).reshape(-1, 1))[:, 1], EPS, 1 - EPS)
yt = y[te]
ln_model = np.sum(yt * np.log(p) + (1 - yt) * np.log(1 - p))
pt = yt.mean()
ln_null = len(yt) * (pt * np.log(pt) + (1 - pt) * np.log(1 - pt))
r2_emp = 1 - np.exp(-2 * (ln_model - ln_null) / len(yt))
r2_cons = 0.15 * max_r2
print(f"development: n {n_dev:,}, events {e_dev}, outcome proportion {phi:.4f}; "
      f"maximum Cox-Snell R2 {max_r2:.4f}")
print(f"anticipated Cox-Snell R2: empirical {r2_emp:.4f} (recalibrated primary model on "
      f"held-out data); conservative default {r2_cons:.4f}")

def riley(P, r2):
    S = 0.9
    n1 = P / ((S - 1) * np.log(1 - r2 / S))
    S2 = r2 / (r2 + 0.05 * max_r2)
    n2 = P / ((S2 - 1) * np.log(1 - r2 / S2))
    n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
    n = int(np.ceil(max(n1, n2, n3)))
    return dict(n1=n1, n2=n2, n3=n3, n_required=n, events_required=n * phi,
                epp_required=n * phi / P, S2=S2)

rows = []
for P, lab in ((455, "455 item-level columns (primary model)"),
               (42, "42 scale-score columns"),
               (10, "10 SAPS-derived columns"),
               (2, "2 columns: SAPS sum and prior classification")):
    for r2_lab, r2 in (("empirical", r2_emp), ("conservative", r2_cons)):
        r = riley(P, r2)
        rows.append(dict(analysis="S1_sample_size", parameters=P, label=lab,
                         r2_basis=r2_lab, r2_cs=r2, max_r2_cs=max_r2,
                         n_required=r["n_required"], events_required=r["events_required"],
                         epp_required=r["epp_required"], criterion1_n=r["n1"],
                         criterion2_n=r["n2"], criterion3_n=r["n3"],
                         n_available=n_dev, events_available=e_dev,
                         epp_available=e_dev / P,
                         ratio_available_to_required=n_dev / r["n_required"]))
        print(f"  {lab:<44} R2 {r2_lab:<12} requires n {r['n_required']:>8,} "
              f"({r['events_required']:>7,.0f} events, EPP {r['epp_required']:5.1f}); "
              f"available {n_dev:,} ({e_dev} events, EPP {e_dev / P:5.2f}); "
              f"ratio {n_dev / r['n_required']:.2f}")
pd.DataFrame(rows).to_csv(os.path.join(RES, "sample_size.csv"), index=False,
                          encoding="utf-8")

print("\n" + "=" * 78)
print("S2  TRANSITIONS AND HIGH-RISK TRANSITIONS IN EVERY FOLD")
print("=" * 78)
frows = []
Xd, yd, gd = df[feat].values[dev], y[dev], ids[dev]
for scheme, k in (("inner 3-fold hyperparameter search", 3),
                  ("outer 5-fold threshold derivation", 5)):
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    for i, (tr, va) in enumerate(skf.split(Xd, yd, groups=gd), start=1):
        frows.append(dict(scheme=scheme, repeat=1, fold=i, n_train=len(tr),
                          n_pos_train=int(yd[tr].sum()), n_val=len(va),
                          n_pos_val=int(yd[va].sum())))
for rs, rep in ((42, 1), (43, 2), (44, 3)):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rs)
    for i, (tr, va) in enumerate(skf.split(Xd, yd, groups=gd), start=1):
        frows.append(dict(scheme="repeated grouped cross-validation (model selection)",
                          repeat=rep, fold=i, n_train=len(tr),
                          n_pos_train=int(yd[tr].sum()), n_val=len(va),
                          n_pos_val=int(yd[va].sum())))
FR = pd.DataFrame(frows)
FR.to_csv(os.path.join(RES, "fold_composition.csv"), index=False, encoding="utf-8")
for sch, g in FR.groupby("scheme", sort=False):
    print(f"  {sch:<52} validation events per fold {int(g.n_pos_val.min())}-"
          f"{int(g.n_pos_val.max())} (training {int(g.n_pos_train.min())}-"
          f"{int(g.n_pos_train.max())}); validation n {int(g.n_val.min()):,}-"
          f"{int(g.n_val.max()):,}")
print(f"  held-out test set: n {len(te):,}, events {int(y[te].sum())}")

print("\n" + "=" * 78)
print("S3  HYPERPARAMETER SEARCH SPACES AND SELECTED VALUES")
print("=" * 78)
SPACES = {
    "XGBoost": [("n_estimators", "integer 200-1200"), ("max_depth", "integer 3-10"),
                ("learning_rate", "0.001-0.3, log"), ("subsample", "0.5-1.0"),
                ("colsample_bytree", "0.5-1.0"), ("min_child_weight", "integer 1-20"),
                ("gamma", "0-5"), ("reg_alpha", "1e-8-10, log"),
                ("reg_lambda", "1e-8-10, log")],
    "LightGBM": [("n_estimators", "integer 200-1200"), ("num_leaves", "integer 15-255"),
                 ("max_depth", "integer -1-12"), ("learning_rate", "0.001-0.3, log"),
                 ("min_child_samples", "integer 5-100"), ("subsample", "0.5-1.0"),
                 ("colsample_bytree", "0.5-1.0"), ("reg_alpha", "1e-8-10, log"),
                 ("reg_lambda", "1e-8-10, log")],
    "CatBoost": [("iterations", "integer 200-1200"), ("depth", "integer 4-10"),
                 ("learning_rate", "0.001-0.3, log"), ("l2_leaf_reg", "1-10"),
                 ("border_count", "integer 32-255"),
                 ("bagging_temperature", "0-1"), ("random_strength", "0-10")],
    "RandomForest": [("n_estimators", "integer 200-800"), ("max_depth", "integer 5-25"),
                     ("min_samples_split", "integer 2-30"),
                     ("min_samples_leaf", "integer 1-20"), ("max_features", "0.1-1.0")],
    "ExtraTrees": [("n_estimators", "integer 200-800"), ("max_depth", "integer 5-25"),
                   ("min_samples_split", "integer 2-30"),
                   ("min_samples_leaf", "integer 1-20"), ("max_features", "0.1-1.0")],
    "LogRegEN": [("C", "0.001-10, log"), ("l1_ratio", "0-1")],
}
BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))
hrows = []
for clf, spec in SPACES.items():
    for par, rng_txt in spec:
        v = BEST[clf][par]
        hrows.append(dict(classifier=clf, parameter=par, search_range=rng_txt,
                          selected=(f"{v:.4g}" if isinstance(v, float) else str(v))))
H = pd.DataFrame(hrows)
H.to_csv(os.path.join(RES, "hyperparameters.csv"), index=False, encoding="utf-8")
print(f"  {len(H)} tuned parameters across 6 classifiers; 80 Optuna trials each "
      "(TPE sampler, median pruner, inner 3-fold StratifiedGroupKFold AUROC)")
edge = [(r.classifier, r.parameter, r.selected, r.search_range) for r in H.itertuples()
        if any(b in r.search_range for b in (r.selected + "-", "-" + r.selected + ","))]
if edge:
    print("  selected values sitting on a search boundary:")
    for e in edge:
        print("    ", e)

print(f"\nwritten: sample_size.csv, fold_composition.csv, hyperparameters.csv")
