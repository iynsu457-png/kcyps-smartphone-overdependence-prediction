import io
import os
import sys
import warnings
from importlib import metadata

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", write_through=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import GroupShuffleSplit

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
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
               .split(df[feat].values, y, groups=ids))
rows = []

def add(**k):
    rows.append(k)

inc = df["P_PINCOME"].replace({9999: np.nan})
bands = pd.cut(inc, bins=[0, 3, 6, 9, 12], labels=["< 200", "200 to < 500",
                                                   "500 to < 800", ">= 800"])
for part, idx in (("development", dev), ("test", te)):
    n = len(idx)
    sex = df["YGENDER"].iloc[idx]
    for lab, v in (("Male", (sex == 1).sum()), ("Female", (sex == 2).sum()),
                   ("Missing", sex.isna().sum())):
        add(analysis="M4_table1", partition=part, variable="Sex", level=lab, n=int(v),
            pct=100 * v / n)
    b = bands.iloc[idx]
    for lab in ["< 200", "200 to < 500", "500 to < 800", ">= 800"]:
        v = int((b == lab).sum())
        add(analysis="M4_table1", partition=part, variable="Income", level=lab, n=v,
            pct=100 * v / n)
    v = int(b.isna().sum())
    add(analysis="M4_table1", partition=part, variable="Income", level="Missing", n=v,
        pct=100 * v / n)
    for var in ("YBRT1A", "P_PHOMPOP", "SP_SUM"):
        s = df[var].iloc[idx]
        add(analysis="M4_table1", partition=part, variable=var, level="Missing",
            n=int(s.isna().sum()), pct=100 * s.isna().sum() / n,
            mean=float(s.mean()), sd=float(s.std()))
    print(f"  {part}: n {n}; sex missing {int(sex.isna().sum())}; income missing "
          f"{int(b.isna().sum())}; SAPS sum at t missing {int(df['SP_SUM'].iloc[idx].isna().sum())}")

TP = pd.read_csv(_paths.TEST_PREDICTIONS)
TP.columns = [c.lstrip("﻿") for c in TP.columns]
yt = y[te]

def fauc(a, s):
    pos = a == 1
    r = rankdata(s)
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())

grp = {}
for i, g in enumerate(ids[te]):
    grp.setdefault(g, []).append(i)
grp = {k: np.asarray(v) for k, v in grp.items()}
keys = np.array(list(grp.keys()))
rng = np.random.default_rng(42)
d = []
for _ in range(2000):
    r = np.concatenate([grp[k] for k in rng.choice(keys, len(keys), replace=True)])
    d.append(fauc(yt[r], TP["XGBoost"].values[r]) - fauc(yt[r], TP["LogRegEN"].values[r]))
d = np.asarray(d)
pt = fauc(yt, TP["XGBoost"].values) - fauc(yt, TP["LogRegEN"].values)
add(analysis="M8_xgb_minus_lr_heldout", value=pt, lo=np.percentile(d, 2.5),
    hi=np.percentile(d, 97.5))
print(f"  held-out AUROC, XGBoost minus elastic net: {pt:+.3f} "
      f"({np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})")

items = [f"P_PMDA1C{i:02d}" for i in range(1, 16)]
sc = df[items].copy()
for i in (5, 10, 15):
    sc[items[i - 1]] = 5 - sc[items[i - 1]]
dep = sc.sum(axis=1, skipna=False)
ok = dep.notna()
hi_, gen = dep[ok & (y == 1)], dep[ok & (y == 0)]
a = fauc(y[ok.values], dep[ok].values)
rho = spearmanr(dep[ok & df["SP_SUM"].notna()], df["SP_SUM"][ok & df["SP_SUM"].notna()])
add(analysis="M9_caregiver_dependence", n=int(ok.sum()), mean_high=float(hi_.mean()),
    sd_high=float(hi_.std()), mean_general=float(gen.mean()), sd_general=float(gen.std()),
    auroc=a, rho_with_adolescent_saps=float(rho.statistic))
print(f"  caregiver dependence sum: high risk {hi_.mean():.1f} (SD {hi_.std():.1f}) vs "
      f"general {gen.mean():.1f} (SD {gen.std():.1f}); AUROC {a:.3f}; rho with "
      f"adolescent SAPS {rho.statistic:+.3f}")

for pkg in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm",
            "catboost", "optuna", "shap", "scipy", "matplotlib"):
    try:
        v = metadata.version(pkg)
    except metadata.PackageNotFoundError:
        v = "not installed"
    add(analysis="M6_versions", package=pkg, version=v)
add(analysis="M6_versions", package="python", version=sys.version.split()[0])
print("  versions:", {r["package"]: r["version"] for r in rows
                      if r["analysis"] == "M6_versions"})

pd.DataFrame(rows).to_csv(os.path.join(RES, "minor_numbers.csv"), index=False,
                          encoding="utf-8")
print("written: minor_numbers.csv")
