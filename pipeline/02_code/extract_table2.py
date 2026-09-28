import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from scipy import stats

ROOT = _paths.RERUN_DIR
FINAL = _paths.DATA_DIR

SEED = 42
TEST_SIZE = 0.20

print("[1] Loading transitions.pkl ...")
df = _paths.load_transitions()
print(f"    n_rows = {len(df)}, n_cols = {len(df.columns)}")
print(f"    positive prevalence = {df['y'].mean()*100:.4f}%")

print("[2] Reproducing the 80/20 GroupShuffleSplit by adolescent ID ...")
ids = df["ID"].astype(int).values
X_all = np.zeros((len(df), 1))
y_all = df["y"].astype(int).values
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
tr_idx, te_idx = next(gss.split(X_all, y_all, groups=ids))

dev = df.iloc[tr_idx].reset_index(drop=True)
tst = df.iloc[te_idx].reset_index(drop=True)
print(f"    development set : n_rows = {len(dev)}, "
      f"unique IDs = {dev['ID'].nunique()}, "
      f"positive prevalence = {dev['y'].mean()*100:.4f}%")
print(f"    held-out test   : n_rows = {len(tst)}, "
      f"unique IDs = {tst['ID'].nunique()}, "
      f"positive prevalence = {tst['y'].mean()*100:.4f}%")

def cat_counts(series, labels=None):
    s = series.dropna()
    out = {}
    for v in (labels if labels is not None else sorted(s.unique())):
        n = int((s == v).sum())
        out[str(v)] = n
    out["_n"] = int(s.shape[0])
    out["_missing"] = int(series.isna().sum())
    return out

def cont_summary(series):
    s = series.dropna()
    return {
        "n": int(s.shape[0]),
        "mean": float(s.mean()) if len(s) else None,
        "sd": float(s.std()) if len(s) else None,
        "median": float(s.median()) if len(s) else None,
        "iqr_lo": float(s.quantile(0.25)) if len(s) else None,
        "iqr_hi": float(s.quantile(0.75)) if len(s) else None,
        "missing": int(series.isna().sum()),
    }

def chi2_p(a, b):
    cats = sorted(set(pd.concat([a.dropna(), b.dropna()]).unique()))
    table = np.array([
        [int((a == c).sum()) for c in cats],
        [int((b == c).sum()) for c in cats],
    ])
    if table.shape[1] < 2 or table.sum() == 0:
        return None
    if (table.sum(axis=0) == 0).any():
        return None
    if table.shape == (2, 2) and (table < 5).any():
        try:
            _, p = stats.fisher_exact(table)
            return float(p)
        except Exception:
            pass
    try:
        _, p, _, _ = stats.chi2_contingency(table)
        return float(p)
    except Exception:
        return None

def cont_p(a, b):
    a = a.dropna().values
    b = b.dropna().values
    if len(a) == 0 or len(b) == 0:
        return None
    try:
        _, p = stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    except Exception:
        return None

results = {
    "n_rows": {"development": int(len(dev)), "test": int(len(tst))},
    "n_unique_ids": {
        "development": int(dev["ID"].nunique()),
        "test": int(tst["ID"].nunique()),
    },
    "positive_prevalence_pct": {
        "development": float(dev["y"].mean() * 100),
        "test": float(tst["y"].mean() * 100),
    },
    "outcome_y_high_risk_n": {
        "development": int(dev["y"].sum()),
        "test": int(tst["y"].sum()),
    },
    "variables": {},
}

results["variables"]["sex"] = {
    "label": "Sex",
    "type": "categorical",
    "labels": {"1": "Male", "2": "Female"},
    "development": cat_counts(dev["YGENDER"], labels=[1.0, 2.0]),
    "test": cat_counts(tst["YGENDER"], labels=[1.0, 2.0]),
    "p_value": chi2_p(dev["YGENDER"], tst["YGENDER"]),
}

results["variables"]["cohort"] = {
    "label": "KCYPS cohort source",
    "type": "categorical",
    "labels": {"0": "Elementary G1 panel (m1)", "1": "Elementary G4 panel (e4)"},
    "development": cat_counts(dev["COHORT_SRC_e4"], labels=[0, 1]),
    "test": cat_counts(tst["COHORT_SRC_e4"], labels=[0, 1]),
    "p_value": chi2_p(dev["COHORT_SRC_e4"], tst["COHORT_SRC_e4"]),
}

results["variables"]["transition"] = {
    "label": "Grade transition",
    "type": "categorical",
    "labels": {"G7->G8": "G7-to-G8", "G8->G9": "G8-to-G9", "G9->G10": "G9-to-G10"},
    "development": cat_counts(dev["TRANSITION"], labels=["G7->G8", "G8->G9", "G9->G10"]),
    "test": cat_counts(tst["TRANSITION"], labels=["G7->G8", "G8->G9", "G9->G10"]),
    "p_value": chi2_p(dev["TRANSITION"], tst["TRANSITION"]),
}

results["variables"]["wave_idx"] = {
    "label": "Wave index (transition)",
    "type": "categorical",
    "labels": {"0": "Transition 1", "1": "Transition 2", "2": "Transition 3"},
    "development": cat_counts(dev["WAVE_IDX"], labels=[0, 1, 2]),
    "test": cat_counts(tst["WAVE_IDX"], labels=[0, 1, 2]),
    "p_value": chi2_p(dev["WAVE_IDX"], tst["WAVE_IDX"]),
}

def income_clean(series):
    s = series.copy()
    s = s.replace({9999: np.nan})
    return s

dev_inc = income_clean(dev["P_PINCOME"])
tst_inc = income_clean(tst["P_PINCOME"])

def income_bin(s):
    bins = pd.cut(s, bins=[0, 3, 6, 9, 12], labels=["<=300", "300-600", "600-900", ">900"])
    return bins

dev_ibin = income_bin(dev_inc)
tst_ibin = income_bin(tst_inc)
results["variables"]["household_income"] = {
    "label": "Monthly household income (10,000 KRW)",
    "type": "categorical",
    "labels": {
        "<=300": "<= 300",
        "300-600": "301-600",
        "600-900": "601-900",
        ">900": "> 900",
    },
    "development": cat_counts(dev_ibin, labels=["<=300", "300-600", "600-900", ">900"]),
    "test": cat_counts(tst_ibin, labels=["<=300", "300-600", "600-900", ">900"]),
    "p_value": chi2_p(dev_ibin, tst_ibin),
}

results["variables"]["birth_year"] = {
    "label": "Birth year",
    "type": "continuous",
    "development": cont_summary(dev["YBRT1A"]),
    "test": cont_summary(tst["YBRT1A"]),
    "p_value": cont_p(dev["YBRT1A"], tst["YBRT1A"]),
}

if "P_PHOMPOP" in df.columns:
    results["variables"]["household_size"] = {
        "label": "Household size (persons)",
        "type": "continuous",
        "development": cont_summary(dev["P_PHOMPOP"]),
        "test": cont_summary(tst["P_PHOMPOP"]),
        "p_value": cont_p(dev["P_PHOMPOP"], tst["P_PHOMPOP"]),
    }

edu_col = None
for c in ["P_PEDU1A00", "P_PEDU1B01", "P_PEDU1C00"]:
    if c in df.columns:
        edu_col = c
        break
if edu_col:
    results["variables"]["parent_education"] = {
        "label": f"Parent education ({edu_col})",
        "type": "categorical",
        "development": cat_counts(dev[edu_col]),
        "test": cat_counts(tst[edu_col]),
        "p_value": chi2_p(dev[edu_col], tst[edu_col]),
    }

results["variables"]["prev_high_risk"] = {
    "label": "Prior-wave SAPS high-risk indicator",
    "type": "categorical",
    "labels": {"0": "Not high-risk at wave t",
               "1": "High-risk at wave t"},
    "development": cat_counts(dev["PREV_HIGH_RISK"], labels=[0, 1]),
    "test": cat_counts(tst["PREV_HIGH_RISK"], labels=[0, 1]),
    "p_value": chi2_p(dev["PREV_HIGH_RISK"], tst["PREV_HIGH_RISK"]),
}

if "SP_SUM_BASELINE" in df.columns:
    results["variables"]["saps_baseline"] = {
        "label": "SAPS 15-item sum at wave t",
        "type": "continuous",
        "development": cont_summary(dev["SP_SUM_BASELINE"]),
        "test": cont_summary(tst["SP_SUM_BASELINE"]),
        "p_value": cont_p(dev["SP_SUM_BASELINE"], tst["SP_SUM_BASELINE"]),
    }

results["variables"]["outcome_high_risk"] = {
    "label": "SAPS high-risk at wave t+1 (outcome)",
    "type": "categorical",
    "labels": {"0": "General", "1": "High-risk"},
    "development": cat_counts(dev["y"], labels=[0, 1]),
    "test": cat_counts(tst["y"], labels=[0, 1]),
    "p_value": chi2_p(dev["y"], tst["y"]),
}

print("\n[3] Table 2 summary preview")
print("=" * 70)
print(f"  Development set: n = {len(dev):,} transitions, "
      f"{dev['ID'].nunique():,} unique adolescents, "
      f"prev = {dev['y'].mean()*100:.2f}%")
print(f"  Held-out test : n = {len(tst):,} transitions, "
      f"{tst['ID'].nunique():,} unique adolescents, "
      f"prev = {tst['y'].mean()*100:.2f}%")

for key, v in results["variables"].items():
    print(f"\n  {key} ({v['type']})")
    print(f"    development: {v['development']}")
    print(f"    test       : {v['test']}")
    print(f"    p-value    : {v.get('p_value')}")

out_path = os.path.join(ROOT, "table2_demographics.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[OK] {out_path}")
