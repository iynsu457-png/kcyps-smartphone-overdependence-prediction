import os
import sys

import numpy as np
import pandas as pd
import pyreadstat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

RAW = os.path.join(_paths.DATA_DIR, "raw")
PANELS = {
    "m1": (os.path.join(RAW, "KCYPS2018m1[SPSS]"),
           "KCYPS2018m1{k}w{w}.sav", {7: 1, 8: 2, 9: 3, 10: 4}),
    "e4": (os.path.join(RAW, "KCYPS2018e4[SPSS]"),
           "KCYPS2018e4{k}w{w}.sav", {7: 4, 8: 5, 9: 6, 10: 7}),
}
SPECIAL = {("e4", "Y", 3): "KCYPS2018e4Yw3_0907.sav",
           ("e4", "Y", 7): "KCYPS2018e4Yw7_0907.sav"}

REVERSE_ITEMS = [5, 10, 15]
SUBSCALES = {"SP_DAILY": range(1, 6), "SP_VIRTUAL": range(6, 8),
             "SP_WITHDRAWAL": range(8, 12), "SP_TOLERANCE": range(12, 16)}
ENGINEERED = {"WAVE_IDX", "COHORT_SRC_e4", "PREV_HIGH_RISK"}

def feature_lists():
    frozen = _paths.load_transitions()
    feats = [c for c in frozen.columns if c not in ("y", "ID", "TRANSITION")]
    derived = {c for c in feats if c.startswith("SP_")}
    youth = [c for c in feats
             if not c.startswith("P_") and c not in ENGINEERED and c not in derived]
    parent = [c for c in feats if c.startswith("P_")]
    return frozen, youth, parent, sorted(derived)

def wave_frame(panel, grade, youth_cols, parent_cols):
    base, tpl, gmap = PANELS[panel]
    w = gmap[grade]
    fy = SPECIAL.get((panel, "Y", w), tpl.format(k="Y", w=w))
    fp = SPECIAL.get((panel, "P", w), tpl.format(k="P", w=w))
    saps_items = [f"YMDA1C{i:02d}w{w}" for i in range(1, 16)]

    y, _ = pyreadstat.read_sav(
        os.path.join(base, fy),
        usecols=["ID"] + [f"{c}w{w}" for c in youth_cols] + saps_items)
    y = y.rename(columns={f"{c}w{w}": c for c in youth_cols})
    y.index = y["ID"].astype("int64")

    p, _ = pyreadstat.read_sav(
        os.path.join(base, fp),
        usecols=["ID"] + [f"{c[2:]}w{w}" for c in parent_cols])
    p = p.rename(columns={f"{c[2:]}w{w}": c for c in parent_cols})
    p.index = p["ID"].astype("int64")

    scored = y[saps_items].copy()
    for i in REVERSE_ITEMS:
        scored[saps_items[i - 1]] = 5 - scored[saps_items[i - 1]]

    d = pd.DataFrame(index=y.index)
    d["SP_SUM"] = scored.sum(axis=1, skipna=False)
    d["SP_MEAN"] = d["SP_SUM"] / 15
    for name, items in SUBSCALES.items():
        d[name] = scored[[saps_items[i - 1] for i in items]].sum(axis=1, skipna=False)

    return pd.concat([y[youth_cols], p[parent_cols].reindex(y.index), d], axis=1)

def build(youth_cols, parent_cols):
    rows = []
    for panel, (_, _, gmap) in PANELS.items():
        frames = {g: wave_frame(panel, g, youth_cols, parent_cols) for g in gmap}
        baseline = frames[7]["SP_SUM"]
        for g in (7, 8, 9):
            cur, nxt = frames[g], frames[g + 1]
            idx = cur.index.union(nxt.index)
            a = cur.reindex(idx).copy()
            a["SP_SUM_BASELINE"] = baseline.reindex(idx)
            a["SP_SUM_DELTA"] = a["SP_SUM"] - a["SP_SUM_BASELINE"]
            a["WAVE_IDX"] = g - 7
            a["SP_SUM_SLOPE"] = 0.0 if g == 7 else a["SP_SUM_DELTA"] / (g - 7)
            a["COHORT_SRC_e4"] = int(panel == "e4")
            a["PREV_HIGH_RISK"] = (a["SP_SUM"] >= 45).astype(int)
            a["TRANSITION"] = f"G{g}->G{g+1}"
            a["ID"] = idx
            a["_sum_t1"] = nxt["SP_SUM"].reindex(idx)
            a["_daily_t1"] = nxt["SP_DAILY"].reindex(idx)
            rows.append(a)

    T = pd.concat(rows, ignore_index=True)
    missing = T["_sum_t1"].isna()
    high = (~missing) & (T["_sum_t1"] >= 45)
    potential = (~missing) & (~high) & ((T["_sum_t1"] >= 42) | (T["_daily_t1"] >= 14))

    A = T[~missing & ~potential].copy()
    A["y"] = high[~missing & ~potential].astype(int).values
    return T, missing, potential, A.drop(columns=["_sum_t1", "_daily_t1"])

def main():
    frozen, youth_cols, parent_cols, derived = feature_lists()
    print(f"columns to rebuild: {len(youth_cols)} youth + {len(parent_cols)} parent "
          f"+ {len(derived)} derived + {len(ENGINEERED)} engineered "
          f"= {len(youth_cols)+len(parent_cols)+len(derived)+len(ENGINEERED)}\n")

    T, missing, potential, A = build(youth_cols, parent_cols)

    expected = {"candidate transitions": 15591, "potential-risk excluded": 2658,
                "missing-outcome excluded": 1961, "analytic transitions": 10972,
                "unique adolescents": 4634, "high-risk outcomes": 517}
    got = {"candidate transitions": len(T), "potential-risk excluded": int(potential.sum()),
           "missing-outcome excluded": int(missing.sum()), "analytic transitions": len(A),
           "unique adolescents": A["ID"].nunique(), "high-risk outcomes": int(A["y"].sum())}

    print("Fig. 1 flow")
    print(f"  {'':<26}{'rebuilt':>10}{'manuscript':>13}")
    for k in expected:
        print(f"  {k:<26}{got[k]:>10,}{expected[k]:>13,}   "
              f"{'OK' if got[k] == expected[k] else 'MISMATCH'}")
    print(f"  {'prevalence':<26}{A['y'].mean()*100:>9.2f}%{4.71:>12.2f}%")
    print("  per transition: " + "  ".join(
        f"{t} {n:,}" for t, n in A["TRANSITION"].value_counts().sort_index().items()))

    key = ["ID", "TRANSITION", "COHORT_SRC_e4"]
    def norm(d):
        d = d.copy()
        d["ID"] = d["ID"].astype("int64")
        d["COHORT_SRC_e4"] = d["COHORT_SRC_e4"].astype(int)
        d["TRANSITION"] = d["TRANSITION"].astype(str)
        return d.sort_values(key).reset_index(drop=True)
    F, R = norm(frozen), norm(A)

    print("\nComparison against 01_data/transitions.pkl")
    if len(F) != len(R) or not (F[key] == R[key]).all().all():
        print(f"  ROW KEYS DIFFER — frozen {len(F):,} vs rebuilt {len(R):,}")
        return
    print(f"  rows aligned            : {len(F):,}")
    bad = []
    for c in [c for c in F.columns if c not in key]:
        a = pd.to_numeric(F[c], errors="coerce")
        b = pd.to_numeric(R[c], errors="coerce")
        n = int((~(np.isclose(a, b, equal_nan=True) | (a.isna() & b.isna()))).sum())
        if n:
            bad.append((c, n))
    total = len(F.columns) - len(key)
    print(f"  columns compared        : {total}")
    print(f"  columns matching exactly: {total - len(bad)}")
    print(f"  columns differing       : {len(bad)}")
    for c, n in bad[:20]:
        print(f"      {c:24s} {n:,} cells")

    out = os.path.join(_paths.RERUN_DIR, "transitions_rebuilt.pkl")
    R.to_pickle(out)
    print(f"\n  written: {out}")

if __name__ == "__main__":
    main()
