import os, sys
import numpy as np
import pandas as pd

BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "pipeline")
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths
import build_transitions as bt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)

def main():
    frozen, youth_cols, parent_cols, derived = bt.feature_lists()
    print(f"features: {len(youth_cols)} youth + {len(parent_cols)} parent "
          f"+ {len(derived)} derived + {len(bt.ENGINEERED)} engineered")

    T, missing, potential, A = bt.build(youth_cols, parent_cols)
    print(f"candidate transitions        : {len(T):,}")
    print(f"missing outcome SAPS         : {int(missing.sum()):,}")
    print(f"potential-risk (submitted excl): {int(potential.sum()):,}")
    print(f"submitted analytic file      : {len(A):,}  (high-risk {int(A['y'].sum()):,})")

    F = T[~missing].copy()
    high = F["_sum_t1"] >= 45
    pot = (~high) & ((F["_sum_t1"] >= 42) | (F["_daily_t1"] >= 14))
    F["y_ord"] = np.where(high, 2, np.where(pot, 1, 0))
    F["y_bin3"] = high.astype(int)
    F["sp_t1"] = F["_sum_t1"].astype(float)
    F = F.drop(columns=["_sum_t1", "_daily_t1"]).reset_index(drop=True)

    print(f"\nfull-spectrum file           : {len(F):,} transitions, "
          f"{F['ID'].nunique():,} adolescents")

    lab = {0: "General use", 1: "Potential risk", 2: "High risk"}
    rows = []
    for t in ["G7->G8", "G8->G9", "G9->G10"]:
        sub = F[F["TRANSITION"] == t]
        for k in (0, 1, 2):
            rows.append(dict(stratum=t, outcome=lab[k], n=int((sub["y_ord"] == k).sum()),
                             pct=100 * (sub["y_ord"] == k).mean(), denom=len(sub)))
    for k in (0, 1, 2):
        rows.append(dict(stratum="All transitions", outcome=lab[k],
                         n=int((F["y_ord"] == k).sum()),
                         pct=100 * (F["y_ord"] == k).mean(), denom=len(F)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "outcome_distribution_before_exclusion.csv"),
             index=False, encoding="utf-8")
    print("\nOutcome distribution BEFORE exclusion of the potential-risk band")
    for _, r in D.iterrows():
        print(f"  {r.stratum:<16}{r.outcome:<16}{r.n:>7,} / {r.denom:>7,}  {r.pct:>6.2f}%")

    R = F[F["y_ord"] != 1].copy()
    key = ["ID", "TRANSITION", "COHORT_SRC_e4"]
    def norm(d):
        d = d.copy()
        d["ID"] = d["ID"].astype("int64"); d["COHORT_SRC_e4"] = d["COHORT_SRC_e4"].astype(int)
        d["TRANSITION"] = d["TRANSITION"].astype(str)
        return d.sort_values(key).reset_index(drop=True)
    Fz, Rn = norm(frozen), norm(R)
    ok = len(Fz) == len(Rn) and (Fz[key] == Rn[key]).all().all()
    print(f"\nrestricted subset vs frozen transitions.pkl: "
          f"{len(Rn):,} rows, keys {'aligned' if ok else 'MISALIGNED'}")
    if ok:
        bad = 0
        for c in [c for c in Fz.columns if c not in key + ["y"]]:
            a = pd.to_numeric(Fz[c], errors="coerce"); b = pd.to_numeric(Rn[c], errors="coerce")
            bad += int((~(np.isclose(a, b, equal_nan=True) | (a.isna() & b.isna()))).sum())
        print(f"  differing cells across 455 features: {bad}")
        print(f"  outcome label mismatches           : "
              f"{int((Fz['y'].astype(int) != Rn['y_bin3'].astype(int)).sum())}")

    out = os.path.join(OUT, "transitions_fullspectrum.pkl")
    F.to_pickle(out)
    print(f"\nwritten: {out}")

if __name__ == "__main__":
    main()
