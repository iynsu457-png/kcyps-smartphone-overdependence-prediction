import io
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", write_through=True)

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
y_all = df["y"].astype(int).values
ids_all = df["ID"].astype("int64").values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                random_state=SEED).split(df[feat].values, y_all,
                                                         groups=ids_all))
TP = pd.read_csv(_paths.TEST_PREDICTIONS)
assert (TP["ID"].astype("int64").values == ids_all[te]).all()
y = TP["y_true"].astype(int).values
gid = TP["ID"].astype("int64").values
full = TP["XGBoost"].values

med = np.nanmedian(df["SP_SUM"].values[dev])
score = np.where(np.isnan(df["SP_SUM"].values[te]), med, df["SP_SUM"].values[te])

STRATA = [("low", score < 35, "wave-t SAPS sum below 35"),
          ("middle", (score >= 35) & (score < 42), "wave-t SAPS sum 35 to 41"),
          ("upper", score >= 42, "wave-t SAPS sum 42 or above")]

print("=" * 78)
print("INCREMENTAL VALUE BY STRATUM OF CURRENT SAPS SCORE")
print("=" * 78)
print(f"held-out n = {len(y):,}; high risk {int(y.sum())} ({100 * y.mean():.2f} %)")

rows = []
for name, mask, lab in STRATA:
    n, npos = int(mask.sum()), int(y[mask].sum())
    print(f"\n{name:<7} {lab:<34} n = {n:>5}  high risk {npos:>3} "
          f"({100 * npos / max(n, 1):.2f} %)")
    if npos < 5 or len(np.unique(y[mask])) < 2:
        print("        too few events for a stratum-specific comparison")
        rows.append(dict(analysis="stratum", stratum=name, label=lab, n=n, n_pos=npos,
                         note="too few events"))
        continue
    g = {}
    for i, k in enumerate(gid[mask]):
        g.setdefault(k, []).append(i)
    g = {k: np.asarray(v) for k, v in g.items()}
    keys = np.array(list(g.keys()))
    ys, fs, ss = y[mask], full[mask], score[mask]
    rng = np.random.default_rng(SEED)
    da, dp, af, asc = [], [], [], []
    for _ in range(2000):
        r = np.concatenate([g[k] for k in rng.choice(keys, len(keys), replace=True)])
        if len(np.unique(ys[r])) < 2:
            continue
        a_f = roc_auc_score(ys[r], fs[r])
        a_s = roc_auc_score(ys[r], ss[r])
        af.append(a_f)
        asc.append(a_s)
        da.append(a_f - a_s)
        dp.append(average_precision_score(ys[r], fs[r])
                  - average_precision_score(ys[r], ss[r]))
    rec = dict(analysis="stratum", stratum=name, label=lab, n=n, n_pos=npos,
               AUROC_model=roc_auc_score(ys, fs),
               AUROC_model_lo=float(np.percentile(af, 2.5)),
               AUROC_model_hi=float(np.percentile(af, 97.5)),
               AUROC_score=roc_auc_score(ys, ss),
               AUROC_score_lo=float(np.percentile(asc, 2.5)),
               AUROC_score_hi=float(np.percentile(asc, 97.5)),
               delta_AUROC=roc_auc_score(ys, fs) - roc_auc_score(ys, ss),
               delta_AUROC_lo=float(np.percentile(da, 2.5)),
               delta_AUROC_hi=float(np.percentile(da, 97.5)),
               PR_AUC_model=average_precision_score(ys, fs),
               PR_AUC_score=average_precision_score(ys, ss),
               delta_PR_AUC=average_precision_score(ys, fs)
               - average_precision_score(ys, ss),
               delta_PR_AUC_lo=float(np.percentile(dp, 2.5)),
               delta_PR_AUC_hi=float(np.percentile(dp, 97.5)))
    rows.append(rec)
    print(f"        model AUROC {rec['AUROC_model']:.3f} "
          f"({rec['AUROC_model_lo']:.3f}-{rec['AUROC_model_hi']:.3f})   "
          f"SAPS score {rec['AUROC_score']:.3f} "
          f"({rec['AUROC_score_lo']:.3f}-{rec['AUROC_score_hi']:.3f})")
    print(f"        difference  {rec['delta_AUROC']:+.3f} "
          f"({rec['delta_AUROC_lo']:+.3f} to {rec['delta_AUROC_hi']:+.3f})   "
          f"PR-AUC {rec['delta_PR_AUC']:+.3f} "
          f"({rec['delta_PR_AUC_lo']:+.3f} to {rec['delta_PR_AUC_hi']:+.3f})")

print("\n" + "=" * 78)
print("PRECISION OF THE POOLED COMPARISON")
print("=" * 78)
g = {}
for i, k in enumerate(gid):
    g.setdefault(k, []).append(i)
g = {k: np.asarray(v) for k, v in g.items()}
keys = np.array(list(g.keys()))
rng = np.random.default_rng(SEED)
d = []
for _ in range(2000):
    r = np.concatenate([g[k] for k in rng.choice(keys, len(keys), replace=True)])
    if len(np.unique(y[r])) < 2:
        continue
    d.append(roc_auc_score(y[r], full[r]) - roc_auc_score(y[r], score[r]))
d = np.asarray(d)
se = float(d.std(ddof=1))
point = roc_auc_score(y, full) - roc_auc_score(y, score)
mde = 1.96 * se
print(f"pooled paired difference {point:+.4f}; bootstrap standard error {se:.4f}")
print(f"the smallest difference this held-out set could have declared significant is "
      f"about {mde:.3f} AUROC")
print(f"so a true advantage of up to {mde:.3f} would not have been detected with "
      f"{int(y.sum())} events; the interval is consistent with anything from "
      f"{point - mde:+.3f} to {point + mde:+.3f}")
rows.append(dict(analysis="precision", stratum="pooled",
                 label="precision of the pooled paired difference",
                 delta_AUROC=point, delta_AUROC_lo=point - mde,
                 delta_AUROC_hi=point + mde, se=se,
                 minimum_detectable_difference=mde, n=len(y), n_pos=int(y.sum())))

pd.DataFrame(rows).to_csv(os.path.join(RES, "incremental_by_stratum.csv"),
                          index=False, encoding="utf-8")
print(f"\nwritten: {os.path.join(RES, 'incremental_by_stratum.csv')}")
