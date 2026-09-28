import io
import os
import sys
import json
import types
import warnings
from collections import defaultdict

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
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

SEED = 42
N_PERM = 20
N_BOOT = 1000
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

BEST = json.load(open(os.path.join(BUNDLE, "03_results", "best_hyperparameters.json")))

FEATURE_LABELS = {
    "SP_MEAN": "SAPS mean score", "SP_SUM": "SAPS sum score",
    "SP_TOLERANCE": "SAPS tolerance", "SP_WITHDRAWAL": "SAPS withdrawal",
    "SP_VIRTUAL": "SAPS virtual-life", "SP_DAILY": "Daily smartphone use",
    "SP_SUM_BASELINE": "Baseline SAPS", "SP_SUM_DELTA": "SAPS change from baseline",
    "SP_SUM_SLOPE": "SAPS slope", "PREV_HIGH_RISK": "Prior high-risk",
    "WAVE_IDX": "Wave", "COHORT_SRC_e4": "Cohort", "YGENDER": "Sex",
    "YBRT1A": "Birth year", "YBRT1B": "Birth month", "YTWIN": "Twin",
    "P_PFBRTA1": "Father age", "P_PHOMPOP": "Household size",
    "P_PINCOME": "Household income",
}
_DOM = [("YPSY1", "Self-esteem"), ("YPSY2", "Depression"), ("YPSY3", "Social anxiety"),
        ("YPSY4", "Psychological well-being"), ("YPSY5", "Peer relationship"),
        ("YPSY6", "Self-control"), ("YPSY7", "Aggression"), ("YINT", "Smartphone use"),
        ("YTIM", "Time use"), ("YMDA", "Device access"), ("YEDU", "School engagement"),
        ("YDLQ", "Delinquency"), ("YFAM", "Family relationship"),
        ("YACT", "After-school activity"), ("YFUR", "Career orientation"),
        ("YPHY", "Physical health"), ("P_PMDA", "Parental mediation"),
        ("P_PPSY", "Parenting stress"), ("P_PFAM", "Parent-reported family"),
        ("P_PEDU", "Parent education involvement"),
        ("P_PSCHOOL", "Parent school involvement"), ("P_PWORK", "Parent work hours"),
        ("P_PPHY", "Parent health"), ("P_PFBRT", "Father info"),
        ("P_PYBRT", "Mother info"), ("P_PHOM", "Household structure"),
        ("P_PINC", "Household income")]

def group_of(code):
    if code in FEATURE_LABELS:
        return FEATURE_LABELS[code]
    for pref, lab in _DOM:
        if code.startswith(pref):
            return lab
    return None

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
group_codes = defaultdict(list)
for c in feat:
    g = group_of(c)
    if g is not None:
        group_codes[g].append(c)

AGG = pd.read_csv(os.path.join(BUNDLE, "03_results",
                               "shap_top25_xgboost_aggregated.csv"))
AGG.columns = ["scale", "summed_shap"]
frozen = dict(zip(AGG.scale, AGG.summed_shap))

print("=" * 78)
print("PART A  EXACT AUDIT, CONSISTENT WITH THE PUBLISHED FIGURES")
print("=" * 78)
print(f"{len(feat)} features map to {len(group_codes)} scales; "
      f"{sum(len(v) for v in group_codes.values())} features are grouped")

y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
dev, te = next(gss.split(df[feat].values, y, groups=ids))

imp_med = SimpleImputer(strategy="median").fit(df[feat].values[dev])
Xte = imp_med.transform(df[feat].values[te])
Xte_df = pd.DataFrame(Xte, columns=feat)

def cronbach(block):
    k = block.shape[1]
    if k < 2:
        return np.nan
    v = block.var(axis=0, ddof=1)
    tot = block.sum(axis=1).var(ddof=1)
    if tot <= 0:
        return np.nan
    return k / (k - 1) * (1 - v.sum() / tot)

rows = []
for g, codes in sorted(group_codes.items()):
    n = len(codes)
    tot = frozen.get(g, np.nan)
    rec = dict(scale=g, n_items=n, summed_shap=tot,
               per_item_shap=tot / n if n and not pd.isna(tot) else np.nan)
    if n >= 2:
        block = Xte_df[codes].to_numpy()
        keep = np.nanstd(block, axis=0) > 0
        block = block[:, keep]
        if block.shape[1] >= 2:
            z = StandardScaler().fit_transform(block)
            rho = spearmanr(block).statistic
            if np.ndim(rho) == 0:
                mabs = abs(float(rho))
            else:
                iu = np.triu_indices_from(rho, k=1)
                mabs = float(np.nanmean(np.abs(rho[iu])))
            cov = np.cov(z, rowvar=False)
            w, V = np.linalg.eigh(cov)
            order = np.argsort(w)[::-1]
            w, V = w[order], V[:, order]
            pc1 = w[0] / w.sum()
            load = V[:, 0]
            pos = float(np.mean(load > 0))
            rec.update(mean_abs_within_correlation=mabs, cronbach_alpha=cronbach(block),
                       pc1_variance_share=float(pc1),
                       share_items_positive_pc1_loading=max(pos, 1 - pos),
                       n_items_used=int(block.shape[1]))
    rows.append(rec)

A = pd.DataFrame(rows)
multi = A[A.n_items >= 2].dropna(subset=["summed_shap"])
rho_count = spearmanr(multi.n_items, multi.summed_shap)
rho_count_pi = spearmanr(multi.n_items, multi.per_item_shap)
print(f"\nitem count vs summed importance   Spearman rho {rho_count.statistic:+.3f} "
      f"(p = {rho_count.pvalue:.4f}, {len(multi)} multi-item scales)")
print(f"item count vs per-item importance Spearman rho {rho_count_pi.statistic:+.3f} "
      f"(p = {rho_count_pi.pvalue:.4f})")

A["rank_summed"] = A.summed_shap.rank(ascending=False)
A["rank_per_item"] = A.per_item_shap.rank(ascending=False)
A["rank_shift"] = A.rank_summed - A.rank_per_item
print("\ntop 8 by the published summed metric, with the per-item view alongside:")
for _, r in A.sort_values("summed_shap", ascending=False).head(8).iterrows():
    print(f"  {r.scale:<32} items {int(r.n_items):>3}  summed {r.summed_shap:.3f} "
          f"(rank {int(r.rank_summed):>2})   per item {r.per_item_shap:.4f} "
          f"(rank {int(r.rank_per_item):>2})")
print("\nlargest rank gains when normalising per item:")
for _, r in A.dropna(subset=["rank_shift"]).sort_values("rank_shift",
                                                        ascending=False).head(6).iterrows():
    print(f"  {r.scale:<32} items {int(r.n_items):>3}  summed rank "
          f"{int(r.rank_summed):>2} -> per-item rank {int(r.rank_per_item):>2}")

print("\ninternal structure of the largest batteries (beeswarm composite validity):")
for _, r in A[A.n_items >= 8].sort_values("summed_shap", ascending=False).head(8).iterrows():
    print(f"  {r.scale:<32} items {int(r.n_items):>3}  mean |rho| "
          f"{r.mean_abs_within_correlation:.3f}  alpha "
          f"{r.cronbach_alpha:>6.3f}  PC1 {100 * r.pc1_variance_share:>5.1f} %  "
          f"same-sign loadings {100 * r.share_items_positive_pc1_loading:>5.1f} %")

print("\n" + "=" * 78)
print("PART B  METHOD COMPARISON ON A LOCALLY REFITTED MODEL")
print("=" * 78)
pipe = ImbPipeline([("imp", SimpleImputer(strategy="median")),
                    ("smote_enn", SMOTEENN(random_state=SEED)),
                    ("clf", xgb.XGBClassifier(**BEST["XGBoost"], random_state=SEED,
                                              eval_metric="auc", tree_method="hist",
                                              n_jobs=-1, verbosity=0))])
pipe.fit(df[feat].values[dev], y[dev])
base_prob = pipe.predict_proba(df[feat].values[te])[:, 1]
base_auc = roc_auc_score(y[te], base_prob)
print(f"locally refitted held-out AUROC {base_auc:.4f} "
      f"(frozen value 0.8191; library-version difference)")

Xte_raw = df[feat].values[te].copy()
rng = np.random.default_rng(SEED)
perm = []
for g, codes in sorted(group_codes.items()):
    idx = [feat.index(c) for c in codes]
    drops = []
    for _ in range(N_PERM):
        Xp = Xte_raw.copy()
        order = rng.permutation(len(Xp))
        Xp[:, idx] = Xp[np.ix_(order, idx)]
        drops.append(base_auc - roc_auc_score(y[te], pipe.predict_proba(Xp)[:, 1]))
    perm.append(dict(scale=g, perm_importance=float(np.mean(drops)),
                     perm_sd=float(np.std(drops, ddof=1))))
P = pd.DataFrame(perm)
A = A.merge(P, on="scale", how="left")
A["rank_perm"] = A.perm_importance.rank(ascending=False)

m = A.dropna(subset=["summed_shap", "perm_importance"])
print(f"\ngrouped permutation importance, top 10 (AUROC drop when the whole scale is "
      f"permuted, {N_PERM} repeats):")
for _, r in A.sort_values("perm_importance", ascending=False).head(10).iterrows():
    print(f"  {r.scale:<32} items {int(r.n_items):>3}  drop {r.perm_importance:+.4f} "
          f"(SD {r.perm_sd:.4f})  summed-SHAP rank {int(r.rank_summed):>2} -> "
          f"permutation rank {int(r.rank_perm):>2}")
mm = m[m.n_items >= 2]
print(f"\nitem count vs grouped permutation importance  Spearman rho "
      f"{spearmanr(mm.n_items, mm.perm_importance).statistic:+.3f} "
      f"(p = {spearmanr(mm.n_items, mm.perm_importance).pvalue:.4f})")
print(f"rank agreement, summed SHAP vs permutation      Spearman rho "
      f"{spearmanr(m.summed_shap, m.perm_importance).statistic:+.3f}")
print(f"rank agreement, per-item SHAP vs permutation    Spearman rho "
      f"{spearmanr(m.per_item_shap, m.perm_importance).statistic:+.3f}")

A.to_csv(os.path.join(RES, "grouped_shap_audit.csv"), index=False, encoding="utf-8")

import shap
sub = rng.choice(len(te), size=min(1000, len(te)), replace=False)
Xsub = SimpleImputer(strategy="median").fit(df[feat].values[dev]).transform(
    df[feat].values[te][sub])
expl = shap.TreeExplainer(pipe.named_steps["clf"])
sv = expl.shap_values(Xsub)
if isinstance(sv, list):
    sv = sv[1]
gidx = {g: [feat.index(c) for c in codes] for g, codes in group_codes.items()}
names = sorted(gidx)
boot_ranks = {g: [] for g in names}
for _ in range(N_BOOT):
    r = rng.integers(0, sv.shape[0], sv.shape[0])
    ma = np.abs(sv[r]).mean(axis=0)
    tot = pd.Series({g: ma[i].sum() for g, i in gidx.items()})
    rk = tot.rank(ascending=False)
    for g in names:
        boot_ranks[g].append(rk[g])
S = pd.DataFrame([dict(scale=g, median_rank=float(np.median(v)),
                       rank_lo=float(np.percentile(v, 2.5)),
                       rank_hi=float(np.percentile(v, 97.5)),
                       share_top5=float(np.mean(np.asarray(v) <= 5)))
                  for g, v in boot_ranks.items()]).sort_values("median_rank")
S.to_csv(os.path.join(RES, "grouped_shap_stability.csv"), index=False, encoding="utf-8")
print(f"\nstability of the summed-SHAP ranking over {N_BOOT} bootstrap resamples "
      f"of the SHAP sample, top 8 by median rank:")
for _, r in S.head(8).iterrows():
    print(f"  {r.scale:<32} median rank {r.median_rank:>4.1f} "
          f"(95 % {r.rank_lo:>4.1f}-{r.rank_hi:>4.1f})  in top 5 on "
          f"{100 * r.share_top5:>5.1f} % of resamples")

print(f"\nwritten: {os.path.join(RES, 'grouped_shap_audit.csv')}")
print(f"written: {os.path.join(RES, 'grouped_shap_stability.csv')}")
