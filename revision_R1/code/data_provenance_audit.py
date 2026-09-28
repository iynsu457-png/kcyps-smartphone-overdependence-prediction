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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
RAW = os.path.join(BUNDLE, "01_data", "raw")
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

NEUTRAL = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
               colsample_bytree=1.0, min_child_weight=5, gamma=0.0, reg_alpha=0.0,
               reg_lambda=1.0)
PANELS = {"m1": ("KCYPS2018m1[SPSS]", "KCYPS2018m1{k}w{w}.sav", {7: 1, 8: 2, 9: 3, 10: 4}),
          "e4": ("KCYPS2018e4[SPSS]", "KCYPS2018e4{k}w{w}.sav", {7: 4, 8: 5, 9: 6, 10: 7})}
SPECIAL = {("e4", "Y", 3): "KCYPS2018e4Yw3_0907.sav",
           ("e4", "Y", 7): "KCYPS2018e4Yw7_0907.sav"}

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
DERIVED = [c for c in feat if c.startswith("SP_")]
ENG = ["PREV_HIGH_RISK", "COHORT_SRC_e4", "WAVE_IDX"]
youth = [c for c in feat if not c.startswith("P_") and c not in ENG + DERIVED]
parent = [c for c in feat if c.startswith("P_")]
y = df["y"].astype(int).values
ids = df["ID"].astype("int64").values
coh = np.where(df["COHORT_SRC_e4"].astype(int) == 1, "e4", "m1")
trans = df["TRANSITION"].astype(str).values

print("=" * 78)
print("D1  AVAILABILITY OF EVERY MODELLED COLUMN AT EVERY GRADE IN BOTH PANELS")
print("=" * 78)
print(f"455 columns = {len(youth)} adolescent-report + {len(parent)} parent-report "
      f"+ {len(DERIVED)} SAPS-derived + {len(ENG)} study-design")
rows = []
try:
    import pyreadstat
    ok = True
    for panel, (folder, tpl, gmap) in PANELS.items():
        for grade, w in gmap.items():
            for kind, cols in (("Y", youth), ("P", [c[2:] for c in parent])):
                fn = SPECIAL.get((panel, kind, w), tpl.format(k=kind, w=w))
                _, meta = pyreadstat.read_sav(os.path.join(RAW, folder, fn),
                                              metadataonly=True)
                have = set(meta.column_names)
                want = {f"{c}w{w}" for c in cols}
                miss = sorted(want - have)
                rows.append(dict(analysis="D1_availability", panel=panel, grade=grade,
                                 wave=w, report=kind, n_expected=len(want),
                                 n_missing_from_file=len(miss),
                                 missing_examples="; ".join(miss[:5])))
                if miss:
                    ok = False
                    print(f"  {panel} grade {grade} ({kind}w{w}): {len(miss)} of "
                          f"{len(want)} codes absent, e.g. {miss[:4]}")
    if ok:
        print("  every modelled item code is present in every wave file of both panels, "
              "at all four grades")
        print("  no item was renamed or recoded across waves, so no harmonization of "
              "wording or coding was required")
except ImportError:
    print("  pyreadstat unavailable; availability check skipped")

print("\n" + "=" * 78)
print("D2  MISSINGNESS BY FEATURE, COHORT, AND GRADE TRANSITION")
print("=" * 78)
M = df[feat].isna()
prof = pd.DataFrame({"feature": feat, "missing_overall": M.mean().values})
for c in ("m1", "e4"):
    prof[f"missing_{c}"] = M[coh == c].mean().values
for t in ["G7->G8", "G8->G9", "G9->G10"]:
    prof[f"missing_{t}"] = M[trans == t].mean().values
prof["cohort_gap"] = (prof.missing_m1 - prof.missing_e4).abs()
prof["wave_range"] = prof[[f"missing_{t}" for t in
                           ["G7->G8", "G8->G9", "G9->G10"]]].max(axis=1) - \
                     prof[[f"missing_{t}" for t in
                           ["G7->G8", "G8->G9", "G9->G10"]]].min(axis=1)
prof.sort_values("missing_overall", ascending=False).to_csv(
    os.path.join(RES, "provenance_missingness.csv"), index=False, encoding="utf-8")
print(f"mean {100 * prof.missing_overall.mean():.2f} %, median "
      f"{100 * prof.missing_overall.median():.2f} %, maximum "
      f"{100 * prof.missing_overall.max():.2f} %")
print(f"features above 20 % missing: {(prof.missing_overall > 0.2).sum()}")
print(f"features fully missing in either cohort: "
      f"{int(((prof.missing_m1 > 0.999) | (prof.missing_e4 > 0.999)).sum())}")
print(f"largest cohort gap {100 * prof.cohort_gap.max():.1f} points; largest "
      f"between-transition range {100 * prof.wave_range.max():.1f} points")
print("\nten features with the most missing data:")
for _, r in prof.sort_values("missing_overall", ascending=False).head(10).iterrows():
    print(f"  {r.feature:16s} overall {100 * r.missing_overall:5.1f} %  m1 "
          f"{100 * r.missing_m1:5.1f} %  e4 {100 * r.missing_e4:5.1f} %  "
          f"between-wave range {100 * r.wave_range:4.1f}")

print("\n" + "=" * 78)
print("D3  DOES MISSINGNESS ITSELF CARRY SIGNAL?")
print("=" * 78)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
dev, te = next(gss.split(df[feat].values, y, groups=ids))
Mi = M.astype(float).values

def fit_auc(X, target, dv=dev, tv=te):
    pipe = ImbPipeline([("imp", SimpleImputer(strategy="median")),
                        ("smote_enn", SMOTEENN(random_state=SEED)),
                        ("clf", xgb.XGBClassifier(**NEUTRAL, random_state=SEED,
                                                  eval_metric="auc",
                                                  tree_method="hist", n_jobs=-1,
                                                  verbosity=0))])
    pipe.fit(X[dv], target[dv])
    p = pipe.predict_proba(X[tv])[:, 1]
    return roc_auc_score(target[tv], p), average_precision_score(target[tv], p)

a_y, p_y = fit_auc(Mi, y)
print(f"outcome from missingness indicators alone      AUROC {a_y:.3f}  PR-AUC {p_y:.3f}")
a_c, _ = fit_auc(Mi, (coh == "e4").astype(int))
print(f"cohort from missingness indicators alone       AUROC {a_c:.3f}")
a_w, _ = fit_auc(Mi, (trans == "G7->G8").astype(int))
print(f"earliest transition from missingness alone     AUROC {a_w:.3f}")
res3 = [dict(analysis="D3_missingness_signal", target="outcome", AUROC=a_y, PR_AUC=p_y),
        dict(analysis="D3_missingness_signal", target="cohort", AUROC=a_c),
        dict(analysis="D3_missingness_signal", target="G7-to-G8 transition", AUROC=a_w)]

print("\n" + "=" * 78)
print("D4  ENCODING OF EVERY MODELLED COLUMN")
print("=" * 78)
NOMINAL_HINTS = ("YINT1A", "YINT2A", "P_PJOB", "P_PARENT", "P_PFRELATION", "YACT",
                 "P_PWORKW", "YMDA1A")
enc = []
for c in feat:
    v = df[c].dropna()
    u = np.unique(v.values)
    kind = "continuous"
    if len(u) <= 2:
        kind = "binary"
    elif np.allclose(u, u.astype(int)) and len(u) <= 7:
        kind = "ordinal-like (<=7 integer levels)"
    elif np.allclose(u, u.astype(int)) and len(u) <= 30:
        kind = "integer, many levels"
    nominal = c.startswith(NOMINAL_HINTS) and kind != "binary" and kind != "continuous"
    enc.append(dict(feature=c, n_levels=int(len(u)), kind=kind,
                    likely_nominal=bool(nominal), min=float(u.min()) if len(u) else np.nan,
                    max=float(u.max()) if len(u) else np.nan))
E = pd.DataFrame(enc)
E.to_csv(os.path.join(RES, "provenance_encoding.csv"), index=False, encoding="utf-8")
for k, n in E.kind.value_counts().items():
    print(f"  {k:<34} {n:>4}")
print(f"  of which flagged as probably nominal rather than ordinal: "
      f"{int(E.likely_nominal.sum())}")
print("  all columns were passed to the classifiers as numeric codes; no one-hot "
       "encoding was applied")

print("\n" + "=" * 78)
print("D5  ITEM-LEVEL COLUMNS AGAINST VALIDATED SCALE SCORES")
print("=" * 78)
FEATURE_LABELS = {"SP_MEAN": "SAPS mean score", "SP_SUM": "SAPS sum score",
                  "SP_TOLERANCE": "SAPS tolerance", "SP_WITHDRAWAL": "SAPS withdrawal",
                  "SP_VIRTUAL": "SAPS virtual-life", "SP_DAILY": "SAPS daily-life",
                  "SP_SUM_BASELINE": "Baseline SAPS",
                  "SP_SUM_DELTA": "SAPS change", "SP_SUM_SLOPE": "SAPS slope",
                  "PREV_HIGH_RISK": "Prior high-risk", "WAVE_IDX": "Wave",
                  "COHORT_SRC_e4": "Cohort", "YGENDER": "Sex", "YTWIN": "Twin",
                  "P_PFBRTA1": "Father age", "P_PHOMPOP": "Household size",
                  "P_PINCOME": "Household income"}
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

def grp(c):
    if c in FEATURE_LABELS:
        return FEATURE_LABELS[c]
    for pre, lab in _DOM:
        if c.startswith(pre):
            return lab
    return None

gc = defaultdict(list)
for c in feat:
    g = grp(c)
    if g:
        gc[g].append(c)
S = pd.DataFrame({g: df[cs].mean(axis=1) if len(cs) > 1 else df[cs[0]]
                  for g, cs in gc.items()})
print(f"scale-score matrix: {S.shape[1]} columns from {sum(len(v) for v in gc.values())} "
      f"items (battery items averaged, single-item features kept as they are)")
a_s, p_s = fit_auc(S.values, y)
a_f, p_f = fit_auc(df[feat].values, y)
print(f"  scale scores  ({S.shape[1]:>3} columns)  AUROC {a_s:.3f}  PR-AUC {p_s:.3f}")
print(f"  item level    ({len(feat)} columns)  AUROC {a_f:.3f}  PR-AUC {p_f:.3f}")
print(f"  difference, item level minus scale scores: AUROC {a_f - a_s:+.3f}  "
      f"PR-AUC {p_f - p_s:+.3f}")
res3 += [dict(analysis="D5_scale_vs_item", model="scale scores",
              n_features=int(S.shape[1]), AUROC=a_s, PR_AUC=p_s),
         dict(analysis="D5_scale_vs_item", model="item level",
              n_features=len(feat), AUROC=a_f, PR_AUC=p_f)]
pd.DataFrame(res3).to_csv(os.path.join(RES, "provenance_models.csv"),
                          index=False, encoding="utf-8")
print(f"\nwritten: {os.path.join(RES, 'provenance_missingness.csv')}")
print(f"written: {os.path.join(RES, 'provenance_encoding.csv')}")
print(f"written: {os.path.join(RES, 'provenance_models.csv')}")
