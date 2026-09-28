import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = os.environ.get(
    "REV_RESULTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))

CLFS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "LogRegEN"]
PRETTY = {"XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost",
          "RandomForest": "Random Forest", "ExtraTrees": "Extra Trees",
          "LogRegEN": "Elastic-net Logistic Regression"}

SUBMITTED = {
    "XGBoost": (0.819, 0.767, 0.866, 0.409, 0.958, 0.935, 0.347),
    "ExtraTrees": (0.819, 0.766, 0.866, 0.409, 0.950, 0.927, 0.321),
    "RandomForest": (0.814, 0.763, 0.861, 0.387, 0.954, 0.930, 0.320),
    "LightGBM": (0.811, 0.758, 0.860, 0.387, 0.967, 0.942, 0.362),
    "CatBoost": (0.804, 0.748, 0.851, 0.409, 0.939, 0.917, 0.293),
    "LogRegEN": (0.742, 0.685, 0.793, 0.333, 0.922, 0.897, 0.215),
}

FLOW = dict(enrolled=5197, m1=2590, e4=2607, candidate=15591, missing=1961,
            fullspec=13630, fullspec_ids=4882, general=10455, potential=2658,
            high=517, primary=10972, primary_ids=4634,
            pct_general=76.71, pct_potential=19.50, pct_high=3.79,
            pot_below42=1934, pot_42_44=724,
            mean_general=30.35, sd_general=5.61, mean_potential=38.81,
            sd_potential=3.55, mean_high=47.41, sd_high=2.79,
            fs_dev=10901, fs_dev_ids=3905, fs_dev_pos=417, fs_dev_prev=3.83,
            fs_te=2729, fs_te_ids=977, fs_te_pos=100, fs_te_prev=3.66,
            fs_te_extreme=2195, fs_te_extreme_prev=4.56,
            fs_dev_extreme=8777, fs_dev_extreme_prev=4.75)

BY_TRANSITION = {
    "G7-to-G8": (3624, 914, 154, 77.24, 19.48, 3.28, 4692),
    "G8-to-G9": (3424, 940, 214, 74.79, 20.53, 4.67, 4578),
    "G9-to-G10": (3407, 804, 149, 78.14, 18.44, 3.42, 4360),
}

DEV_FROZEN = {
    "XGBoost": (0.8111, 0.8133), "CatBoost": (0.8127, 0.8088),
    "ExtraTrees": (0.8106, 0.8068), "LightGBM": (0.8144, 0.8062),
    "RandomForest": (0.7995, 0.7986), "LogRegEN": (0.7622, 0.7573),
}
TEST_AUC_EXACT = {"ExtraTrees": 0.8193, "XGBoost": 0.8191}

COHORT_YEARS = {"m1": "2019-2021", "e4": "2022-2024"}
COHORT_N = {"m1": (5427, 2344, 268, 4.94), "e4": (5545, 2290, 249, 4.49)}
FROZEN_TEST_POS = 93

def load():
    df = pd.read_csv(os.path.join(RES, "sensitivity_results.csv"))
    out = {}

    def grab(analysis, scope):
        return {r.classifier: r
                for _, r in df[(df.analysis == analysis) & (df.scope == scope)].iterrows()}

    out["s1"] = grab("S1_fullspectrum_training", "full spectrum test")
    out["s2"] = grab("S2_fullspectrum_training", "extreme-group test")
    out["d2"] = grab("S2_spectrum_delta", "AUROC(extreme) - AUROC(full)")
    out["s3ex"] = grab("S3_restricted_training", "extreme-group test")
    out["s3fs"] = grab("S3_restricted_training", "full spectrum test")
    out["d3"] = grab("S3_spectrum_delta", "AUROC(extreme) - AUROC(full)")
    out["s4"] = df[df.analysis == "S4_three_category"].set_index("scope")["AUROC"].to_dict()
    out["s5"] = {}
    for _, r in df[df.analysis == "S5_continuous"].iterrows():
        v = r.get("value")
        out["s5"][r.scope] = r.AUROC if (v is None or pd.isna(v)) else v
    return out

def load_selection():
    S = pd.read_csv(os.path.join(RES, "selection_repeated_cv.csv"))
    F = pd.read_csv(os.path.join(RES, "selection_repeated_cv_folds.csv"))
    sel = S.loc[S["selected"], "classifier"].iloc[0]
    return dict(S=S, folds=F, selected=sel,
                rows={r.classifier: r for _, r in S.iterrows()},
                order=list(S.sort_values("mean_AUROC", ascending=False).classifier),
                n_folds=int(F.groupby("classifier").size().iloc[0]),
                thr=float(S.loc[S["selected"], "prespecified_threshold_rep1"].iloc[0]))

def load_robustness():
    R = pd.read_csv(os.path.join(RES, "robustness_results.csv"))
    holdout = {}
    for _, r in R[R.analysis == "R2_cohort_holdout"].iterrows():
        holdout[(r.train_cohort, r.classifier, r.threshold_source)] = r
    return dict(raw=R,
                seeds=R[R.analysis == "R1_seed_variation"],
                summary={r.classifier: r
                         for _, r in R[R.analysis == "R1_seed_summary"].iterrows()},
                holdout=holdout,
                per_year=R[R.analysis == "R3_per_year"].sort_values(
                    ["train_cohort", "outcome_year"]))

def load_performance():
    P = pd.read_csv(os.path.join(RES, "performance_extended.csv"))
    C = pd.read_csv(os.path.join(RES, "calibration.csv"))
    D = pd.read_csv(os.path.join(RES, "decision_curve.csv"))
    T = pd.read_csv(os.path.join(RES, "threshold_analysis.csv"))
    rec = {}
    for _, r in C[C.analysis == "P3_recalibration"].iterrows():
        rec[(r.variant, r.quantity)] = r
    return dict(ext={r.classifier: r for _, r in P.iterrows()},
                cal={r.quantity: r
                     for _, r in C[C.analysis == "P2_calibration"].iterrows()},
                rel=C[C.analysis == "P2_reliability"].sort_values("bin"),
                rec=rec,
                variants=list(dict.fromkeys(C[C.analysis == "P3_recalibration"].variant)),
                dca=D, thr={r.label: r for _, r in T.iterrows()}, thr_table=T)

def load_incremental():
    H = pd.read_csv(os.path.join(RES, "shap_full_hierarchy.csv"))
    I = pd.read_csv(os.path.join(RES, "incremental_value.csv"))
    B = pd.read_csv(os.path.join(RES, "raw_saps_benchmark.csv"))
    nested, delta = {}, {}
    for _, r in I[I.analysis == "I2_nested"].iterrows():
        nested[(r.algorithm, r.model)] = r
    for _, r in I[I.analysis == "I2_delta_vs_full"].iterrows():
        delta[(r.algorithm, r.model)] = r
    groups = H.groupby("group").mean_abs_SHAP.sum()
    total = float(H.mean_abs_SHAP.sum())
    return dict(hier=H, total_shap=total, group_shap=groups,
                group_share={g: 100 * v / total for g, v in groups.items()},
                nested=nested, delta=delta,
                bench={r.model: r
                       for _, r in B[B.analysis == "M0_benchmark"].iterrows()},
                bdelta={r.model: r
                        for _, r in B[B.analysis == "M0_delta_full_minus"].iterrows()},
                bcal=B[B.analysis == "M0_calibration"].iloc[0],
                bdca={r.label: r
                      for _, r in B[B.analysis == "M0_decision_curve"].iterrows()},
                bflag={r.model: r
                       for _, r in B[B.analysis == "M0_matched_flag"].iterrows()},
                sets=list(dict.fromkeys(I[I.analysis == "I2_nested"].model)))

def load_stratum():
    S = pd.read_csv(os.path.join(RES, "incremental_by_stratum.csv"))
    strata = {r.stratum: r for _, r in S[S.analysis == "stratum"].iterrows()}
    return dict(strata=strata, precision=S[S.analysis == "precision"].iloc[0],
                order=[k for k in ("low", "middle", "upper") if k in strata])

def load_audit():
    A = pd.read_csv(os.path.join(RES, "grouped_shap_audit.csv"))
    S = pd.read_csv(os.path.join(RES, "grouped_shap_stability.csv"))
    multi = A[(A.n_items >= 2) & A.summed_shap.notna()]
    ok = A.dropna(subset=["summed_shap", "perm_importance"])
    mm = ok[ok.n_items >= 2]
    return dict(A=A, S=S,
                rows={r.scale: r for _, r in A.iterrows()},
                stab={r.scale: r for _, r in S.iterrows()},
                n_scales=int(len(A)), n_multi=int(len(multi)),
                rho_count_summed=float(spearmanr(multi.n_items,
                                                 multi.summed_shap).statistic),
                p_count_summed=float(spearmanr(multi.n_items,
                                               multi.summed_shap).pvalue),
                rho_count_peritem=float(spearmanr(multi.n_items,
                                                  multi.per_item_shap).statistic),
                p_count_peritem=float(spearmanr(multi.n_items,
                                                multi.per_item_shap).pvalue),
                rho_count_perm=float(spearmanr(mm.n_items,
                                               mm.perm_importance).statistic),
                p_count_perm=float(spearmanr(mm.n_items,
                                             mm.perm_importance).pvalue),
                rho_summed_perm=float(spearmanr(ok.summed_shap,
                                                ok.perm_importance).statistic),
                rho_peritem_perm=float(spearmanr(ok.per_item_shap,
                                                 ok.perm_importance).statistic),
                perm_order=list(A.sort_values("perm_importance",
                                              ascending=False).scale))

def load_provenance():
    M = pd.read_csv(os.path.join(RES, "provenance_missingness.csv"))
    E = pd.read_csv(os.path.join(RES, "provenance_encoding.csv"))
    D = pd.read_csv(os.path.join(RES, "provenance_models.csv"))
    out = dict(
        M=M, E=E,
        mean_missing=100 * float(M.missing_overall.mean()),
        median_missing=100 * float(M.missing_overall.median()),
        max_missing=100 * float(M.missing_overall.max()),
        n_over20=int((M.missing_overall > 0.2).sum()),
        n_fully_missing_cohort=int(((M.missing_m1 > 0.999) |
                                    (M.missing_e4 > 0.999)).sum()),
        max_cohort_gap=100 * float(M.cohort_gap.max()),
        max_wave_range=100 * float(M.wave_range.max()),
        worst_feature=str(M.sort_values("missing_overall",
                                        ascending=False).feature.iloc[0]),
        kinds=E.kind.value_counts().to_dict(),
        n_nominal=int(E.likely_nominal.sum()),
        sig={r.target: r
             for _, r in D[D.analysis == "D3_missingness_signal"].iterrows()},
        sv={r.model: r for _, r in D[D.analysis == "D5_scale_vs_item"].iterrows()},
    )
    sel_path = os.path.join(RES, "provenance_selection.csv")
    if os.path.exists(sel_path):
        S = pd.read_csv(sel_path)
        out["sel"] = {r.classifier: r for _, r in S.iterrows()}
    return out

def load_encoding_experiment():
    X = pd.read_csv(os.path.join(RES, "encoding_experiment.csv"))
    lab = X[X.analysis == "E1_label_review"]
    nominal = lab[lab.verdict.str.contains("unordered|unlabelled", na=False)]
    return dict(X=X, label_review=lab,
                onehot={r.classifier: r
                        for _, r in X[X.analysis == "E2_onehot"].iterrows()},
                n_flagged=int(len(lab)), n_nominal=int(len(nominal)),
                nominal_features=list(nominal.feature))

def ci(row, metric="AUROC", d=3):
    return (f"{row[metric]:.{d}f} (95 % CI {row[metric + '_lo']:.{d}f}-"
            f"{row[metric + '_hi']:.{d}f})")

def num(row, metric, d=3):
    return f"{row[metric]:.{d}f}"

def load_comment8():
    import pickle
    S = pd.read_csv(os.path.join(RES, "sample_size.csv"))
    FC = pd.read_csv(os.path.join(RES, "fold_composition.csv"))
    H = pd.read_csv(os.path.join(RES, "hyperparameters.csv"))
    out = dict(S=S, FC=FC, H=H,
               ss={(int(r.parameters), r.r2_basis): r for _, r in S.iterrows()})
    lc_path = os.path.join(RES, "learning_curve.csv")
    if os.path.exists(lc_path):
        L = pd.read_csv(lc_path)
        out["L"] = L
        out["lc"] = (L.groupby(["representation", "fraction"])
                     .agg(n_train=("n_train", "mean"), n_pos=("n_pos_train", "mean"),
                          AUROC=("AUROC", "mean"), AUROC_min=("AUROC", "min"),
                          AUROC_max=("AUROC", "max"), PR_AUC=("PR_AUC", "mean"))
                     .reset_index())
    ns_path = os.path.join(RES, "nosmote_results.csv")
    if os.path.exists(ns_path):
        N = pd.read_csv(ns_path)
        out["N"] = N
        out["ns"] = {r.classifier: r for _, r in N.iterrows()}
    bundle = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "pipeline")
    boot = pickle.load(open(os.path.join(bundle, "03_results",
                                         "cluster_bootstrap_results.pkl"), "rb"))
    out["boot"] = boot
    tp = pd.read_csv(os.path.join(bundle, "01_data", "test_predictions.csv"))
    yt = tp["y"].astype(int).values if "y" in tp else tp["y_true"].astype(int).values
    out["frozen_brier"] = {c: float(np.mean((tp[c].values - yt) ** 2)) for c in CLFS}
    return out

def load_transition():
    R = pd.read_csv(os.path.join(RES, "transition_uncertainty.csv"))

    def auc(tag, score="XGBoost", **flt):
        sel = (R.analysis == tag) & (R.score == score)
        for k, v in flt.items():
            sel &= (R[k] == v)
        return {r.stratum: r for _, r in R[sel].iterrows()}

    def diff(tag, score="XGBoost", **flt):
        sel = (R.analysis == tag + "_diff") & (R.score == score)
        for k, v in flt.items():
            sel &= (R[k] == v)
        return {r.contrast: r for _, r in R[sel].iterrows()}

    def wald(tag, score="XGBoost", **flt):
        sel = (R.analysis == tag + "_wald") & (R.score == score)
        for k, v in flt.items():
            sel &= (R[k] == v)
        return R[sel].iloc[0]

    counts = {(r.partition, r.stratum, r.cohort): r
              for _, r in R[R.analysis == "T1_counts"].iterrows()}
    abl = {(r.partition, r.stratum): r for _, r in R[R.analysis == "T5_ablation"].iterrows()}
    traj = {r.stratum: r for _, r in R[R.analysis == "T5_trajectory_values"].iterrows()}
    return dict(R=R, auc=auc, diff=diff, wald=wald, counts=counts, abl=abl, traj=traj)

def load_subgroups():
    S = pd.read_csv(os.path.join(RES, "subgroup_fairness.csv"))
    row = {(r.analysis, r.group, r.level): r
           for _, r in S[~S.analysis.str.endswith("_diff")].iterrows()}
    diff = {(r.analysis.replace("_diff", ""), r.group, r.level, r.metric): r
            for _, r in S[S.analysis.str.endswith("_diff")].iterrows()}
    return dict(S=S, row=row, diff=diff)

def load_flow():
    F = pd.read_csv(os.path.join(RES, "flow_audit.csv"))

    def n(step, unit, panel="all"):
        s = F[F.step.str.startswith(step) & (F.unit == unit)]
        if panel == "all" and "all" not in set(s.panel):
            return int(s.n.sum())
        return int(s[s.panel == panel].n.iloc[0])
    return dict(F=F, n=n)

LABEL = {
    "Self-esteem": "Life satisfaction",
    "Depression": "Subjective happiness",
    "Social anxiety": "Self-esteem",
    "Psychological well-being": "Emotional and behavioral problems",
    "Peer relationship": "Cooperation",
    "Self-control": "Creative personality",
    "Aggression": "Grit",
    "Smartphone use": "Academic engagement and achievement",
    "Device access": "Smartphone use by purpose",
    "School engagement": "School life and relationships",
    "Family relationship": "Perceived parenting",
    "Parental mediation": "Caregiver smartphone use and dependence",
    "Parenting stress": "Caregiver psychological traits",
    "Parent-reported family": "Caregiver-reported family and child",
    "Parent education involvement": "Private tutoring",
    "Parent school involvement": "Parental educational attainment",
    "Parent work hours": "Parental work hours",
    "Parent health": "Health (caregiver report)",
    "Father age": "Adolescent birth year (household roster)",
    "Father info": "Household members' birth dates",
    "Mother info": "Adolescent birth date (caregiver report)",
    "Daily smartphone use": "SAPS adaptive-function disturbance",
}

def lab(name):
    return LABEL.get(name, name)

TITLE = ("Development and internal validation of an explainable machine-learning model "
         "to predict next-wave high-risk smartphone overdependence in adolescents: a "
         "longitudinal prediction-modeling study using the Korean Children and Youth "
         "Panel Survey 2018")

ITEM_EN = {
    "P_PMDA1C12": "Tried to cut down smartphone time but failed",
    "P_PMDA1C04": "Could not concentrate on current work because of smartphone use",
    "P_PMDA1C13": "Keeps using the smartphone while thinking of stopping",
    "P_PMDA1C14": "Spending much time on the smartphone has become a habit",
    "YINT2A08": "When waking up, wants to go to school for classes",
    "YINT1B00": "Satisfaction with last semester's grades",
    "YINT2B05": "Does not want to bother with studying",
    "YPSY4E09": "The future does not seem hopeful",
    "YPSY4A07": "Tends to leave out letters when writing",
    "YPSY4A06": "Finds it hard to sit still when studying",
    "P_PPSY2A04": "Degree of resemblance to people who are very unhappy overall",
    "P_PPSY4A27": "Describes self as quick-witted",
    "P_PPSY5A05": "Often sets a goal and then switches to another before reaching it",
    "YTIM1C02": "Weekend sleep quality",
    "YTIM1K01": "Weekday time spent playing with a smartphone",
    "YTIM1K02": "Weekend time spent playing with a smartphone",
    "YFAM2E01": "Parents show how to do things the adolescent tries to do",
    "YFAM2A02": "Parents like being with the adolescent",
    "YFAM2A01": "Parents express love for the adolescent",
    "YMDA1B08": "Frequency of smartphone use for TV and video",
    "YMDA1B01": "Frequency of smartphone use for calls with family",
    "YMDA1B09": "Frequency of smartphone use for music",
    "YPSY7A01": "Finds it hard to concentrate once another thought arises",
    "YPSY7A06": "Finds it hard to keep working on something that takes long",
    "YPSY7A03": "Has focused briefly on a problem and soon lost interest",
    "P_PSCHOOL2": "Mother's educational attainment",
    "P_PFBRTA1": "Birth year of household member 1 (the adolescent)",
}
