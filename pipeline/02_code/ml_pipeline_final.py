from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths


import sys, types
if "numba" not in sys.modules:
    numba = types.ModuleType("numba"); numba.__path__ = []
    def _njit(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        def deco(f): return f
        return deco
    numba.njit = _njit; numba.jit = _njit; numba.prange = range
    typed = types.ModuleType("numba.typed")
    class _List(list):
        @classmethod
        def empty_list(cls, *a, **k): return cls()
    typed.List = _List; typed.Dict = dict
    core = types.ModuleType("numba.core"); core.__path__ = []
    coretypes = types.ModuleType("numba.core.types")
    coretypes.int64 = int; coretypes.float64 = float
    sys.modules.update({"numba": numba, "numba.typed": typed,
                        "numba.core": core, "numba.core.types": coretypes})
    numba.typed = typed; numba.core = core

import io, os, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", write_through=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, brier_score_loss, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             accuracy_score)
from sklearn.calibration import calibration_curve

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import shap

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED            = 42
N_TRIALS        = 80
N_FOLDS_INNER   = 3
N_FOLDS_OUTER   = 5
TEST_SIZE       = 0.20
OUT             = _paths.RERUN_DIR
FIG_DIR         = _paths.RERUN_FIGS
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print(">>> Loading transitions.pkl")
df = _paths.load_transitions()
feat_cols = [c for c in df.columns if c not in ("y", "ID", "TRANSITION")]
X_all     = df[feat_cols].astype(float).values
y_all     = df["y"].astype(int).values
ids_all   = df["ID"].astype(int).values
trans_all = df["TRANSITION"].values

print(f"    transitions = {len(df)}")
print(f"    features    = {len(feat_cols)}")
print(f"    positive prevalence = {y_all.mean()*100:.2f}%")

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
tr_idx, te_idx = next(gss.split(X_all, y_all, groups=ids_all))
X_tr, X_te = X_all[tr_idx], X_all[te_idx]
y_tr, y_te = y_all[tr_idx], y_all[te_idx]
g_tr       = ids_all[tr_idx]
t_te       = trans_all[te_idx]

print(f"    train n={len(y_tr)} (pos={y_tr.sum()}, {y_tr.mean()*100:.2f}%)")
print(f"    test  n={len(y_te)} (pos={y_te.sum()}, {y_te.mean()*100:.2f}%)")

def pipeline_tree(model):
    return ImbPipeline([
        ("imp",       SimpleImputer(strategy="median")),
        ("smote_enn", SMOTEENN(random_state=SEED)),
        ("clf",       model),
    ])

def pipeline_linear(model):
    return ImbPipeline([
        ("imp",       SimpleImputer(strategy="median")),
        ("sc",        StandardScaler()),
        ("smote_enn", SMOTEENN(random_state=SEED)),
        ("clf",       model),
    ])

def cv_oof(pipe_factory, params, X, y, groups, n_folds):
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    for tr, va in skf.split(X, y, groups=groups):
        pipe = pipe_factory(params)
        pipe.fit(X[tr], y[tr])
        oof[va] = pipe.predict_proba(X[va])[:, 1]
    return oof

def space_xgb(trial):
    return dict(
        n_estimators     = trial.suggest_int  ("n_estimators",     200, 1200),
        max_depth        = trial.suggest_int  ("max_depth",        3, 10),
        learning_rate    = trial.suggest_float("learning_rate",    1e-3, 0.3, log=True),
        subsample        = trial.suggest_float("subsample",        0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int  ("min_child_weight", 1, 20),
        gamma            = trial.suggest_float("gamma",            0.0, 5.0),
        reg_alpha        = trial.suggest_float("reg_alpha",        1e-8, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda",       1e-8, 10.0, log=True),
    )

def build_xgb(params):
    return xgb.XGBClassifier(
        **params, random_state=SEED, eval_metric="auc",
        tree_method="hist", n_jobs=-1, verbosity=0)

def space_lgb(trial):
    return dict(
        n_estimators      = trial.suggest_int  ("n_estimators",      200, 1200),
        num_leaves        = trial.suggest_int  ("num_leaves",        15, 255),
        max_depth         = trial.suggest_int  ("max_depth",         -1, 12),
        learning_rate     = trial.suggest_float("learning_rate",     1e-3, 0.3, log=True),
        min_child_samples = trial.suggest_int  ("min_child_samples", 5, 100),
        subsample         = trial.suggest_float("subsample",         0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.5, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha",         1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",        1e-8, 10.0, log=True),
    )

def build_lgb(params):
    return lgb.LGBMClassifier(
        **params, random_state=SEED, n_jobs=-1, verbose=-1)

def space_cat(trial):
    return dict(
        iterations          = trial.suggest_int  ("iterations",          200, 1200),
        depth               = trial.suggest_int  ("depth",               4, 10),
        learning_rate       = trial.suggest_float("learning_rate",       1e-3, 0.3, log=True),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg",         1.0, 10.0),
        border_count        = trial.suggest_int  ("border_count",        32, 255),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength     = trial.suggest_float("random_strength",     0.0, 10.0),
    )

def build_cat(params):
    return cb.CatBoostClassifier(
        **params, random_seed=SEED, verbose=0,
        allow_writing_files=False, thread_count=-1)

def space_rf(trial):
    return dict(
        n_estimators      = trial.suggest_int  ("n_estimators",      200, 800),
        max_depth         = trial.suggest_int  ("max_depth",         5, 25),
        min_samples_split = trial.suggest_int  ("min_samples_split", 2, 30),
        min_samples_leaf  = trial.suggest_int  ("min_samples_leaf",  1, 20),
        max_features      = trial.suggest_float("max_features",      0.1, 1.0),
    )

def build_rf(params):
    return RandomForestClassifier(**params, random_state=SEED, n_jobs=-1)

def space_et(trial):
    return dict(
        n_estimators      = trial.suggest_int  ("n_estimators",      200, 800),
        max_depth         = trial.suggest_int  ("max_depth",         5, 25),
        min_samples_split = trial.suggest_int  ("min_samples_split", 2, 30),
        min_samples_leaf  = trial.suggest_int  ("min_samples_leaf",  1, 20),
        max_features      = trial.suggest_float("max_features",      0.1, 1.0),
    )

def build_et(params):
    return ExtraTreesClassifier(**params, random_state=SEED, n_jobs=-1)

def space_lr(trial):
    return dict(
        C        = trial.suggest_float("C",        1e-3, 10.0, log=True),
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0),
    )

def build_lr(params):
    return LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=3000,
        random_state=SEED, n_jobs=-1, **params)

MODELS = {
    "XGBoost":      ("tree",   space_xgb, build_xgb),
    "LightGBM":     ("tree",   space_lgb, build_lgb),
    "CatBoost":     ("tree",   space_cat, build_cat),
    "RandomForest": ("tree",   space_rf,  build_rf),
    "ExtraTrees":   ("tree",   space_et,  build_et),
    "LogRegEN":     ("linear", space_lr,  build_lr),
}

def tune_threshold_f1(y_true, y_prob):
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(f1s))
    return float(grid[best]), float(f1s[best])

def bootstrap_ci(metric_fn, y_true, y_prob, n_boot=500, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            continue
    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def full_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob)
    auc_lo, auc_hi = bootstrap_ci(roc_auc_score, y_true, y_prob)
    pr  = average_precision_score(y_true, y_prob)
    pr_lo, pr_hi   = bootstrap_ci(average_precision_score, y_true, y_prob)
    return {
        "AUC": auc, "AUC_lo": auc_lo, "AUC_hi": auc_hi,
        "PR_AUC": pr, "PR_lo": pr_lo, "PR_hi": pr_hi,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "PPV":         tp / (tp + fp) if (tp + fp) else np.nan,
        "NPV":         tn / (tn + fn) if (tn + fn) else np.nan,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "threshold": thr,
    }

def factory_for(kind, build_fn):
    def make(params):
        m = build_fn(params)
        return pipeline_tree(m) if kind == "tree" else pipeline_linear(m)
    return make

results        = {}
best_params    = {}
thresholds     = {}
test_probs     = {}

t_start = time.time()
for name, (kind, space_fn, build_fn) in MODELS.items():
    print(f"\n{'='*60}\n[{name}] Optuna {N_TRIALS} trials\n{'='*60}")
    pipe_factory = factory_for(kind, build_fn)

    def objective(trial, _factory=pipe_factory, _space=space_fn, _name=name):
        params = _space(trial)
        oof = cv_oof(_factory, params, X_tr, y_tr, g_tr, n_folds=N_FOLDS_INNER)
        return roc_auc_score(y_tr, oof)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=name,
    )
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    best = study.best_trial
    print(f"  best CV-AUC = {best.value:.4f}")
    best_params[name] = best.params

    oof = cv_oof(pipe_factory, best.params, X_tr, y_tr, g_tr,
                 n_folds=N_FOLDS_OUTER)
    thr, f1_at = tune_threshold_f1(y_tr, oof)
    thresholds[name] = thr
    print(f"  OOF AUC={roc_auc_score(y_tr, oof):.4f}  thr*={thr:.3f}  F1@thr={f1_at:.4f}")

    pipe_final = pipe_factory(best.params)
    pipe_final.fit(X_tr, y_tr)
    p_te = pipe_final.predict_proba(X_te)[:, 1]
    test_probs[name] = p_te

    metrics_te = full_metrics(y_te, p_te, thr)

    per_trans = {}
    for tr_name in np.unique(t_te):
        mask = (t_te == tr_name)
        if mask.sum() > 0 and (y_te[mask] == 1).any():
            per_trans[tr_name] = full_metrics(y_te[mask], p_te[mask], thr)

    results[name] = {
        "cv_auc":         float(best.value),
        "oof_auc":        float(roc_auc_score(y_tr, oof)),
        "test_metrics":   metrics_te,
        "per_transition": per_trans,
        "best_params":    best.params,
        "threshold":      thr,
        "study":          study,
        "pipe":           pipe_final,
    }
    print(f"  Test AUC = {metrics_te['AUC']:.4f} "
          f"[{metrics_te['AUC_lo']:.4f}-{metrics_te['AUC_hi']:.4f}]  "
          f"PR-AUC = {metrics_te['PR_AUC']:.4f}  "
          f"F1 = {metrics_te['F1']:.4f}  Brier = {metrics_te['Brier']:.4f}")

print(f"\n>>> All models complete in {(time.time()-t_start)/60:.1f} min")

rows = []
for name, r in results.items():
    row = {"Model": name, "CV_AUC": r["cv_auc"], "OOF_AUC": r["oof_auc"]}
    row.update({f"Test_{k}": v for k, v in r["test_metrics"].items()})
    rows.append(row)
perf = pd.DataFrame(rows).sort_values("Test_AUC", ascending=False)
perf.to_csv(f"{OUT}/model_performance.csv", index=False, encoding="utf-8-sig")

trans_rows = []
for name, r in results.items():
    for tr, m in r["per_transition"].items():
        row = {"Model": name, "Transition": tr}
        row.update({k: v for k, v in m.items()})
        trans_rows.append(row)
pd.DataFrame(trans_rows).to_csv(f"{OUT}/transition_performance.csv",
                                index=False, encoding="utf-8-sig")

with open(f"{OUT}/best_hyperparameters.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=False, indent=2)
with open(f"{OUT}/thresholds.json", "w", encoding="utf-8") as f:
    json.dump(thresholds, f, indent=2)

print("\n>>> Generating figures")

plt.figure(figsize=(8, 7))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_te, test_probs[name])
    plt.plot(fpr, tpr, lw=1.8,
             label=f"{name} (AUC={r['test_metrics']['AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC — held-out test set")
plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig1_roc.png", dpi=300); plt.close()

plt.figure(figsize=(8, 7))
for name, r in results.items():
    prec, rec, _ = precision_recall_curve(y_te, test_probs[name])
    plt.plot(rec, prec, lw=1.8,
             label=f"{name} (AP={r['test_metrics']['PR_AUC']:.3f})")
plt.axhline(y_te.mean(), color="k", linestyle="--", alpha=0.4,
            label=f"prevalence={y_te.mean():.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall — held-out test set")
plt.legend(loc="best"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig2_pr.png", dpi=300); plt.close()

plt.figure(figsize=(8, 7))
for name in results:
    frac_pos, mean_pred = calibration_curve(y_te, test_probs[name],
                                            n_bins=10, strategy="quantile")
    plt.plot(mean_pred, frac_pos, marker="o", lw=1.5, label=name)
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
plt.title("Calibration — held-out test set")
plt.legend(loc="best"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig3_calibration.png", dpi=300); plt.close()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (name, r) in zip(axes.flat, results.items()):
    vals = [t.value for t in r["study"].trials if t.value is not None]
    best_so_far = np.maximum.accumulate(vals)
    ax.plot(vals, alpha=0.4, label="trial")
    ax.plot(best_so_far, lw=2, label="best so far")
    ax.set_title(f"{name} (best CV-AUC={max(vals):.4f})")
    ax.set_xlabel("Trial"); ax.set_ylabel("CV-AUC")
    ax.grid(alpha=0.3); ax.legend()
plt.suptitle(f"Optuna optimisation history ({N_TRIALS} trials per model)")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig4_optuna.png", dpi=300); plt.close()

trans_df = pd.DataFrame(trans_rows)
plt.figure(figsize=(10, 6))
pivot = trans_df.pivot(index="Transition", columns="Model", values="AUC")
pivot.plot(kind="bar", ax=plt.gca())
plt.ylabel("AUC (held-out test)"); plt.xticks(rotation=0)
plt.title("Per-grade-transition AUC on held-out test set")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig5_transition_auc.png", dpi=300); plt.close()

best_name = perf.iloc[0]["Model"]
print(f"\n>>> SHAP analysis on best model = {best_name}")
best_pipe = results[best_name]["pipe"]
best_clf  = best_pipe.named_steps["clf"]
X_te_t = best_pipe.named_steps["imp"].transform(X_te)
if "sc" in best_pipe.named_steps:
    X_te_t = best_pipe.named_steps["sc"].transform(X_te_t)

rng = np.random.default_rng(SEED)
sample_n = min(1000, len(X_te_t))
sample_idx = rng.choice(len(X_te_t), size=sample_n, replace=False)
X_sample = X_te_t[sample_idx]

try:
    if best_name in ("XGBoost", "LightGBM", "CatBoost",
                     "RandomForest", "ExtraTrees"):
        explainer = shap.TreeExplainer(best_clf)
        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif shap_vals.ndim == 3:
            shap_vals = shap_vals[..., 1]
    else:
        explainer = shap.LinearExplainer(best_clf, X_tr[:min(500, len(X_tr))])
        shap_vals = explainer.shap_values(X_sample)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    imp = pd.Series(mean_abs, index=feat_cols).sort_values(ascending=False)
    imp.head(25).to_csv(f"{OUT}/shap_top25.csv", encoding="utf-8-sig")

    plt.figure(figsize=(9, 8))
    top = imp.head(25)[::-1]
    plt.barh(range(len(top)), top.values, color="#3B75AF")
    plt.yticks(range(len(top)), top.index)
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"SHAP top-25 features — {best_name}")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig6_shap.png", dpi=300); plt.close()

    plt.figure()
    shap.summary_plot(shap_vals, X_sample, feature_names=feat_cols,
                      show=False, max_display=20)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig7_shap_beeswarm.png",
                                    dpi=300); plt.close()
    print(f"    SHAP figures saved to {FIG_DIR}/")
except Exception as e:
    print(f"    SHAP failed: {e}")

proba_df = pd.DataFrame(test_probs)
proba_df["y_true"]     = y_te
proba_df["TRANSITION"] = t_te
proba_df["ID"]         = ids_all[te_idx]
proba_df.to_csv(f"{OUT}/test_predictions.csv", index=False, encoding="utf-8-sig")

print(f"\n>>> Held-out performance (sorted by Test AUC)")
print(perf[["Model", "CV_AUC", "OOF_AUC", "Test_AUC",
            "Test_AUC_lo", "Test_AUC_hi",
            "Test_PR_AUC", "Test_F1", "Test_MCC", "Test_Brier",
            "Test_Sensitivity", "Test_Specificity"]].round(4).to_string(index=False))
print(f"\n>>> Total elapsed: {(time.time()-t_start)/60:.1f} min")
print(f">>> Outputs written to {OUT}/")
