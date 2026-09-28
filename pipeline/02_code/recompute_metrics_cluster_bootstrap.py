import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

import sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, recall_score,
    precision_score, average_precision_score, brier_score_loss,
    confusion_matrix,
)

sys.stdout.reconfigure(encoding='utf-8')

ROOT = _paths.RERUN_DIR
df = pd.read_csv(_paths.TEST_PREDICTIONS)
with open(_paths.THRESHOLDS_JSON) as f:
    THR = json.load(f)

CLASSIFIERS = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'ExtraTrees', 'LogRegEN']
TRANSITIONS = ['G7->G8', 'G8->G9', 'G9->G10']
N_BOOT = 2000
RNG = np.random.default_rng(42)

def compute_metrics(y_true, prob, threshold):
    if len(np.unique(y_true)) < 2:
        return None
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return {
        'AUROC': roc_auc_score(y_true, prob),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'sensitivity': sens,
        'specificity': spec,
        'accuracy': accuracy_score(y_true, y_pred),
        'PPV': ppv,
        'NPV': npv,
        'PR_AUC': average_precision_score(y_true, prob),
        'Brier': brier_score_loss(y_true, prob),
    }

def cluster_bootstrap_ci(df_subset, clf, threshold, n_boot=N_BOOT, rng=RNG):
    point = compute_metrics(df_subset['y_true'].values,
                            df_subset[clf].values, threshold)
    if point is None:
        return None
    grouped = df_subset.groupby('ID').indices
    ids = np.array(list(grouped.keys()))
    n_ids = len(ids)
    boot_metrics = {k: [] for k in point}
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=n_ids, replace=True)
        rows = np.concatenate([grouped[i] for i in sampled])
        sub = df_subset.iloc[rows]
        m = compute_metrics(sub['y_true'].values, sub[clf].values, threshold)
        if m is None:
            continue
        for k, v in m.items():
            boot_metrics[k].append(v)
    ci = {}
    for k, arr in boot_metrics.items():
        arr = np.array(arr)
        if len(arr) == 0:
            ci[k] = (point[k], np.nan, np.nan)
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            ci[k] = (point[k], lo, hi)
    return ci

results = {}
print('Computing cluster bootstrap CI for 6 classifiers x 4 strata...')
for clf in CLASSIFIERS:
    thr = THR[clf]
    print(f'  {clf} (threshold={thr:.3f})...')
    overall = cluster_bootstrap_ci(df, clf, thr)
    per_trans = {}
    for t in TRANSITIONS:
        sub = df[df['TRANSITION'] == t]
        per_trans[t] = cluster_bootstrap_ci(sub, clf, thr)
    results[clf] = {'overall': overall, 'transitions': per_trans}

import pickle
with open(os.path.join(_paths.RERUN_DIR, 'cluster_bootstrap_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

print('\n=== Overall (all transitions) ===')
print(f'{"Classifier":<14} {"Metric":<12} {"Point":>7} {"95% CI":>20}')
print('-' * 60)
for clf, data in results.items():
    for metric, (p, lo, hi) in data['overall'].items():
        print(f'{clf:<14} {metric:<12} {p:>7.3f} {f"({lo:.3f}-{hi:.3f})":>20}')
    print()

print('\n=== Per-grade-transition AUROC ===')
print(f'{"Classifier":<14} {"Transition":<10} {"AUROC":>7} {"95% CI":>20}')
for clf, data in results.items():
    for t in TRANSITIONS:
        if data['transitions'][t]:
            p, lo, hi = data['transitions'][t]['AUROC']
            print(f'{clf:<14} {t:<10} {p:>7.3f} {f"({lo:.3f}-{hi:.3f})":>20}')
    print()
