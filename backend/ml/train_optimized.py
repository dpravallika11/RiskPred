"""
IEEE-CIS Fraud Detection — Optimized Training Pipeline

Improvements over baseline:
1. Fixed preprocessor (handles __MISSING__ correctly)
2. Enhanced feature engineering (time-of-day, frequency signals)
3. XGBoost hyperparameter optimization via Bayesian-style search
4. Proper threshold selection on validation data
5. Full held-out test evaluation

Usage:
    cd backend
    python -m ml.train_optimized
"""

import os
import sys
import json
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

from ml.preprocessing import FraudPreprocessor

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data():
    print("Loading data...")
    txn = pd.read_csv(os.path.join(DATA_DIR, 'train_transaction.csv'))
    try:
        identity = pd.read_csv(os.path.join(DATA_DIR, 'train_identity.csv'))
    except FileNotFoundError:
        identity = None
    print(f"  Transactions: {txn.shape[0]:,} rows, {txn.shape[1]} cols")
    if identity is not None:
        print(f"  Identity: {identity.shape[0]:,} rows, {identity.shape[1]} cols")
    return txn, identity


def temporal_split(df, train_ratio=0.85):
    df_sorted = df.sort_values('TransactionDT').reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def cost_sensitive_threshold(y_true, y_prob, fn_cost=500, fp_cost=50):
    thresholds = np.arange(0.05, 0.95, 0.005)
    best_cost = float('inf')
    best_threshold = 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        total_cost = fn * fn_cost + fp * fp_cost
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(t)
    return best_threshold, best_cost


def evaluate_model(model, X_val, y_val, threshold):
    proba = model.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)
    y_pred = (proba >= threshold).astype(int)
    p = precision_score(y_val, y_pred, zero_division=0)
    r = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return {
        'roc_auc': roc, 'pr_auc': pr_auc, 'precision': p,
        'recall': r, 'f1': f1, 'fpr': fpr,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def sweep_thresholds(y_true, y_prob):
    thresholds = np.arange(0.10, 0.91, 0.02)
    results = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        results.append({
            'threshold': round(float(t), 4),
            'precision': round(float(p), 4),
            'recall': round(float(r), 4),
            'f1': round(float(f1), 4),
            'fpr': round(float(fpr), 4),
            'fnr': round(float(fnr), 4),
        })
    return results


def main():
    print("=" * 70)
    print("  IEEE-CIS FRAUD DETECTION — OPTIMIZED TRAINING PIPELINE")
    print("=" * 70)

    txn, identity = load_data()

    print("\nPerforming temporal train/val split...")
    txn_train, txn_val = temporal_split(txn, train_ratio=0.85)

    fraud_train = txn_train['isFraud'].sum()
    fraud_val = txn_val['isFraud'].sum()
    print(f"  Train: {len(txn_train):,} samples ({fraud_train:,} fraud, {fraud_train/len(txn_train)*100:.2f}%)")
    print(f"  Val:   {len(txn_val):,} samples ({fraud_val:,} fraud, {fraud_val/len(txn_val)*100:.2f}%)")

    identity_train = None
    identity_val = None
    if identity is not None:
        train_ids = set(txn_train['TransactionID'].values)
        val_ids = set(txn_val['TransactionID'].values)
        identity_train = identity[identity['TransactionID'].isin(train_ids)]
        identity_val = identity[identity['TransactionID'].isin(val_ids)]
        print(f"  Identity (train): {len(identity_train):,} matched")
        print(f"  Identity (val):   {len(identity_val):,} matched")

    print("\nFitting preprocessor on training data...")
    preprocessor = FraudPreprocessor()
    preprocessor.fit(txn_train, identity_train)
    print(f"  Final feature count: {len(preprocessor.feature_columns)}")

    print("\nTransforming data...")
    X_train = preprocessor.transform(txn_train, identity_train)
    y_train = txn_train['isFraud'].values
    X_val = preprocessor.transform(txn_val, identity_val)
    y_val = txn_val['isFraud'].values

    feature_names = preprocessor.get_feature_names()
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"  Class ratio (neg/pos): {ratio:.2f}")
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

    # ============================================================
    # PHASE A: Baseline models with fixed preprocessor
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE A: BASELINE MODELS (Fixed Preprocessor)")
    print("=" * 70)

    baseline_models = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000, class_weight='balanced', C=0.1, random_state=42
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight='balanced',
            random_state=42, n_jobs=-1, min_samples_leaf=20
        )),
        ("XGBoost (Baseline)", XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.05,
            scale_pos_weight=ratio * 0.5, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=3,
            eval_metric='aucpr', tree_method='hist',
            random_state=42, n_jobs=-1, reg_alpha=0.1, reg_lambda=1.0
        )),
    ]

    baseline_results = []
    for name, model in baseline_models:
        print(f"\n  Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        cost_thresh, _ = cost_sensitive_threshold(y_val, model.predict_proba(X_val)[:, 1])
        metrics = evaluate_model(model, X_val, y_val, cost_thresh)
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | "
              f"F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} "
              f"| threshold: {cost_thresh:.3f} | {elapsed:.1f}s")
        baseline_results.append({
            'name': name, 'model': model, 'threshold': cost_thresh, **metrics
        })

    # ============================================================
    # PHASE B: XGBoost hyperparameter optimization
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE B: XGBOOST HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [5, 7],
        'learning_rate': [0.03, 0.05, 0.08],
        'min_child_weight': [2, 3, 5],
        'subsample': [0.8, 0.85],
        'colsample_bytree': [0.7, 0.8],
        'reg_alpha': [0.05, 0.1],
        'reg_lambda': [0.5, 1.0],
        'scale_pos_weight': [ratio * 0.5, ratio * 0.75, ratio],
    }

    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*[param_grid[k] for k in keys]))

    np.random.seed(42)
    n_candidates = 15
    sampled_indices = np.random.choice(len(all_combos), size=min(n_candidates, len(all_combos)), replace=False)
    candidates = [all_combos[i] for i in sampled_indices]

    print(f"  Searching {len(candidates)} hyperparameter combinations...")

    best_xgb_score = -1
    best_xgb_params = None
    best_xgb_model = None
    best_xgb_thresh = 0.5

    for idx, combo in enumerate(candidates):
        params = dict(zip(keys, combo))
        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            scale_pos_weight=params['scale_pos_weight'],
            eval_metric='aucpr', tree_method='hist',
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, proba)
        cost_thresh, _ = cost_sensitive_threshold(y_val, proba)
        metrics = evaluate_model(model, X_val, y_val, cost_thresh)

        if pr_auc > best_xgb_score:
            best_xgb_score = pr_auc
            best_xgb_params = params
            best_xgb_model = model
            best_xgb_thresh = cost_thresh
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"    [{idx+1}/{len(candidates)}] PR-AUC: {pr_auc:.4f} | "
                      f"F1: {metrics['f1']:.4f} | depth={params['max_depth']} "
                      f"lr={params['learning_rate']} n_est={params['n_estimators']} "
                      f"spw={params['scale_pos_weight']:.1f}")

    print(f"\n  Best XGBoost PR-AUC (val): {best_xgb_score:.4f}")
    print(f"  Params: {json.dumps(best_xgb_params, indent=4)}")
    print(f"  Threshold: {best_xgb_thresh:.3f}")

    best_xgb_metrics = evaluate_model(best_xgb_model, X_val, y_val, best_xgb_thresh)
    print(f"  Val metrics: P={best_xgb_metrics['precision']:.4f} R={best_xgb_metrics['recall']:.4f} "
          f"F1={best_xgb_metrics['f1']:.4f} ROC={best_xgb_metrics['roc_auc']:.4f}")

    # ============================================================
    # PHASE C: Threshold sweep on best XGBoost
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE C: THRESHOLD SWEEP (Best XGBoost on Validation)")
    print("=" * 70)

    best_val_proba = best_xgb_model.predict_proba(X_val)[:, 1]
    threshold_sweep = sweep_thresholds(y_val, best_val_proba)

    print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10} {'FNR':>10}")
    print("  " + "-" * 60)
    for row in threshold_sweep:
        marker = " <-- chosen" if abs(row['threshold'] - best_xgb_thresh) < 0.006 else ""
        print(f"  {row['threshold']:>10.2f} {row['precision']:>10.4f} {row['recall']:>10.4f} "
              f"{row['f1']:>10.4f} {row['fpr']:>10.4f} {row['fnr']:>10.4f}{marker}")

    best_f1_thresh = max(threshold_sweep, key=lambda x: x['f1'])
    best_precision_thresh = max(
        [r for r in threshold_sweep if r['recall'] >= 0.50],
        key=lambda x: x['precision'],
        default=threshold_sweep[-1]
    )
    best_recall_thresh = max(threshold_sweep, key=lambda x: x['recall'])

    print(f"\n  High-Recall:   threshold={best_recall_thresh['threshold']:.2f} "
          f"(R={best_recall_thresh['recall']:.4f}, P={best_recall_thresh['precision']:.4f}, "
          f"F1={best_recall_thresh['f1']:.4f})")
    print(f"  Balanced (F1): threshold={best_f1_thresh['threshold']:.2f} "
          f"(R={best_f1_thresh['recall']:.4f}, P={best_f1_thresh['precision']:.4f}, "
          f"F1={best_f1_thresh['f1']:.4f})")
    print(f"  High-Prec:     threshold={best_precision_thresh['threshold']:.2f} "
          f"(R={best_precision_thresh['recall']:.4f}, P={best_precision_thresh['precision']:.4f}, "
          f"F1={best_precision_thresh['f1']:.4f})")

    # ============================================================
    # PHASE D: Final held-out test evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE D: HELD-OUT TEST EVALUATION")
    print("=" * 70)

    test_df = pd.read_csv(os.path.join(DATA_DIR, 'held_out_test_set.csv'))
    print(f"  Test set: {test_df.shape[0]:,} rows, {test_df.shape[1]} cols")
    y_test = test_df['isFraud'].values
    fraud_count = int(y_test.sum())
    print(f"  Fraud: {fraud_count:,} ({fraud_count/len(y_test)*100:.2f}%)")

    test_for_preprocess = test_df.drop(columns=['isFraud'], errors='ignore')
    if 'TransactionID' in test_for_preprocess.columns:
        test_for_preprocess = test_for_preprocess.drop(columns=['TransactionID'], errors='ignore')

    X_test = preprocessor.transform(test_for_preprocess)
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_names]

    test_proba = best_xgb_model.predict_proba(X_test)[:, 1]

    test_roc = roc_auc_score(y_test, test_proba)
    test_pr = average_precision_score(y_test, test_proba)

    # Evaluate at cost-optimized threshold (selected on validation)
    y_pred = (test_proba >= best_xgb_thresh).astype(int)
    test_p = precision_score(y_test, y_pred, zero_division=0)
    test_r = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    test_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  FINAL HELD-OUT TEST RESULTS (threshold={best_xgb_thresh:.3f})")
    print(f"  {'Metric':<25} {'Value':>10}")
    print("  " + "-" * 35)
    print(f"  {'Precision':<25} {test_p:>10.4f}")
    print(f"  {'Recall':<25} {test_r:>10.4f}")
    print(f"  {'F1 Score':<25} {test_f1:>10.4f}")
    print(f"  {'ROC-AUC':<25} {test_roc:>10.4f}")
    print(f"  {'PR-AUC':<25} {test_pr:>10.4f}")
    print(f"  {'FPR':<25} {test_fpr:>10.4f}")
    print(f"  {'FNR':<25} {test_fnr:>10.4f}")
    print(f"\n  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Also evaluate at F1-optimal and high-recall thresholds
    for label, t in [("Balanced(F1)", best_f1_thresh['threshold']),
                      ("High-Recall", best_recall_thresh['threshold'])]:
        yp = (test_proba >= t).astype(int)
        tp_ = precision_score(y_test, yp, zero_division=0)
        tr_ = recall_score(y_test, yp, zero_division=0)
        tf1_ = f1_score(y_test, yp, zero_division=0)
        print(f"  [{label} t={t:.2f}] P={tp_:.4f} R={tr_:.4f} F1={tf1_:.4f}")

    # ============================================================
    # PHASE E: Compare all models on test set
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE E: MODEL COMPARISON ON HELD-OUT TEST SET")
    print("=" * 70)

    print(f"\n  {'Model':<25} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'PR-AUC':>10} {'ROC-AUC':>10}")
    print("  " + "-" * 85)

    all_test_results = []

    # Baseline XGBoost on test
    base_xgb = baseline_results[2]['model']
    base_thresh = baseline_results[2]['threshold']
    base_proba = base_xgb.predict_proba(X_test)[:, 1]
    base_yp = (base_proba >= base_thresh).astype(int)
    base_metrics_test = {
        'name': 'XGBoost (Baseline)',
        'threshold': base_thresh,
        'precision': precision_score(y_test, base_yp, zero_division=0),
        'recall': recall_score(y_test, base_yp, zero_division=0),
        'f1': f1_score(y_test, base_yp, zero_division=0),
        'pr_auc': average_precision_score(y_test, base_proba),
        'roc_auc': roc_auc_score(y_test, base_proba),
    }
    all_test_results.append(base_metrics_test)
    print(f"  {base_metrics_test['name']:<25} {base_metrics_test['threshold']:>10.3f} "
          f"{base_metrics_test['precision']:>10.4f} {base_metrics_test['recall']:>10.4f} "
          f"{base_metrics_test['f1']:>10.4f} {base_metrics_test['pr_auc']:>10.4f} "
          f"{base_metrics_test['roc_auc']:>10.4f}")

    # Optimized XGBoost on test
    opt_metrics_test = {
        'name': 'XGBoost (Optimized)',
        'threshold': best_xgb_thresh,
        'precision': test_p,
        'recall': test_r,
        'f1': test_f1,
        'pr_auc': test_pr,
        'roc_auc': test_roc,
    }
    all_test_results.append(opt_metrics_test)
    print(f"  {opt_metrics_test['name']:<25} {opt_metrics_test['threshold']:>10.3f} "
          f"{opt_metrics_test['precision']:>10.4f} {opt_metrics_test['recall']:>10.4f} "
          f"{opt_metrics_test['f1']:>10.4f} {opt_metrics_test['pr_auc']:>10.4f} "
          f"{opt_metrics_test['roc_auc']:>10.4f}")

    # Optimized XGBoost at F1-optimal threshold
    opt_f1_yp = (test_proba >= best_f1_thresh['threshold']).astype(int)
    opt_f1_metrics = {
        'name': 'XGBoost (F1-Optimal)',
        'threshold': best_f1_thresh['threshold'],
        'precision': precision_score(y_test, opt_f1_yp, zero_division=0),
        'recall': recall_score(y_test, opt_f1_yp, zero_division=0),
        'f1': f1_score(y_test, opt_f1_yp, zero_division=0),
        'pr_auc': test_pr,
        'roc_auc': test_roc,
    }
    all_test_results.append(opt_f1_metrics)
    print(f"  {opt_f1_metrics['name']:<25} {opt_f1_metrics['threshold']:>10.3f} "
          f"{opt_f1_metrics['precision']:>10.4f} {opt_f1_metrics['recall']:>10.4f} "
          f"{opt_f1_metrics['f1']:>10.4f} {opt_f1_metrics['pr_auc']:>10.4f} "
          f"{opt_f1_metrics['roc_auc']:>10.4f}")

    # ============================================================
    # PHASE F: Decision — old vs new
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE F: OLD vs NEW COMPARISON")
    print("=" * 70)

    old_eval_path = os.path.join(ARTIFACTS_DIR, 'evaluation_metrics.json')
    with open(old_eval_path) as f:
        old_eval = json.load(f)

    old_metrics = {
        'precision': old_eval['precision'],
        'recall': old_eval['recall'],
        'f1': old_eval['f1'],
        'pr_auc': old_eval['pr_auc'],
        'roc_auc': old_eval['roc_auc'],
        'fpr': old_eval['fpr'],
        'fnr': old_eval['fnr'],
        'threshold': old_eval['threshold'],
    }

    new_metrics = opt_metrics_test

    print(f"\n  {'Metric':<15} {'OLD':>12} {'NEW':>12} {'Delta':>12}")
    print("  " + "-" * 51)
    for key in ['precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'fpr', 'fnr']:
        old_val = old_metrics[key]
        new_val = new_metrics[key]
        delta = new_val - old_val
        pct = (delta / old_val * 100) if old_val != 0 else 0
        marker = "+" if delta > 0 else ""
        print(f"  {key:<15} {old_val:>12.4f} {new_val:>12.4f} {marker}{delta:>10.4f} ({pct:>+.1f}%)")
    print(f"  {'threshold':<15} {old_metrics['threshold']:>12.3f} {new_metrics['threshold']:>12.3f}")

    # ============================================================
    # PHASE G: Save artifacts
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE G: SAVING ARTIFACTS")
    print("=" * 70)

    joblib.dump(best_xgb_model, os.path.join(ARTIFACTS_DIR, 'fraud_model.joblib'))
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'))

    with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)

    threshold_config = {
        'threshold': best_xgb_thresh,
        'method': 'cost_sensitive',
        'fn_cost_assumption': 500,
        'fp_cost_assumption': 50,
        'rationale': (
            'Threshold optimized to minimize total expected cost on validation set. '
            'False Negative cost ($500) reflects average fraud loss. '
            'False Positive cost ($50) reflects customer friction and manual review overhead.'
        ),
        'best_f1_threshold': best_f1_thresh['threshold'],
        'validation_roc_auc': best_xgb_metrics['roc_auc'],
        'validation_pr_auc': best_xgb_metrics['pr_auc'],
    }
    with open(os.path.join(ARTIFACTS_DIR, 'threshold_config.json'), 'w') as f:
        json.dump(threshold_config, f, indent=2)

    metadata = {
        'model': 'XGBClassifier',
        'model_params': best_xgb_model.get_params(),
        'features': feature_names,
        'feature_count': len(feature_names),
        'threshold': best_xgb_thresh,
        'threshold_method': 'cost_sensitive_optimization',
        'validation_metrics': {
            'roc_auc': best_xgb_metrics['roc_auc'],
            'pr_auc': best_xgb_metrics['pr_auc'],
            'f1': best_xgb_metrics['f1'],
            'precision': best_xgb_metrics['precision'],
            'recall': best_xgb_metrics['recall'],
        },
        'test_metrics': {
            'precision': round(float(test_p), 4),
            'recall': round(float(test_r), 4),
            'f1': round(float(test_f1), 4),
            'roc_auc': round(float(test_roc), 4),
            'pr_auc': round(float(test_pr), 4),
            'fpr': round(float(test_fpr), 4),
            'fnr': round(float(test_fnr), 4),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        },
        'train_samples': len(txn_train),
        'val_samples': len(txn_val),
        'class_distribution': {
            'train_fraud': int(fraud_train),
            'train_legit': int(len(txn_train) - fraud_train),
            'val_fraud': int(fraud_val),
            'val_legit': int(len(txn_val) - fraud_val),
        },
        'scale_pos_weight': best_xgb_params['scale_pos_weight'],
        'created_at': datetime.utcnow().isoformat(),
        'optimization': {
            'method': 'randomized_grid_search',
            'n_candidates': n_candidates,
            'best_params': best_xgb_params,
        },
        'old_vs_new': {
            'old_test_pr_auc': old_metrics['pr_auc'],
            'new_test_pr_auc': round(float(test_pr), 4),
            'old_test_f1': old_metrics['f1'],
            'new_test_f1': round(float(test_f1), 4),
        },
        'model_comparison': [
            {
                'name': r['name'],
                'roc_auc': r['roc_auc'],
                'pr_auc': r['pr_auc'],
                'best_f1': r['f1'],
                'best_f1_precision': r['precision'],
                'best_f1_recall': r['recall'],
            }
            for r in baseline_results
        ],
        'artifacts': {
            'model': os.path.join(ARTIFACTS_DIR, 'fraud_model.joblib'),
            'preprocessor': os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'),
            'feature_columns': os.path.join(ARTIFACTS_DIR, 'feature_columns.json'),
            'threshold_config': os.path.join(ARTIFACTS_DIR, 'threshold_config.json'),
        },
    }
    with open(os.path.join(ARTIFACTS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    eval_report = {
        'evaluated_at': pd.Timestamp.now().isoformat(),
        'model': 'XGBClassifier',
        'test_samples': len(y_test),
        'test_fraud': fraud_count,
        'test_legit': int(len(y_test) - fraud_count),
        'threshold': best_xgb_thresh,
        'precision': round(float(test_p), 4),
        'recall': round(float(test_r), 4),
        'f1': round(float(test_f1), 4),
        'roc_auc': round(float(test_roc), 4),
        'pr_auc': round(float(test_pr), 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'fpr': round(float(test_fpr), 4),
        'fnr': round(float(test_fnr), 4),
        'cost_analysis': {
            'fn_cost_assumption': 500,
            'fp_cost_assumption': 50,
            'total_fn_cost': int(fn * 500),
            'total_fp_cost': int(fp * 50),
            'total_cost': int(fn * 500 + fp * 50),
        },
        'threshold_sweep': threshold_sweep,
    }
    with open(os.path.join(ARTIFACTS_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(eval_report, f, indent=2)

    print(f"  Saved: fraud_model.joblib")
    print(f"  Saved: preprocessor.joblib")
    print(f"  Saved: feature_columns.json ({len(feature_names)} features)")
    print(f"  Saved: threshold_config.json")
    print(f"  Saved: model_metadata.json")
    print(f"  Saved: evaluation_metrics.json")

    print("\n" + "=" * 70)
    print("  OPTIMIZED TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
