"""Quick ML optimization — runs all phases in sequence."""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml.preprocessing import FraudPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')


def cost_threshold(y_true, y_prob, fn_cost=500, fp_cost=50):
    best_cost, best_t = float('inf'), 0.5
    for t in np.arange(0.05, 0.95, 0.005):
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        c = fn * fn_cost + fp * fp_cost
        if c < best_cost:
            best_cost, best_t = c, float(t)
    return best_t, best_cost


def eval_model(model, X, y, t):
    proba = model.predict_proba(X)[:, 1]
    roc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)
    yp = (proba >= t).astype(int)
    p = precision_score(y, yp, zero_division=0)
    r = recall_score(y, yp, zero_division=0)
    f1 = f1_score(y, yp, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {
        'roc_auc': roc, 'pr_auc': pr, 'precision': p, 'recall': r,
        'f1': f1, 'fpr': fpr, 'fnr': fnr,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def main():
    print("=" * 70)
    print("  OPTIMIZED ML TRAINING")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    txn = pd.read_csv(os.path.join(DATA_DIR, 'train_transaction.csv'))
    identity = pd.read_csv(os.path.join(DATA_DIR, 'train_identity.csv'))
    print(f"  Transactions: {txn.shape}")

    # Temporal split
    txn_sorted = txn.sort_values('TransactionDT').reset_index(drop=True)
    split_idx = int(len(txn_sorted) * 0.85)
    txn_train = txn_sorted.iloc[:split_idx].copy()
    txn_val = txn_sorted.iloc[split_idx:].copy()
    print(f"  Train: {len(txn_train):,}, Val: {len(txn_val):,}")

    train_ids = set(txn_train['TransactionID'].values)
    val_ids = set(txn_val['TransactionID'].values)
    identity_train = identity[identity['TransactionID'].isin(train_ids)]
    identity_val = identity[identity['TransactionID'].isin(val_ids)]

    # Fit preprocessor
    print("\nFitting preprocessor...")
    t0 = time.time()
    preprocessor = FraudPreprocessor()
    preprocessor.fit(txn_train, identity_train)
    print(f"  Fitted in {time.time() - t0:.1f}s, features: {len(preprocessor.feature_columns)}")

    print("Transforming...")
    t0 = time.time()
    X_train = preprocessor.transform(txn_train, identity_train)
    X_val = preprocessor.transform(txn_val, identity_val)
    print(f"  Done in {time.time() - t0:.1f}s")

    y_train = txn_train['isFraud'].values
    y_val = txn_val['isFraud'].values
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    feature_names = preprocessor.get_feature_names()
    print(f"  Class ratio: {ratio:.2f}")

    # ============================================================
    # Baseline models
    # ============================================================
    print("\n" + "=" * 70)
    print("  BASELINE MODELS")
    print("=" * 70)

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1, random_state=42)
    t0 = time.time()
    lr.fit(X_train, y_train)
    lr_time = time.time() - t0
    lr_thresh, _ = cost_threshold(y_val, lr.predict_proba(X_val)[:, 1])
    lr_m = eval_model(lr, X_val, y_val, lr_thresh)
    print(f"  LR: P={lr_m['precision']:.4f} R={lr_m['recall']:.4f} F1={lr_m['f1']:.4f} "
          f"PR-AUC={lr_m['pr_auc']:.4f} ROC={lr_m['roc_auc']:.4f} ({lr_time:.1f}s)")

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced',
                                random_state=42, n_jobs=-1, min_samples_leaf=20)
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_thresh, _ = cost_threshold(y_val, rf.predict_proba(X_val)[:, 1])
    rf_m = eval_model(rf, X_val, y_val, rf_thresh)
    print(f"  RF: P={rf_m['precision']:.4f} R={rf_m['recall']:.4f} F1={rf_m['f1']:.4f} "
          f"PR-AUC={rf_m['pr_auc']:.4f} ROC={rf_m['roc_auc']:.4f} ({rf_time:.1f}s)")

    xgb_base = XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        scale_pos_weight=ratio * 0.5, subsample=0.85,
        colsample_bytree=0.8, min_child_weight=3,
        eval_metric='aucpr', tree_method='hist',
        random_state=42, n_jobs=-1, reg_alpha=0.1, reg_lambda=1.0
    )
    t0 = time.time()
    xgb_base.fit(X_train, y_train)
    xgb_base_time = time.time() - t0
    xgb_base_thresh, _ = cost_threshold(y_val, xgb_base.predict_proba(X_val)[:, 1])
    xgb_base_m = eval_model(xgb_base, X_val, y_val, xgb_base_thresh)
    print(f"  XGB(base): P={xgb_base_m['precision']:.4f} R={xgb_base_m['recall']:.4f} "
          f"F1={xgb_base_m['f1']:.4f} PR-AUC={xgb_base_m['pr_auc']:.4f} ROC={xgb_base_m['roc_auc']:.4f} "
          f"thresh={xgb_base_thresh:.3f} ({xgb_base_time:.1f}s)")

    # ============================================================
    # XGBoost Hyperparameter Search (15 candidates)
    # ============================================================
    print("\n" + "=" * 70)
    print("  XGBOOST HYPERPARAMETER SEARCH (15 candidates)")
    print("=" * 70)

    np.random.seed(42)
    candidates = []
    for _ in range(15):
        candidates.append({
            'n_estimators': np.random.choice([300, 500]),
            'max_depth': np.random.choice([5, 7]),
            'learning_rate': np.random.choice([0.03, 0.05, 0.08]),
            'min_child_weight': np.random.choice([2, 3, 5]),
            'subsample': np.random.choice([0.8, 0.85]),
            'colsample_bytree': np.random.choice([0.7, 0.8]),
            'reg_alpha': np.random.choice([0.05, 0.1]),
            'reg_lambda': np.random.choice([0.5, 1.0]),
            'scale_pos_weight': np.random.choice([ratio * 0.5, ratio * 0.75, ratio]),
        })

    best_pr = -1
    best_params = None
    best_model = None
    best_thresh = 0.5
    best_metrics = None

    for i, p in enumerate(candidates):
        model = XGBClassifier(
            n_estimators=int(p['n_estimators']), max_depth=int(p['max_depth']),
            learning_rate=p['learning_rate'], min_child_weight=int(p['min_child_weight']),
            subsample=p['subsample'], colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
            scale_pos_weight=p['scale_pos_weight'],
            eval_metric='aucpr', tree_method='hist',
            random_state=42, n_jobs=-1,
        )
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        proba = model.predict_proba(X_val)[:, 1]
        pr = average_precision_score(y_val, proba)
        ct, _ = cost_threshold(y_val, proba)
        m = eval_model(model, X_val, y_val, ct)
        tag = ""
        if pr > best_pr:
            best_pr = pr
            best_params = p
            best_model = model
            best_thresh = ct
            best_metrics = m
            tag = " <-- BEST"
        print(f"  [{i+1:2d}/15] PR-AUC={pr:.4f} F1={m['f1']:.4f} depth={int(p['max_depth'])} "
              f"lr={p['learning_rate']} n={int(p['n_estimators'])} spw={p['scale_pos_weight']:.1f} "
              f"({elapsed:.1f}s){tag}")

    print(f"\n  Best XGB: PR-AUC={best_pr:.4f}")
    clean_params = {}
    for k, v in best_params.items():
        if k in ['n_estimators', 'max_depth', 'min_child_weight']:
            clean_params[k] = int(v)
        elif isinstance(v, float):
            clean_params[k] = round(v, 4)
        else:
            clean_params[k] = v
    print(f"  Params: {json.dumps(clean_params, indent=4)}")
    print(f"  Val: P={best_metrics['precision']:.4f} R={best_metrics['recall']:.4f} F1={best_metrics['f1']:.4f}")

    # ============================================================
    # Threshold sweep
    # ============================================================
    print("\n" + "=" * 70)
    print("  THRESHOLD SWEEP (Validation)")
    print("=" * 70)

    val_proba = best_model.predict_proba(X_val)[:, 1]
    sweep = []
    for t in np.arange(0.10, 0.91, 0.02):
        yp = (val_proba >= t).astype(int)
        p = precision_score(y_val, yp, zero_division=0)
        r = recall_score(y_val, yp, zero_division=0)
        f1 = f1_score(y_val, yp, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_val, yp).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        sweep.append({'threshold': t, 'precision': p, 'recall': r, 'f1': f1, 'fpr': fpr, 'fnr': fnr})

    print(f"\n  {'Thr':>6} {'Prec':>8} {'Rec':>8} {'F1':>8} {'FPR':>8} {'FNR':>8}")
    print("  " + "-" * 46)
    for row in sweep:
        marker = " *" if abs(row['threshold'] - best_thresh) < 0.011 else ""
        print(f"  {row['threshold']:>6.2f} {row['precision']:>8.4f} {row['recall']:>8.4f} "
              f"{row['f1']:>8.4f} {row['fpr']:>8.4f} {row['fnr']:>8.4f}{marker}")

    best_f1_row = max(sweep, key=lambda x: x['f1'])
    best_prec_row = max([r for r in sweep if r['recall'] >= 0.50], key=lambda x: x['precision'], default=sweep[-1])
    best_rec_row = max(sweep, key=lambda x: x['recall'])

    print(f"\n  High-Recall: t={best_rec_row['threshold']:.2f} R={best_rec_row['recall']:.4f} P={best_rec_row['precision']:.4f}")
    print(f"  Balanced(F1): t={best_f1_row['threshold']:.2f} R={best_f1_row['recall']:.4f} P={best_f1_row['precision']:.4f} F1={best_f1_row['f1']:.4f}")
    print(f"  High-Precision: t={best_prec_row['threshold']:.2f} R={best_prec_row['recall']:.4f} P={best_prec_row['precision']:.4f}")

    # ============================================================
    # Held-out test evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("  HELD-OUT TEST EVALUATION")
    print("=" * 70)

    test_df = pd.read_csv(os.path.join(DATA_DIR, 'held_out_test_set.csv'))
    y_test = test_df['isFraud'].values
    fraud_count = int(y_test.sum())
    print(f"  Test: {len(y_test):,} rows, {fraud_count:,} fraud ({fraud_count / len(y_test) * 100:.2f}%)")

    test_pre = test_df.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
    X_test = preprocessor.transform(test_pre)
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_names]

    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_roc = roc_auc_score(y_test, test_proba)
    test_pr = average_precision_score(y_test, test_proba)

    # Cost-optimized threshold
    y_pred = (test_proba >= best_thresh).astype(int)
    test_p = precision_score(y_test, y_pred, zero_division=0)
    test_r = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    test_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  OPTIMIZED XGBOOST — TEST (threshold={best_thresh:.3f})")
    print(f"  Precision:  {test_p:.4f}")
    print(f"  Recall:     {test_r:.4f}")
    print(f"  F1:         {test_f1:.4f}")
    print(f"  PR-AUC:     {test_pr:.4f}")
    print(f"  ROC-AUC:    {test_roc:.4f}")
    print(f"  FPR:        {test_fpr:.4f} ({fp:,})")
    print(f"  FNR:        {test_fnr:.4f} ({fn:,})")
    print(f"  CM: TN={tn} FP={fp} FN={fn} TP={tp}")

    # Baseline XGBoost on test
    base_proba = xgb_base.predict_proba(X_test)[:, 1]
    base_yp = (base_proba >= xgb_base_thresh).astype(int)
    base_test_p = precision_score(y_test, base_yp, zero_division=0)
    base_test_r = recall_score(y_test, base_yp, zero_division=0)
    base_test_f1 = f1_score(y_test, base_yp, zero_division=0)
    base_test_pr = average_precision_score(y_test, base_proba)
    base_test_roc = roc_auc_score(y_test, base_proba)
    bt_cm = confusion_matrix(y_test, base_yp)
    bt_tn, bt_fp, bt_fn, bt_tp = bt_cm.ravel()
    base_test_fpr = bt_fp / (bt_fp + bt_tn) if (bt_fp + bt_tn) > 0 else 0
    base_test_fnr = bt_fn / (bt_fn + bt_tp) if (bt_fn + bt_tp) > 0 else 0

    # Also test at F1-optimal threshold
    f1_yp = (test_proba >= best_f1_row['threshold']).astype(int)
    f1_test_p = precision_score(y_test, f1_yp, zero_division=0)
    f1_test_r = recall_score(y_test, f1_yp, zero_division=0)
    f1_test_f1 = f1_score(y_test, f1_yp, zero_division=0)

    # ============================================================
    # OLD vs NEW comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("  OLD vs NEW COMPARISON")
    print("=" * 70)

    with open(os.path.join(ARTIFACTS_DIR, 'evaluation_metrics.json')) as f:
        old_eval = json.load(f)

    print(f"\n  {'Metric':<15} {'OLD':>12} {'NEW(opt)':>12} {'Delta':>12}")
    print("  " + "-" * 51)
    old_vals = [
        ('precision', old_eval['precision'], test_p),
        ('recall', old_eval['recall'], test_r),
        ('f1', old_eval['f1'], test_f1),
        ('pr_auc', old_eval['pr_auc'], test_pr),
        ('roc_auc', old_eval['roc_auc'], test_roc),
        ('fpr', old_eval['fpr'], test_fpr),
        ('fnr', old_eval['fnr'], test_fnr),
    ]
    for key, old_v, new_v in old_vals:
        delta = new_v - old_v
        pct = (delta / old_v * 100) if old_v != 0 else 0
        sign = "+" if delta > 0 else ""
        print(f"  {key:<15} {old_v:>12.4f} {new_v:>12.4f} {sign}{delta:>10.4f} ({pct:>+.1f}%)")

    print(f"\n  OLD threshold: {old_eval['threshold']:.3f}")
    print(f"  NEW threshold: {best_thresh:.3f}")

    # ============================================================
    # Save artifacts
    # ============================================================
    print("\n" + "=" * 70)
    print("  SAVING ARTIFACTS")
    print("=" * 70)

    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, 'fraud_model.joblib'))
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'))

    with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)

    threshold_config = {
        'threshold': best_thresh,
        'method': 'cost_sensitive',
        'fn_cost_assumption': 500,
        'fp_cost_assumption': 50,
        'rationale': (
            'Threshold optimized to minimize total expected cost on validation set. '
            'FN cost ($500) = avg fraud loss. FP cost ($50) = customer friction.'
        ),
        'best_f1_threshold': best_f1_row['threshold'],
        'validation_roc_auc': best_metrics['roc_auc'],
        'validation_pr_auc': best_metrics['pr_auc'],
    }
    with open(os.path.join(ARTIFACTS_DIR, 'threshold_config.json'), 'w') as f:
        json.dump(threshold_config, f, indent=2)

    metadata = {
        'model': 'XGBClassifier',
        'model_params': best_model.get_params(),
        'features': feature_names,
        'feature_count': len(feature_names),
        'threshold': best_thresh,
        'threshold_method': 'cost_sensitive_optimization',
        'validation_metrics': {
            'roc_auc': best_metrics['roc_auc'],
            'pr_auc': best_metrics['pr_auc'],
            'f1': best_metrics['f1'],
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
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
            'train_fraud': int(sum(y_train)),
            'train_legit': int(len(y_train) - sum(y_train)),
            'val_fraud': int(sum(y_val)),
            'val_legit': int(len(y_val) - sum(y_val)),
        },
        'scale_pos_weight': best_params['scale_pos_weight'],
        'created_at': pd.Timestamp.now().isoformat(),
        'optimization': {
            'method': 'randomized_search_15_candidates',
            'best_params': {k: round(float(v), 4) if isinstance(v, float) else int(v)
                           for k, v in best_params.items()},
        },
        'model_comparison': [
            {'name': 'Logistic Regression', 'roc_auc': lr_m['roc_auc'], 'pr_auc': lr_m['pr_auc'],
             'best_f1': lr_m['f1'], 'best_f1_precision': lr_m['precision'], 'best_f1_recall': lr_m['recall']},
            {'name': 'Random Forest', 'roc_auc': rf_m['roc_auc'], 'pr_auc': rf_m['pr_auc'],
             'best_f1': rf_m['f1'], 'best_f1_precision': rf_m['precision'], 'best_f1_recall': rf_m['recall']},
            {'name': 'XGBoost (Baseline)', 'roc_auc': xgb_base_m['roc_auc'], 'pr_auc': xgb_base_m['pr_auc'],
             'best_f1': xgb_base_m['f1'], 'best_f1_precision': xgb_base_m['precision'], 'best_f1_recall': xgb_base_m['recall']},
            {'name': 'XGBoost (Optimized)', 'roc_auc': best_metrics['roc_auc'], 'pr_auc': best_metrics['pr_auc'],
             'best_f1': best_metrics['f1'], 'best_f1_precision': best_metrics['precision'], 'best_f1_recall': best_metrics['recall']},
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
        'threshold': best_thresh,
        'precision': round(float(test_p), 4),
        'recall': round(float(test_r), 4),
        'f1': round(float(test_f1), 4),
        'roc_auc': round(float(test_roc), 4),
        'pr_auc': round(float(test_pr), 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'fpr': round(float(test_fpr), 4),
        'fnr': round(float(test_fnr), 4),
        'cost_analysis': {
            'fn_cost_assumption': 500, 'fp_cost_assumption': 50,
            'total_fn_cost': int(fn * 500), 'total_fp_cost': int(fp * 50),
            'total_cost': int(fn * 500 + fp * 50),
        },
        'threshold_sweep': [
            {'threshold': round(float(row['threshold']), 4),
             'precision': round(float(row['precision']), 4),
             'recall': round(float(row['recall']), 4),
             'f1': round(float(row['f1']), 4)}
            for row in sweep
        ],
    }
    with open(os.path.join(ARTIFACTS_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(eval_report, f, indent=2)

    # Cleanup temp baseline file
    temp_path = os.path.join(ARTIFACTS_DIR, 'xgb_baseline.joblib')
    if os.path.exists(temp_path):
        os.remove(temp_path)

    print("  All artifacts saved.")
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
