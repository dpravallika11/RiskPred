"""
IEEE-CIS Fraud Detection — Training Pipeline

Trains Logistic Regression, Random Forest, and XGBoost models.
Compares on validation set. Selects best model. Saves artifacts.

Usage:
    cd backend
    python -m ml.train
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix
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


def compute_threshold_metrics(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.01)

    results = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        results.append({
            'threshold': round(float(t), 4),
            'precision': round(float(p), 4),
            'recall': round(float(r), 4),
            'f1': round(float(f1), 4),
            'fpr': round(float(fpr), 4),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        })
    return results


def cost_sensitive_threshold(y_true, y_prob, fn_cost=500, fp_cost=50):
    """
    Find threshold that minimizes expected cost.

    Assumptions (documented):
    - False Negative cost (missed fraud): $500 per incident
    - False Positive cost (blocked legitimate transaction): $50 per incident
    - These are reasonable estimates for mid-size merchant fraud prevention
    """
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


def train_model(name, model, X_train, y_train, X_val, y_val):
    print(f"\nTraining {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    proba = model.predict_proba(X_val)[:, 1]

    roc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)

    threshold_results = compute_threshold_metrics(y_val, proba)
    best_by_f1 = max(threshold_results, key=lambda x: x['f1'])

    cost_threshold, min_cost = cost_sensitive_threshold(y_val, proba)

    print(f"  ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"  Best F1 threshold: {best_by_f1['threshold']:.2f} "
          f"(P={best_by_f1['precision']:.4f}, R={best_by_f1['recall']:.4f}, F1={best_by_f1['f1']:.4f})")
    print(f"  Cost-optimal threshold: {cost_threshold:.3f} (min cost: ${min_cost:,.0f})")
    print(f"  Train time: {train_time:.1f}s")

    return {
        'name': name,
        'model': model,
        'roc_auc': roc,
        'pr_auc': pr_auc,
        'best_f1_threshold': best_by_f1,
        'cost_threshold': cost_threshold,
        'min_cost': min_cost,
        'threshold_results': threshold_results,
        'train_time': train_time,
        'val_proba': proba,
    }


def main():
    print("=" * 70)
    print("  IEEE-CIS FRAUD DETECTION — TRAINING PIPELINE")
    print("=" * 70)

    txn, identity = load_data()

    print("\nPerforming temporal train/val split...")
    txn_train, txn_val = temporal_split(txn, train_ratio=0.85)

    fraud_train = txn_train['isFraud'].sum()
    fraud_val = txn_val['isFraud'].sum()
    print(f"  Train: {len(txn_train):,} samples ({fraud_train:,} fraud, {fraud_train/len(txn_train)*100:.2f}%)")
    print(f"  Val:   {len(txn_val):,} samples ({fraud_val:,} fraud, {fraud_val/len(txn_val)*100:.2f}%)")

    print("\nFitting preprocessor on training data...")
    preprocessor = FraudPreprocessor()

    identity_train = None
    identity_val = None
    if identity is not None:
        train_ids = set(txn_train['TransactionID'].values)
        val_ids = set(txn_val['TransactionID'].values)
        identity_train = identity[identity['TransactionID'].isin(train_ids)]
        identity_val = identity[identity['TransactionID'].isin(val_ids)]
        print(f"  Identity (train): {len(identity_train):,} matched")
        print(f"  Identity (val):   {len(identity_val):,} matched")

    preprocessor.fit(txn_train, identity_train)

    print("  Fitting label encoders and computing fill values...")
    print(f"  Final feature count: {len(preprocessor.feature_columns)}")

    print("\nTransforming training data...")
    X_train = preprocessor.transform(txn_train, identity_train)
    y_train = txn_train['isFraud'].values

    print("Transforming validation data...")
    X_val = preprocessor.transform(txn_val, identity_val)
    y_val = txn_val['isFraud'].values

    feature_names = preprocessor.get_feature_names()

    ratio = (len(y_train) - sum(y_train)) / sum(y_train)

    models_config = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000, class_weight='balanced', C=0.1, random_state=42
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight='balanced',
            random_state=42, n_jobs=-1, min_samples_leaf=20
        )),
        ("XGBoost", XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.05,
            scale_pos_weight=ratio * 0.5, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=3,
            eval_metric='aucpr', tree_method='hist',
            random_state=42, n_jobs=-1, reg_alpha=0.1, reg_lambda=1.0
        )),
    ]

    results = []
    for name, model in models_config:
        result = train_model(name, model, X_train, y_train, X_val, y_val)
        results.append(result)

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON (Validation Set)")
    print("=" * 70)
    print(f"{'Model':<25} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 75)
    for r in results:
        b = r['best_f1_threshold']
        print(f"{r['name']:<25} {r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f} "
              f"{b['f1']:>10.4f} {b['precision']:>10.4f} {b['recall']:>10.4f}")

    best_result = max(results, key=lambda x: x['pr_auc'])
    print(f"\n  BEST MODEL: {best_result['name']} (by PR-AUC)")

    best_model = best_result['model']
    best_threshold = best_result['cost_threshold']

    print(f"\n  Chosen threshold: {best_threshold:.3f} (cost-optimized)")

    preds = (best_result['val_proba'] >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"  False Negative Rate: {fn/(fn+tp):.4f}")

    print("\nSaving artifacts...")
    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, 'fraud_model.joblib'))
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'))

    with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)

    threshold_config = {
        'threshold': best_threshold,
        'method': 'cost_sensitive',
        'fn_cost_assumption': 500,
        'fp_cost_assumption': 50,
        'rationale': (
            'Threshold optimized to minimize total expected cost. '
            'False Negative cost ($500) reflects average fraud loss. '
            'False Positive cost ($50) reflects customer friction and manual review overhead. '
            'Chosen to balance fraud detection with operational cost.'
        ),
        'best_f1_threshold': best_result['best_f1_threshold']['threshold'],
        'validation_roc_auc': best_result['roc_auc'],
        'validation_pr_auc': best_result['pr_auc'],
    }
    with open(os.path.join(ARTIFACTS_DIR, 'threshold_config.json'), 'w') as f:
        json.dump(threshold_config, f, indent=2)

    metadata = {
        'model': type(best_model).__name__,
        'model_params': best_model.get_params(),
        'features': feature_names,
        'feature_count': len(feature_names),
        'threshold': best_threshold,
        'threshold_method': 'cost_sensitive_optimization',
        'validation_metrics': {
            'roc_auc': best_result['roc_auc'],
            'pr_auc': best_result['pr_auc'],
            'f1': best_result['best_f1_threshold']['f1'],
            'precision': best_result['best_f1_threshold']['precision'],
            'recall': best_result['best_f1_threshold']['recall'],
        },
        'train_samples': len(txn_train),
        'val_samples': len(txn_val),
        'class_distribution': {
            'train_fraud': int(fraud_train),
            'train_legit': int(len(txn_train) - fraud_train),
            'val_fraud': int(fraud_val),
            'val_legit': int(len(txn_val) - fraud_val),
        },
        'scale_pos_weight': round(ratio * 0.5, 4),
        'created_at': datetime.utcnow().isoformat(),
        'model_comparison': [
            {
                'name': r['name'],
                'roc_auc': r['roc_auc'],
                'pr_auc': r['pr_auc'],
                'best_f1': r['best_f1_threshold']['f1'],
                'best_f1_precision': r['best_f1_threshold']['precision'],
                'best_f1_recall': r['best_f1_threshold']['recall'],
            }
            for r in results
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

    print(f"\n  Saved: fraud_model.joblib")
    print(f"  Saved: preprocessor.joblib")
    print(f"  Saved: feature_columns.json ({len(feature_names)} features)")
    print(f"  Saved: threshold_config.json")
    print(f"  Saved: model_metadata.json")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
