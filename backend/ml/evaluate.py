"""
IEEE-CIS Fraud Detection — Evaluation Pipeline

Evaluates the trained model on the completely held-out test set.
The test set must NOT have been used for feature selection, hyperparameter
tuning, threshold selection, or model selection.

Usage:
    cd backend
    python -m ml.evaluate
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

warnings.filterwarnings('ignore')

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def main():
    print("=" * 70)
    print("  IEEE-CIS FRAUD DETECTION — HELD-OUT TEST EVALUATION")
    print("=" * 70)

    print("\nLoading artifacts...")
    model = joblib.load(os.path.join(ARTIFACTS_DIR, 'fraud_model.joblib'))
    preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'))

    with open(os.path.join(ARTIFACTS_DIR, 'threshold_config.json')) as f:
        threshold_config = json.load(f)

    with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.json')) as f:
        feature_columns = json.load(f)

    threshold = threshold_config['threshold']
    print(f"  Model: {type(model).__name__}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Threshold: {threshold:.3f}")

    print("\nLoading held-out test set...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'held_out_test_set.csv'))
    print(f"  Test set: {test_df.shape[0]:,} rows, {test_df.shape[1]} cols")

    y_test = test_df['isFraud'].values
    fraud_count = int(y_test.sum())
    print(f"  Fraud: {fraud_count:,} ({fraud_count/len(y_test)*100:.2f}%)")
    print(f"  Legit: {len(y_test) - fraud_count:,}")

    print("\nPreprocessing test data...")
    if 'TransactionID' in test_df.columns:
        test_for_preprocess = test_df.drop(columns=['isFraud'], errors='ignore')
    else:
        test_for_preprocess = test_df.copy()

    X_test = preprocessor.transform(test_for_preprocess)

    available_features = [c for c in feature_columns if c in X_test.columns]
    missing_features = [c for c in feature_columns if c not in X_test.columns]

    if missing_features:
        print(f"  Warning: {len(missing_features)} features missing, filling with 0")
        for col in missing_features:
            X_test[col] = 0

    X_test = X_test[feature_columns]

    print("\nGenerating predictions...")
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print("\n" + "=" * 70)
    print("  FINAL HELD-OUT TEST RESULTS")
    print("=" * 70)
    print(f"\n  Model:              {type(model).__name__}")
    print(f"  Test samples:       {len(y_test):,}")
    print(f"  Fraud samples:      {fraud_count:,}")
    print(f"  Threshold:          {threshold:.3f}")
    print(f"\n  {'Metric':<25} {'Value':>10}")
    print("  " + "-" * 35)
    print(f"  {'Precision':<25} {precision:>10.4f}")
    print(f"  {'Recall':<25} {recall:>10.4f}")
    print(f"  {'F1 Score':<25} {f1:>10.4f}")
    print(f"  {'ROC-AUC':<25} {roc_auc:>10.4f}")
    print(f"  {'PR-AUC':<25} {pr_auc:>10.4f}")
    print(f"\n  False Positive Rate: {fpr:.4f} ({fp:,} transactions)")
    print(f"  False Negative Rate: {fnr:.4f} ({fn:,} missed fraud)")

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Legit    Fraud")
    print(f"  Actual Legit    [{tn:>6}   {fp:>5}]")
    print(f"  Actual Fraud    [{fn:>6}   {tp:>5}]")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'], digits=4))

    print("\n  Threshold Sweep (on test set for reference):")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}")
    print("  " + "-" * 50)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, threshold]:
        preds = (y_proba >= t).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        tn_, fp_, fn_, tp_ = cm.ravel()
        fpr_ = fp_ / (fp_ + tn_) if (fp_ + tn_) > 0 else 0
        marker = " <-- chosen" if abs(t - threshold) < 0.001 else ""
        print(f"  {t:>10.2f} {p:>10.4f} {r:>10.4f} {f:>10.4f} {fpr_:>10.4f}{marker}")

    print("\n  Cost Analysis (assumptions: FN=$500, FP=$50):")
    fn_cost = threshold_config.get('fn_cost_assumption', 500)
    fp_cost = threshold_config.get('fp_cost_assumption', 50)
    total_cost = fn * fn_cost + fp * fp_cost
    print(f"  False Negative cost: {fn:,} × ${fn_cost} = ${fn * fn_cost:,}")
    print(f"  False Positive cost: {fp:,} × ${fp_cost} = ${fp * fp_cost:,}")
    print(f"  Total estimated cost: ${total_cost:,}")

    report = {
        'evaluated_at': pd.Timestamp.now().isoformat(),
        'model': type(model).__name__,
        'test_samples': len(y_test),
        'test_fraud': fraud_count,
        'test_legit': int(len(y_test) - fraud_count),
        'threshold': threshold,
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1': round(float(f1), 4),
        'roc_auc': round(float(roc_auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'fpr': round(float(fpr), 4),
        'fnr': round(float(fnr), 4),
        'cost_analysis': {
            'fn_cost_assumption': fn_cost,
            'fp_cost_assumption': fp_cost,
            'total_fn_cost': int(fn * fn_cost),
            'total_fp_cost': int(fp * fp_cost),
            'total_cost': int(total_cost),
        },
        'threshold_sweep': [
            {
                'threshold': round(float(t), 4),
                'precision': round(float(precision_score(y_test, (y_proba >= t).astype(int), zero_division=0)), 4),
                'recall': round(float(recall_score(y_test, (y_proba >= t).astype(int), zero_division=0)), 4),
                'f1': round(float(f1_score(y_test, (y_proba >= t).astype(int), zero_division=0)), 4),
            }
            for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ],
    }

    report_path = os.path.join(ARTIFACTS_DIR, 'evaluation_metrics.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved evaluation report: {report_path}")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
