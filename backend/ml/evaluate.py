import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    average_precision_score, confusion_matrix, classification_report
)
from ml.preprocessing import FraudDataPreprocessor

def evaluate_held_out_test_set():
    print("📊 Evaluating Model on Held-Out Test Set...")
    
    # 1. Load Held-out Test Data
    test_path = "ml/data/held_out_test_set.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError("Held-out test set not found. Run 'python -m ml.train' first.")
        
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df.drop(columns=['isFraud'])
    y_test = test_df['isFraud']

    # 2. Load Model and Scaler Artifacts
    model = joblib.load("ml/artifacts/fraud_model.joblib")
    preprocessor = FraudDataPreprocessor()
    preprocessor.load_scaler("ml/artifacts/scaler.joblib")

    # 3. Transform Test Data
    X_test_scaled = preprocessor.transform(X_test_raw)

    # 4. Generate Predictions & Probabilities
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Configurable Fraud Threshold
    threshold = 0.50
    y_pred = (y_prob >= threshold).astype(int)

    # 5. Compute Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # 6. Display Performance
    print("\n=================== HELD-OUT TEST EVALUATION METRICS ===================")
    print(f" Precision           : {precision:.4f}")
    print(f" Recall              : {recall:.4f}")
    print(f" F1 Score            : {f1:.4f}")
    print(f" ROC-AUC Score       : {roc_auc:.4f}")
    print(f" PR-AUC Score        : {pr_auc:.4f}")
    print("\n Confusion Matrix    :")
    print(f"   TN: {cm[0][0]}  |  FP: {cm[0][1]}")
    print(f"   FN: {cm[1][0]}  |  TP: {cm[1][1]}")
    print("========================================================================\n")
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    evaluate_held_out_test_set()