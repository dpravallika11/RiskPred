import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from ml.preprocessing import FraudDataPreprocessor

def run_training_pipeline():
    print("🚀 Starting RiskPred ML Training Pipeline...")
    
    # 1. Instantiate Preprocessor & Load Data
    preprocessor = FraudDataPreprocessor()
    df = preprocessor.load_and_prepare_ieee_data(data_dir="ml/data")

    # 2. Features and Target
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    # 3. Train / Validation / Held-Out Test Split (80% Train, 10% Val, 10% Test)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Save held-out test set for formal evaluation
    os.makedirs("ml/data", exist_ok=True)
    test_df = pd.concat([X_test_raw, y_test], axis=1)
    test_df.to_csv("ml/data/held_out_test_set.csv", index=False)
    print("📁 Held-out test set saved to ml/data/held_out_test_set.csv")

    # 4. Feature Preprocessing & Scaling
    X_train_scaled = preprocessor.fit_transform(X_train_raw)
    X_val_scaled = preprocessor.transform(X_val_raw)

    # Save Fitted Scaler
    scaler_path = "ml/artifacts/scaler.joblib"
    preprocessor.save_scaler(scaler_path)
    print(f"✅ Scaler artifact saved to {scaler_path}")

    # 5. Handle Class Imbalance using scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"⚖️ Class Imbalance: {neg_count} Non-Fraud / {pos_count} Fraud (scale_pos_weight = {scale_pos_weight:.2f})")

    # 6. Model Training (XGBoost Classifier)
    print("🧠 Training XGBoost Fraud Detection Model...")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_scaled, 
        y_train, 
        eval_set=[(X_val_scaled, y_val)], 
        verbose=False
    )

    # 7. Save Model Artifact
    model_path = "ml/artifacts/fraud_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model artifact successfully saved to {model_path}")
    print("🎉 Training Phase Complete!")

if __name__ == "__main__":
    run_training_pipeline()