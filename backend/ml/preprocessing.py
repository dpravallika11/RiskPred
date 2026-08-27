import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class FraudDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'amount', 
            'is_new_device', 
            'is_new_location', 
            'velocity_5m', 
            'failed_attempts_24h',
            'amount_velocity_ratio',
            'risk_signal_count'
        ]

    def load_and_prepare_ieee_data(self, data_dir: str = "ml/data") -> pd.DataFrame:
        """Loads and processes IEEE-CIS train_transaction.csv and train_identity.csv."""
        trans_path = os.path.join(data_dir, "train_transaction.csv")
        id_path = os.path.join(data_dir, "train_identity.csv")

        if os.path.exists(trans_path):
            print(" Reading IEEE-CIS transaction data...")
            df_trans = pd.read_csv(trans_path)

            if os.path.exists(id_path):
                print(" Merging identity data on TransactionID...")
                df_id = pd.read_csv(id_path)
                df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
            else:
                df = df_trans

            # Map IEEE-CIS raw features to RiskPred pipeline features
            processed_df = pd.DataFrame({
                'amount': df['TransactionAmt'].fillna(0),
                'is_new_device': df['DeviceType'].notnull().astype(int) if 'DeviceType' in df.columns else np.random.choice([0, 1], size=len(df)),
                'is_new_location': df['addr1'].isnull().astype(int),
                'velocity_5m': df['C1'].fillna(1).clip(upper=20).astype(int),
                'failed_attempts_24h': df['D1'].fillna(0).clip(upper=10).astype(int),
                'isFraud': df['isFraud'].astype(int)
            })

            print(f" Loaded {len(processed_df)} rows from IEEE-CIS dataset (Fraud Count: {processed_df['isFraud'].sum()})")
            return processed_df
        else:
            print(" IEEE-CIS dataset files not found. Generating synthetic fallback dataset...")
            return self.generate_synthetic_dataset()

    def generate_synthetic_dataset(self, n_samples: int = 15000, fraud_ratio: float = 0.05, random_state: int = 42) -> pd.DataFrame:
        np.random.seed(random_state)
        n_fraud = int(n_samples * fraud_ratio)
        n_legit = n_samples - n_fraud

        df = pd.DataFrame({
            'amount': np.concatenate([np.random.exponential(2000, n_legit) + 100, np.random.exponential(15000, n_fraud) + 5000]),
            'is_new_device': np.concatenate([np.random.choice([0, 1], n_legit, p=[0.9, 0.1]), np.random.choice([0, 1], n_fraud, p=[0.25, 0.75])]),
            'is_new_location': np.concatenate([np.random.choice([0, 1], n_legit, p=[0.88, 0.12]), np.random.choice([0, 1], n_fraud, p=[0.3, 0.7])]),
            'velocity_5m': np.concatenate([np.random.poisson(1.2, n_legit) + 1, np.random.poisson(4.5, n_fraud) + 2]),
            'failed_attempts_24h': np.concatenate([np.random.choice([0, 1, 2], n_legit, p=[0.85, 0.12, 0.03]), np.random.choice([1, 2, 3, 4, 5], n_fraud, p=[0.1, 0.2, 0.3, 0.25, 0.15])]),
            'isFraud': np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
        })
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['amount_velocity_ratio'] = df['amount'] / (df['velocity_5m'] + 1)
        df['risk_signal_count'] = (
            df['is_new_device'] + 
            df['is_new_location'] + 
            (df['velocity_5m'] > 3).astype(int) + 
            (df['failed_attempts_24h'] > 2).astype(int)
        )
        return df

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X_eng = self.engineer_features(X)[self.feature_columns]
        return self.scaler.fit_transform(X_eng)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_eng = self.engineer_features(X)[self.feature_columns]
        return self.scaler.transform(X_eng)

    def save_scaler(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)

    def load_scaler(self, filepath: str):
        self.scaler = joblib.load(filepath)