"""Preprocessing pipeline module for Credit Card Fraud Detection.

Handles leakage-free feature scaling of Time and Amount, preserves column ordering,
and guarantees reproducibility across training, single inference, and batch predictions.
"""

import os
from typing import Dict, List, Union, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.data_loader import EXPECTED_FEATURE_COLUMNS


class CreditCardPreprocessor:
    """Preprocessor for credit card transactions.
    
    Fits StandardScaler on 'Time' and 'Amount' strictly on training data
    to prevent data leakage, and enforces deterministic feature alignment.
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns if feature_columns else list(EXPECTED_FEATURE_COLUMNS)
        self.scale_columns = ["Time", "Amount"]
        self.time_scaler = StandardScaler()
        self.amount_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "CreditCardPreprocessor":
        """Fit scalers strictly on training data."""
        self._validate_input_columns(X)
        self.time_scaler.fit(X[["Time"]])
        self.amount_scaler.fit(X[["Amount"]])
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Transform input features using fitted scalers and enforce feature order."""
        if not self.is_fitted:
            raise RuntimeError("CreditCardPreprocessor must be fitted before calling transform().")

        df = self._to_ordered_dataframe(X)
        df_transformed = df.copy()

        df_transformed["Time"] = self.time_scaler.transform(df[["Time"]])
        df_transformed["Amount"] = self.amount_scaler.transform(df[["Amount"]])

        return df_transformed[self.feature_columns]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on X and return transformed DataFrame."""
        return self.fit(X).transform(X)

    def _validate_input_columns(self, X: pd.DataFrame) -> None:
        """Ensure all required features exist in the input DataFrame."""
        missing = [col for col in self.feature_columns if col not in X.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")

    def _to_ordered_dataframe(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """Convert dictionary, list of dicts, or DataFrame into an aligned DataFrame."""
        if isinstance(data, dict):
            # Check for missing keys
            missing = [col for col in self.feature_columns if col not in data]
            if missing:
                raise ValueError(f"Transaction data missing required features: {missing}")
            # Ensure float conversion
            df = pd.DataFrame([{col: float(data[col]) for col in self.feature_columns}])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            self._validate_input_columns(df)
            df = df[self.feature_columns].astype(float)
        elif isinstance(data, pd.DataFrame):
            self._validate_input_columns(data)
            df = data[self.feature_columns].astype(float)
        else:
            raise TypeError(f"Unsupported data type for preprocessing: {type(data)}")

        return df

    def save(self, filepath: str) -> None:
        """Serialize preprocessor to disk using joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CreditCardPreprocessor":
        """Load fitted preprocessor from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor artifact not found at: '{filepath}'")
        instance = joblib.load(filepath)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return instance


if __name__ == "__main__":
    print("Testing CreditCardPreprocessor...")
    from src.data_loader import load_creditcard_data

    raw_df, _ = load_creditcard_data()
    X = raw_df[EXPECTED_FEATURE_COLUMNS].head(100)

    preproc = CreditCardPreprocessor()
    transformed = preproc.fit_transform(X)

    print("Transformed shape:", transformed.shape)
    print("Time mean:", transformed["Time"].mean(), "std:", transformed["Time"].std())
    print("Amount mean:", transformed["Amount"].mean(), "std:", transformed["Amount"].std())
    print("Preproc test passed successfully.")
