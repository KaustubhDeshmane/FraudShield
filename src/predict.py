"""Inference module for Credit Card Fraud Detection.

Provides single-transaction prediction and batch CSV scoring with configurable
decision thresholds and clear probabilistic outcome descriptions.
"""

import os
from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from src.data_loader import EXPECTED_FEATURE_COLUMNS
from src.preprocessing import CreditCardPreprocessor


class FraudPredictor:
    """Predictor class that loads models and preprocessors for inference."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        model_type: str = "rf",
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")

        if preprocessor_path is None:
            preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")

        if model_path is None:
            if model_type.lower() in ("lr", "logistic", "baseline"):
                model_path = os.path.join(models_dir, "baseline_model.joblib")
            else:
                model_path = os.path.join(models_dir, "fraud_model.joblib")

        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load the preprocessor and trained classifier from disk."""
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor artifact not found at '{self.preprocessor_path}'. "
                f"Please run 'python -m src.train' first."
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. "
                f"Please run 'python -m src.train' first."
            )

        self.preprocessor = CreditCardPreprocessor.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)

    def predict_single(
        self,
        transaction_data: Dict[str, Union[float, int]],
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Predict fraud probability and outcome for a single transaction dictionary.
        
        Args:
            transaction_data: Dictionary containing Time, Amount, and V1 through V28.
            threshold: Decision threshold for classification (default 0.50).
            
        Returns:
            Dictionary containing prediction, fraud_probability, threshold, label, and message.
        """
        # Validate and preprocess
        transformed_df = self.preprocessor.transform(transaction_data)

        # Get probability
        probabilities = self.model.predict_proba(transformed_df)[0]
        fraud_prob = float(probabilities[1])
        legit_prob = float(probabilities[0])

        # Apply threshold decision rule
        is_fraud = bool(fraud_prob >= threshold)
        prediction_class = 1 if is_fraud else 0
        prediction_label = "Potential Fraud" if is_fraud else "Legitimate"

        if is_fraud:
            status_message = (
                f"The model estimates an elevated fraud probability ({fraud_prob * 100:.2f}%) "
                f"exceeding the decision threshold ({threshold * 100:.1f}%). "
                f"Transaction flagged for verification."
            )
        else:
            status_message = (
                f"The model estimates a low fraud probability ({fraud_prob * 100:.2f}%) "
                f"below the decision threshold ({threshold * 100:.1f}%). "
                f"Transaction appears legitimate."
            )

        return {
            "prediction": prediction_class,
            "prediction_label": prediction_label,
            "fraud_probability": round(fraud_prob, 5),
            "fraud_probability_pct": round(fraud_prob * 100, 2),
            "legitimate_probability_pct": round(legit_prob * 100, 2),
            "threshold": float(threshold),
            "threshold_pct": round(threshold * 100, 1),
            "is_fraud": is_fraud,
            "status_message": status_message,
            "disclaimer": "This is a probabilistic machine learning estimate, not definitive proof of fraud.",
        }

    def predict_batch(
        self,
        df_batch: pd.DataFrame,
        threshold: float = 0.50,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run batch inference on an uploaded DataFrame.
        
        Args:
            df_batch: DataFrame containing the 30 required transaction features.
            threshold: Decision threshold for classification.
            
        Returns:
            Tuple of (Enriched DataFrame with prediction columns, Summary Statistics Dict)
        """
        # Check missing features
        missing_features = [col for col in EXPECTED_FEATURE_COLUMNS if col not in df_batch.columns]
        if missing_features:
            raise ValueError(f"Uploaded batch CSV is missing required columns: {missing_features}")

        # Check and handle non-numeric values
        numeric_df = df_batch[EXPECTED_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if numeric_df.isnull().any().any():
            nan_cols = numeric_df.columns[numeric_df.isnull().any()].tolist()
            raise ValueError(f"Uploaded CSV contains non-numeric or missing values in columns: {nan_cols}")

        # Preprocess features
        transformed = self.preprocessor.transform(numeric_df)

        # Batch probability inference
        probs = self.model.predict_proba(transformed)[:, 1]
        preds = (probs >= threshold).astype(int)
        labels = ["Potential Fraud" if p == 1 else "Legitimate" for p in preds]

        # Construct result DataFrame preserving original data
        result_df = df_batch.copy()
        result_df["fraud_probability"] = np.round(probs, 5)
        result_df["fraud_probability_pct"] = np.round(probs * 100, 2)
        result_df["prediction"] = preds
        result_df["prediction_label"] = labels

        total_records = len(result_df)
        flagged_fraud = int((preds == 1).sum())
        flagged_legit = total_records - flagged_fraud
        fraud_rate = (flagged_fraud / total_records * 100) if total_records > 0 else 0.0

        summary = {
            "total_processed": total_records,
            "flagged_fraud": flagged_fraud,
            "flagged_legitimate": flagged_legit,
            "fraud_rate_pct": round(fraud_rate, 2),
            "threshold_used": float(threshold),
        }

        return result_df, summary


# Cached predictor instances
_PREDICTOR_CACHE: Dict[str, FraudPredictor] = {}


def get_predictor(model_type: str = "rf") -> FraudPredictor:
    """Get or instantiate a cached FraudPredictor."""
    key = model_type.lower()
    if key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[key] = FraudPredictor(model_type=key)
    return _PREDICTOR_CACHE[key]


if __name__ == "__main__":
    print("predict module loaded successfully.")
