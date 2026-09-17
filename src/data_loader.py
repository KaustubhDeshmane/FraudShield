"""Data loading, validation, and profiling module for Credit Card Fraud Detection."""

import os
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

EXPECTED_FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]
TARGET_COLUMN = "Class"


def get_default_dataset_path() -> str:
    """Resolve the default dataset path across standard project directories."""
    candidate_paths = [
        os.path.join("data", "creditcard.csv"),
        "creditcard.csv",
        os.path.join("..", "data", "creditcard.csv"),
        os.path.join("..", "creditcard.csv"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return os.path.abspath(candidate_paths[0])


def validate_dataset_schema(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, str]:
    """Validate that the dataframe contains all expected feature columns and target if required."""
    missing_features = [col for col in EXPECTED_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        return False, f"Missing required feature columns: {missing_features}"

    if require_target and TARGET_COLUMN not in df.columns:
        return False, f"Missing target column: '{TARGET_COLUMN}'"

    return True, "Schema validation passed."


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive statistical profile and health metrics of the dataset."""
    total_records = int(len(df))
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    profile: Dict[str, Any] = {
        "total_records": total_records,
        "total_columns": int(df.shape[1]),
        "total_missing_values": missing_count,
        "duplicate_records": duplicate_count,
        "columns": list(df.columns),
    }

    if TARGET_COLUMN in df.columns:
        # Standardize target to integers
        class_series = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
        class_counts = class_series.value_counts().to_dict()
        legit_count = int(class_counts.get(0, 0))
        fraud_count = int(class_counts.get(1, 0))
        fraud_rate = (fraud_count / total_records * 100) if total_records > 0 else 0.0

        profile.update({
            "legitimate_count": legit_count,
            "fraudulent_count": fraud_count,
            "fraud_rate_pct": round(fraud_rate, 4),
            "class_distribution": {
                "Legitimate (0)": legit_count,
                "Fraudulent (1)": fraud_count,
            },
        })

    # Amount statistics
    if "Amount" in df.columns:
        profile["amount_stats"] = {
            "mean": float(df["Amount"].mean()),
            "std": float(df["Amount"].std()),
            "min": float(df["Amount"].min()),
            "median": float(df["Amount"].median()),
            "max": float(df["Amount"].max()),
        }

    return profile


def add_derived_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive EDA visualization columns like 'Hour' from 'Time' without altering base schema.
    
    Time represents elapsed seconds over a 48-hour period.
    Hour of the day = (Time // 3600) % 24 (integers 0 through 23).
    """
    df_copy = df.copy()
    if "Time" in df_copy.columns:
        df_copy["Hour"] = ((df_copy["Time"] // 3600) % 24).astype(int)
    return df_copy


def load_creditcard_data(filepath: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load credit card dataset, validate schema, and compute summary profile.
    
    Returns:
        Tuple of (DataFrame, Profile Dictionary)
    """
    path = filepath if filepath else get_default_dataset_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Ensure 'creditcard.csv' is placed inside the 'data/' directory."
        )

    df = pd.read_csv(path)
    is_valid, msg = validate_dataset_schema(df, require_target=True)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {msg}")

    # Ensure Class is integer 0 or 1
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    profile = profile_dataset(df)
    return df, profile


if __name__ == "__main__":
    print("Testing data_loader module...")
    df, prof = load_creditcard_data()
    print(f"Loaded successfully! Shape: {df.shape}")
    print(f"Legitimate: {prof['legitimate_count']}, Fraud: {prof['fraudulent_count']} ({prof['fraud_rate_pct']}%)")
    print(f"Duplicates: {prof['duplicate_records']}, Missing: {prof['total_missing_values']}")
