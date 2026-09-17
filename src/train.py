"""Model training, evaluation, and artifact serialization pipeline.

Executes the leakage-free training process:
1. Loads dataset and validates schema
2. Performs stratified 80:20 train/test split (random_state=42)
3. Fits preprocessor strictly on X_train
4. Trains Logistic Regression baseline and Random Forest primary model
5. Computes actual test evaluation metrics (no hardcoded metrics)
6. Serializes model artifacts to models/ and metrics to artifacts/
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.data_loader import load_creditcard_data, EXPECTED_FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import CreditCardPreprocessor
from src.evaluate import evaluate_model


def train_models(
    dataset_path: str = None,
    models_dir: str = "models",
    artifacts_dir: str = "artifacts",
    random_state: int = 42,
    test_size: float = 0.20,
) -> Dict[str, Any]:
    """Execute end-to-end model training, evaluation, and artifact saving."""
    start_time = time.time()
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print("=" * 60)
    print(" CREDIT CARD FRAUD DETECTION — MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/6] Loading and validating dataset...")
    df, profile = load_creditcard_data(dataset_path)
    print(f"      Total records: {profile['total_records']:,}")
    print(f"      Legitimate:    {profile['legitimate_count']:,} ({100 - profile['fraud_rate_pct']:.2f}%)")
    print(f"      Fraudulent:    {profile['fraudulent_count']:,} ({profile['fraud_rate_pct']:.4f}%)")

    # 2. Stratified Train/Test Split
    print(f"\n[2/6] Performing stratified split ({int((1 - test_size) * 100)}:{int(test_size * 100)})...")
    X = df[EXPECTED_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"      Training set:   {len(X_train):,} samples ({(y_train == 1).sum()} fraud)")
    print(f"      Test set:       {len(X_test):,} samples ({(y_test == 1).sum()} fraud)")

    # 3. Fit Preprocessor (Prevent Data Leakage)
    print("\n[3/6] Fitting StandardScaler on training set (leakage-free)...")
    preprocessor = CreditCardPreprocessor(feature_columns=EXPECTED_FEATURE_COLUMNS)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    print("      Preprocessing completed successfully.")

    # 4. Train Models
    # Model 1: Logistic Regression (Baseline)
    print("\n[4/6] Training Logistic Regression (Baseline)...")
    t0 = time.time()
    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_model.fit(X_train_scaled, y_train)
    print(f"      Logistic Regression trained in {time.time() - t0:.2f}s")

    # Model 2: Random Forest (Primary)
    print("\n[5/6] Training Random Forest Classifier (Primary Model)...")
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    rf_model.fit(X_train_scaled, y_train)
    print(f"      Random Forest trained in {time.time() - t0:.2f}s")

    # 5. Evaluate Models
    print("\n[6/6] Computing evaluation metrics on unseen test set...")
    lr_metrics = evaluate_model(
        lr_model,
        X_test_scaled,
        y_test,
        feature_names=EXPECTED_FEATURE_COLUMNS,
        threshold=0.50,
    )
    rf_metrics = evaluate_model(
        rf_model,
        X_test_scaled,
        y_test,
        feature_names=EXPECTED_FEATURE_COLUMNS,
        threshold=0.50,
    )

    # Print summary table
    print("\n" + "-" * 65)
    print(f"{'Metric':<25} {'Logistic Regression':<20} {'Random Forest':<20}")
    print("-" * 65)
    print(f"{'Precision (Fraud)':<25} {lr_metrics['precision']:<20.4f} {rf_metrics['precision']:<20.4f}")
    print(f"{'Recall (Fraud)':<25} {lr_metrics['recall']:<20.4f} {rf_metrics['recall']:<20.4f}")
    print(f"{'F1-Score (Fraud)':<25} {lr_metrics['f1']:<20.4f} {rf_metrics['f1']:<20.4f}")
    print(f"{'ROC-AUC':<25} {lr_metrics['roc_auc']:<20.4f} {rf_metrics['roc_auc']:<20.4f}")
    print(f"{'PR-AUC (Avg Precision)':<25} {lr_metrics['pr_auc']:<20.4f} {rf_metrics['pr_auc']:<20.4f}")
    print(f"{'Balanced Accuracy':<25} {lr_metrics['balanced_accuracy']:<20.4f} {rf_metrics['balanced_accuracy']:<20.4f}")
    print(f"{'Overall Accuracy':<25} {lr_metrics['accuracy']:<20.4f} {rf_metrics['accuracy']:<20.4f}")
    print("-" * 65)
    print(f"Confusion Matrix (LR): TP={lr_metrics['confusion_matrix']['tp']}, FP={lr_metrics['confusion_matrix']['fp']}, FN={lr_metrics['confusion_matrix']['fn']}, TN={lr_metrics['confusion_matrix']['tn']}")
    print(f"Confusion Matrix (RF): TP={rf_metrics['confusion_matrix']['tp']}, FP={rf_metrics['confusion_matrix']['fp']}, FN={rf_metrics['confusion_matrix']['fn']}, TN={rf_metrics['confusion_matrix']['tn']}")
    print("-" * 65)

    # 6. Save Artifacts
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    fraud_model_path = os.path.join(models_dir, "fraud_model.joblib")
    baseline_model_path = os.path.join(models_dir, "baseline_model.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    samples_path = os.path.join(artifacts_dir, "sample_transactions.json")

    print("\nSaving artifacts...")
    preprocessor.save(preprocessor_path)
    joblib.dump(rf_model, fraud_model_path)
    joblib.dump(lr_model, baseline_model_path)

    metadata: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "primary_model": "RandomForestClassifier",
        "baseline_model": "LogisticRegression",
        "feature_columns": EXPECTED_FEATURE_COLUMNS,
        "split_ratio": f"{int((1 - test_size) * 100)}:{int(test_size * 100)} (Stratified)",
        "random_state": random_state,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "rf_params": {
            "n_estimators": 100,
            "random_state": random_state,
        },
        "lr_params": {
            "max_iter": 1000,
            "random_state": random_state,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    all_metrics: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "random_forest": rf_metrics,
        "logistic_regression": lr_metrics,
        "dataset_summary": profile,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Extract clean sample transactions for UI demo (one real legitimate, one real fraud)
    test_df = X_test.copy()
    test_df["Class"] = y_test

    sample_legit = test_df[test_df["Class"] == 0].iloc[10].drop("Class").to_dict()
    sample_fraud = test_df[test_df["Class"] == 1].iloc[5].drop("Class").to_dict()

    sample_transactions = {
        "legitimate": {k: round(float(v), 4) for k, v in sample_legit.items()},
        "fraudulent": {k: round(float(v), 4) for k, v in sample_fraud.items()},
    }

    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(sample_transactions, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nTraining pipeline completed in {elapsed:.2f} seconds.")
    print(f"Artifacts saved:")
    print(f"  - {preprocessor_path}")
    print(f"  - {fraud_model_path}")
    print(f"  - {baseline_model_path}")
    print(f"  - {metadata_path}")
    print(f"  - {metrics_path}")
    print(f"  - {samples_path}")
    print("=" * 60)

    return all_metrics


if __name__ == "__main__":
    train_models()
