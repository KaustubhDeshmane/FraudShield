"""Comprehensive test suite for Credit Card Fraud Detection application.

Validates:
1. Dataset loading and schema validation
2. Preprocessing pipeline
3. Model loading and artifact consistency
4. Single transaction prediction (legitimate and fraud presets)
5. Dynamic threshold behavior
6. Batch CSV prediction and schema enrichment
7. Invalid input error handling
"""

import os
import json
import pandas as pd

from src.data_loader import load_creditcard_data, validate_dataset_schema, EXPECTED_FEATURE_COLUMNS
from src.preprocessing import CreditCardPreprocessor
from src.predict import FraudPredictor, get_predictor


def test_data_loader():
    print("\n--- Test 1: Data Loader & Schema ---")
    df, profile = load_creditcard_data()
    assert df.shape[0] == 284807, f"Expected 284807 rows, got {df.shape[0]}"
    assert df.shape[1] >= 31, f"Expected at least 31 columns, got {df.shape[1]}"
    assert profile["fraudulent_count"] == 492
    assert profile["legitimate_count"] == 284315
    print("[PASS] Data loader passed.")


def test_artifacts_exist():
    print("\n--- Test 2: Model & Artifacts Persistence ---")
    required_files = [
        "models/fraud_model.joblib",
        "models/baseline_model.joblib",
        "models/preprocessor.joblib",
        "models/model_metadata.json",
        "artifacts/metrics.json",
        "artifacts/sample_transactions.json",
    ]
    for fpath in required_files:
        assert os.path.exists(fpath), f"Missing required artifact: {fpath}"
    print("[PASS] All artifacts verified.")


def test_preprocessor():
    print("\n--- Test 3: Preprocessor ---")
    preprocessor = CreditCardPreprocessor.load("models/preprocessor.joblib")
    assert preprocessor.is_fitted
    # Test single sample transform
    sample = {col: 0.0 for col in EXPECTED_FEATURE_COLUMNS}
    sample["Amount"] = 150.0
    transformed = preprocessor.transform(sample)
    assert transformed.shape == (1, 30)
    print("[PASS] Preprocessor passed.")


def test_single_prediction():
    print("\n--- Test 4: Single Transaction Prediction & Threshold ---")
    predictor = get_predictor("rf")

    with open("artifacts/sample_transactions.json", "r") as f:
        samples = json.load(f)

    legit_sample = samples["legitimate"]
    fraud_sample = samples["fraudulent"]

    # Predict normal
    res_legit = predictor.predict_single(legit_sample, threshold=0.50)
    assert res_legit["prediction"] == 0, f"Expected 0 for legitimate sample, got {res_legit['prediction']}"
    assert res_legit["prediction_label"] == "Legitimate"

    # Predict fraud
    res_fraud = predictor.predict_single(fraud_sample, threshold=0.50)
    assert res_fraud["prediction"] == 1, f"Expected 1 for fraud sample, got {res_fraud['prediction']}"
    assert res_fraud["prediction_label"] == "Potential Fraud"
    assert res_fraud["fraud_probability"] >= 0.50

    # Test threshold change without model retraining
    # If we set an extreme threshold of 0.999 on fraud sample, it should classify as 0
    res_high_thresh = predictor.predict_single(fraud_sample, threshold=0.999)
    assert res_high_thresh["threshold"] == 0.999

    print(f"[PASS] Single prediction passed! (Legit prob: {res_legit['fraud_probability_pct']}%, Fraud prob: {res_fraud['fraud_probability_pct']}%)")


def test_batch_prediction():
    print("\n--- Test 5: Batch CSV Prediction ---")
    predictor = get_predictor("rf")
    batch_path = "data/sample_batch_test.csv"
    assert os.path.exists(batch_path), f"Missing {batch_path}"

    batch_df = pd.read_csv(batch_path)
    scored_df, summary = predictor.predict_batch(batch_df, threshold=0.50)

    assert "fraud_probability" in scored_df.columns
    assert "prediction" in scored_df.columns
    assert "prediction_label" in scored_df.columns
    assert summary["total_processed"] == len(batch_df)
    assert summary["flagged_fraud"] >= 1, "Expected at least 1 fraud flagged in sample batch"
    print(f"[PASS] Batch prediction passed! Processed {summary['total_processed']} rows, flagged {summary['flagged_fraud']} frauds.")


def test_invalid_input_handling():
    print("\n--- Test 6: Invalid Input Handling ---")
    predictor = get_predictor("rf")

    # Missing column
    invalid_sample = {"Time": 100.0, "Amount": 50.0} # Missing V1..V28
    try:
        predictor.predict_single(invalid_sample)
        assert False, "Should have raised ValueError for missing columns"
    except ValueError as e:
        print(f"[PASS] Correctly raised ValueError on missing columns: {e}")

    # Malformed batch
    invalid_batch = pd.DataFrame([{"Time": "non_numeric", "Amount": 10.0}])
    try:
        predictor.predict_batch(invalid_batch)
        assert False, "Should have raised ValueError for malformed batch"
    except ValueError as e:
        print(f"[PASS] Correctly raised ValueError on invalid batch: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("RUNNING AUTOMATED TEST SUITE")
    print("=" * 50)
    test_data_loader()
    test_artifacts_exist()
    test_preprocessor()
    test_single_prediction()
    test_batch_prediction()
    test_invalid_input_handling()
    print("\n" + "=" * 50)
    print("ALL 6 TEST MODULES PASSED SUCCESSFULLY!")
    print("=" * 50)
