<div align="center">

# 🛡️ FraudShield — Credit Card Fraud Detection System

### Machine Learning | Fraud Detection | Interactive Analytics

A machine learning system for detecting fraudulent credit card transactions using Logistic Regression and Random Forest, with preprocessing, threshold-based prediction, model evaluation, and an interactive Streamlit dashboard.

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

**FraudShield** is an end-to-end machine learning application designed to identify potentially fraudulent credit card transactions in a highly imbalanced dataset.

The system compares **Logistic Regression** as a baseline model with **Random Forest** as the primary classifier and provides an interactive Streamlit interface for transaction analytics, single prediction, batch prediction, and model evaluation.

This project is developed for **educational and portfolio purposes**, demonstrating practical machine learning, evaluation, and deployment concepts.

---

## ✨ Features

- **Dual ML Models**
  - Logistic Regression baseline
  - Random Forest primary classifier
- **Leakage-Free Preprocessing**
  - `StandardScaler` fitted only on training data
  - Consistent preprocessing during inference
  - Fixed feature ordering
- **Interactive Analytics Dashboard**
  - Dataset overview & health metrics
  - Fraud vs. legitimate distribution
  - Transaction amount analysis
  - Fraud distribution by hour of the day
  - Confusion matrix heatmap
  - ROC curve comparison
  - Precision-Recall curve
  - Random Forest feature importance
- **Single Transaction Prediction**
  - Manual transaction parameter input
  - Quick-load authentic test-set samples
  - Fraud probability calculation
  - Adjustable decision threshold
  - Color-coded fraud / legitimate classification
- **Batch Prediction**
  - Upload transaction CSV
  - Process multiple transactions concurrently
  - Generate fraud probabilities and predictions
  - Download annotated prediction results as CSV
- **Comprehensive Model Evaluation**
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - PR-AUC (Average Precision)
  - Balanced Accuracy
  - Confusion Matrix

---

## 🏗️ System Architecture

```text
┌──────────────────────┐
│   Credit Card Data   │
│    creditcard.csv    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Data Validation    │
│ Schema & Data Checks │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Train/Test Split   │
│   80:20 Stratified   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Preprocessing     │
│    StandardScaler    │
│    Time + Amount     │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌──────────────┐ ┌──────────────┐
│   Logistic   │ │Random Forest │
│  Regression  │ │Primary Model │
└──────┬───────┘ └──────┬───────┘
       │                │
       └──────┬─────────┘
              ▼
┌──────────────────────┐
│   Model Evaluation   │
│  Precision / Recall  │
│F1 / ROC-AUC / PR-AUC │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Serialized Artifacts │
│ Models + Preprocessor│
│  Metrics + Metadata  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Streamlit Dashboard  │
├──────────────────────┤
│      Analytics       │
│  Single Prediction   │
│   Batch Prediction   │
│   Model Comparison   │
└──────────────────────┘
```

---

## 🧠 Machine Learning Pipeline

```text
Raw Transaction
      ↓
Data Validation
      ↓
Feature / Target Separation
      ↓
Stratified Train-Test Split (80:20)
      ↓
Fit Preprocessor on Training Data
      ↓
Transform Train & Test Data
      ↓
Train Logistic Regression (Baseline)
      ↓
Train Random Forest (Primary)
      ↓
Evaluate Models
      ↓
Save Models + Metrics + Preprocessor
      ↓
Streamlit Inference
```

### Data Leakage Prevention
Preprocessing transformations are fitted **only on the training dataset** and subsequently applied to test and inference data. This guarantees that zero information from the test set or future transactions influences model training or parameter estimation.

---

## 📊 Model Performance

Evaluation was performed on an unseen 20% stratified test set containing **56,962 transactions** and **98 actual fraudulent transactions**.

| Metric | Logistic Regression (Baseline) | Random Forest (Primary Model) |
| :--- | :---: | :---: |
| **Precision (Fraud Class 1)** | 82.89% | **94.12%** |
| **Recall (Fraud Class 1)** | 64.29% | **81.63%** |
| **F1-Score (Fraud Class 1)** | 0.7241 | **0.8743** |
| **ROC-AUC** | 0.9559 | **0.9630** |
| **PR-AUC (Average Precision)** | 0.7432 | **0.8734** |
| **Balanced Accuracy** | 82.13% | **90.81%** |
| **Overall Accuracy** | 99.92% | **99.96%** |

### Random Forest Confusion Matrix

| | Predicted Legitimate | Predicted Fraud | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Legitimate** | 56,859 (TN) | 5 (FP) | 56,864 |
| **Actual Fraud** | 18 (FN) | 80 (TP) | 98 |

> [!IMPORTANT]
> Because fraud represents only **0.17%** of the dataset, accuracy alone is not a reliable measure of performance. Precision, Recall, F1-Score, and PR-AUC are therefore emphasized.

---

## 🎚️ Decision Threshold

FraudShield uses a configurable probability threshold for classification:

$$\text{Fraud Probability} \ge \text{Threshold} \implies \text{Potential Fraud}$$
$$\text{Fraud Probability} < \text{Threshold} \implies \text{Legitimate}$$

* **Default Threshold**: `0.50`
* **Lower Threshold**: Increases Recall (catches more fraud, more false alarms)
* **Higher Threshold**: Increases Precision (fewer false alarms, risks missing fraud)

The threshold can be adjusted dynamically in the UI without retraining the model. Dashboard evaluation metrics are calculated using the default `0.50` threshold.

---

## 📂 Dataset

The project uses the **Credit Card Fraud Detection** dataset from the ULB Machine Learning Group / Kaggle. The full `creditcard.csv` dataset is not included in the repository due to its size. Download it separately and place it in `data/creditcard.csv`.

### Dataset Statistics

| Property | Value |
| :--- | :--- |
| **Total Transactions** | 284,807 |
| **Legitimate Transactions** | 284,315 |
| **Fraudulent Transactions** | 492 |
| **Fraud Rate** | 0.1727% |
| **Total Features** | 30 numerical variables |
| **Observation Period** | 48 hours (September 2013) |

### Feature Summary
* `Time` — Number of seconds elapsed between each transaction and the first transaction.
* `Amount` — Transaction amount in Euros.
* `V1` – `V28` — Anonymized numerical features extracted via PCA for confidentiality.
* `Class` — Target label (`0` = Legitimate, `1` = Fraudulent).

---

## 🛠️ Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core application & machine learning logic |
| **Pandas** | Data validation, manipulation, and batch ingestion |
| **NumPy** | Vectorized array operations and numerical computation |
| **Scikit-learn** | ML classification, preprocessing, and evaluation metrics |
| **Streamlit** | Interactive web application and responsive dashboard |
| **Plotly** | Dark-themed interactive visualizations and charts |
| **Joblib** | Model and preprocessor artifact serialization |

---

## 📁 Project Structure

```text
FraudShield/
│
├── app.py                      # Streamlit dashboard application
├── requirements.txt            # Pinned dependency requirements
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
│
├── data/
│   └── sample_batch_test.csv   # Sample batch file for testing
│
├── models/
│   ├── fraud_model.joblib      # Serialized Random Forest model
│   ├── baseline_model.joblib   # Serialized Logistic Regression model
│   ├── preprocessor.joblib     # Serialized StandardScaler pipeline
│   └── model_metadata.json     # Model hyperparameters & training info
│
├── artifacts/
│   ├── metrics.json            # Evaluation metrics and curve coordinates
│   └── sample_transactions.json# Real sample transactions for UI demo
│
├── notebooks/
│   └── creditcard-fraud-detection.ipynb # Reference analysis notebook
│
└── src/
    ├── __init__.py             # Package marker
    ├── data_loader.py          # Dataset loading, validation, profiling
    ├── preprocessing.py       # Leakage-free preprocessing pipeline
    ├── train.py                # Model training and artifact serialization
    ├── evaluate.py             # Evaluation metrics and curve generation
    └── predict.py              # Single and batch inference engine
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd FraudShield
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Models
```bash
python -m src.train
```
*This executes the leakage-free training pipeline and outputs the trained models, preprocessor, evaluation metrics, metadata, and sample transactions into `models/` and `artifacts/`.*

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
*The interactive dashboard will open automatically in your browser at `http://localhost:8501`.*

### 6. Run Tests
```bash
python test_system.py
```

---

## 🖥️ Dashboard Overview

The Streamlit application provides five sections:

```text
Dashboard & Analytics
│
├── 📊 Dataset Overview (Total, Fraud, Legit, Fraud Rate)
├── 📈 Class Distribution (Log Scale)
├── 💳 Transaction Amount Analysis & Boxplots
├── 🕒 Hourly Fraud Analysis (0–23h distribution)
├── 🎯 Confusion Matrix Heatmap
├── 📉 ROC Curve Comparison
├── 📊 Precision-Recall Curve
└── 🔍 Random Forest Feature Importance

Single Transaction Prediction
│
├── 📝 Transaction Input (Time, Amount, V1–V28)
├── ⚡ One-Click Authentic Presets (Normal / Fraud)
├── 📊 Probability Progress Bar & Metric Cards
└── ⚠️ Color-Coded Risk Assessment & Disclaimer

Batch CSV Prediction
│
├── 📁 CSV File Uploader & Validation
├── 📥 Sample Batch Download Button
├── 📊 Processed Risk Summary
└── 💾 Download Enriched Predictions CSV

Model Metrics & Comparison
│
├── ⚖️ Side-by-Side Model Metric Comparison
└── 💡 Imbalance Trade-off & Risk Impact Analysis

Methodology & Discrepancies
│
├── 📌 Project Background & PCA Context
├── ⚠️ Reference Discrepancy Clarifications
└── 🔮 Future Roadmap
```

---

## 🔮 Future Enhancements

Potential improvements for future iterations:
- **Resampling Exploration**: Benchmarking SMOTE, Borderline-SMOTE, and ADASYN.
- **Gradient Boosting**: Benchmarking XGBoost, LightGBM, and CatBoost against Random Forest.
- **Explainable AI (XAI)**: Integrating SHAP (SHapley Additive exPlanations) for local waterfall force plots.
- **Cost-Sensitive Learning**: Optimizing decision thresholds using an explicit business cost matrix (cost of false positive customer friction vs. cost of fraud loss).
- **Inference Microservice**: Deploying a FastAPI endpoint for API-based inference.
- **Containerization**: Packaging with Docker and Docker Compose.
- **Monitoring**: Integrating data drift and model performance drift tracking.

*These enhancements are intentionally documented as future iterations to keep the current implementation focused, maintainable, and scoped.*

---

## ⚠️ Limitations

- **Imbalanced Dataset**: The dataset is highly imbalanced (0.17% fraud) and reflects a specific historical European transaction environment (September 2013).
- **Anonymized Features**: Features `V1`–`V28` are PCA-transformed components, which limits direct domain interpretability of individual features.
- **Probabilistic Estimation**: Model outputs represent statistical probabilities, not definitive legal proof of fraud.
- **Scope**: This project is developed for educational and portfolio purposes and should not be used as an automated financial decision-making system without human review.

---

## 🤝 Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a Pull Request with a clear description of your changes.

For major changes, please open an issue first to discuss the proposed update.

---

## 📬 Contact

**Kaustubh Deshmane**  
- **GitHub**: [KaustubhDeshmane](https://github.com/KaustubhDeshmane)  
- **LinkedIn**: [Kaustubh Deshmane](https://www.linkedin.com/in/kaustubh-deshmane-162924278/)  

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

---

<div align="center">

### 🛡️ FraudShield
**Machine Learning for Smarter Fraud Detection**  
*Built with Python, Scikit-learn & Streamlit.*

</div>
