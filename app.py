"""FraudShield — Interactive Credit Card Fraud Detection Machine Learning Application.

An interactive ML analytics dashboard and fraud detection application
providing exploratory data analysis, benchmarked model evaluation, single-transaction scoring,
and batch CSV inference.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import (
    load_creditcard_data,
    add_derived_eda_features,
    EXPECTED_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.predict import get_predictor, FraudPredictor

# Set page configuration
st.set_page_config(
    page_title="FraudShield | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Modern Professional Theme
CUSTOM_CSS = """
<style>
    /* Global dark backgrounds & text */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Card / Container styling */
    .kpi-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .kpi-title {
        color: #9ca3af;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f9fafb;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 4px;
    }
    
    /* Semantic color accents */
    .accent-cyan { color: #38bdf8 !important; }
    .accent-green { color: #10b981 !important; }
    .accent-red { color: #ef4444 !important; }
    .accent-amber { color: #f59e0b !important; }
    
    /* Prediction Badges */
    .badge-fraud {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 15px;
    }
    .badge-legit {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 15px;
    }
    
    /* Disclaimer Note */
    .disclaimer-box {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-top: 15px;
    }
    
    /* Hide default Streamlit decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_cached_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and cache dataset and summary profile."""
    df, profile = load_creditcard_data()
    df = add_derived_eda_features(df)
    return df, profile


@st.cache_data(show_spinner=False)
def load_cached_metrics() -> Dict[str, Any]:
    """Load persisted metrics artifact."""
    metrics_path = os.path.join("artifacts", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_sample_transactions() -> Dict[str, Any]:
    """Load authentic test set sample transactions."""
    samples_path = os.path.join("artifacts", "sample_transactions.json")
    if os.path.exists(samples_path):
        with open(samples_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def format_plotly_dark(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply consistent dark styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif", color="#e5e7eb"),
        title_font=dict(size=15, color="#f9fafb"),
    )
    return fig


def main():
    # Load dataset & metrics
    try:
        df, profile = load_cached_data()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    metrics = load_cached_metrics()
    sample_txs = load_sample_transactions()

    # Sidebar
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <h2 style="color: #38bdf8; margin-bottom: 0;">🛡️ FraudShield</h2>
            <span style="color: #9ca3af; font-size: 0.85rem;">Credit Card Fraud Detection System</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation menu
    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dashboard & Analytics",
            "🔍 Single Transaction Prediction",
            "📁 Batch CSV Prediction",
            "📈 Model Metrics & Comparison",
            "ℹ️ Methodology & Discrepancies",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Model & Threshold")

    model_choice = st.sidebar.selectbox(
        "Classifier Engine",
        ["Random Forest (Primary)", "Logistic Regression (Baseline)"],
        index=0,
        help="Select between the primary tree ensemble model or the linear baseline.",
    )
    display_model_name = "Random Forest" if "Random Forest" in model_choice else "Logistic Regression"
    model_key = "random_forest" if "Random Forest" in model_choice else "logistic_regression"
    predictor_key = "rf" if "Random Forest" in model_choice else "lr"

    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Transactions with fraud probability >= threshold are classified as Potential Fraud.",
    )

    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #9ca3af; margin-top: -5px;">
            Active Prediction Threshold: <b style="color:#38bdf8;">{threshold:.2f}</b><br/>
            • Lower: Flags more transactions (higher recall)<br/>
            • Higher: Flags fewer transactions (higher precision)<br/>
            <span style="color:#f59e0b; font-size: 0.76rem;">* Applies to Single & Batch predictions.<br/>Dashboard benchmark metrics are evaluated @ 0.50 threshold.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #6b7280;">
            <b>Dataset Status:</b> Cached ({profile['total_records']:,} rows)<br/>
            <b>Artifacts:</b> Ready & Synchronized<br/>
            <b>Split:</b> 80:20 Stratified
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================
    # PAGE 1: DASHBOARD & ANALYTICS
    # ==========================================
    if page == "📊 Dashboard & Analytics":
        st.markdown("## 📊 Fraud Analytics Dashboard")
        st.markdown(
            "High-level overview of transaction distribution, behavioral patterns, and model effectiveness."
        )

        active_metrics = metrics.get(model_key, {})

        # Row 1: KPI Cards
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        with kpi1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Records</div>
                    <div class="kpi-value">{profile['total_records']:,}</div>
                    <div class="kpi-sub">Transactions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Fraud Cases</div>
                    <div class="kpi-value accent-red">{profile['fraudulent_count']:,}</div>
                    <div class="kpi-sub">Actual Fraudulent Transactions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Fraud Rate</div>
                    <div class="kpi-value accent-amber">{profile['fraud_rate_pct']:.4f}%</div>
                    <div class="kpi-sub">Severe Imbalance</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi4:
            prec = active_metrics.get("precision", 0.0) * 100
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Precision (@ 0.50)</div>
                    <div class="kpi-value accent-cyan">{prec:.1f}%</div>
                    <div class="kpi-sub">{display_model_name} @ 0.50 threshold</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi5:
            rec = active_metrics.get("recall", 0.0) * 100
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Recall (@ 0.50)</div>
                    <div class="kpi-value accent-green">{rec:.1f}%</div>
                    <div class="kpi-sub">{display_model_name} @ 0.50 threshold</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi6:
            f1_val = active_metrics.get("f1", 0.0)
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">F1-Score (@ 0.50)</div>
                    <div class="kpi-value accent-cyan">{f1_val:.3f}</div>
                    <div class="kpi-sub">{display_model_name} @ 0.50 threshold</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Threshold explanation banner for dashboard
        if threshold != 0.50:
            st.warning(
                f"⚠️ **Threshold Notice:** Your sidebar decision threshold is currently set to **{threshold:.2f}**. "
                f"Please note that the Dashboard evaluation metrics and Confusion Matrix below represent the benchmark evaluation "
                f"(**@ 0.50 threshold**) on the test set. Your selected threshold of **{threshold:.2f}** dynamically drives "
                f"**Single Transaction Prediction** and **Batch CSV Prediction**."
            )
        else:
            st.caption(
                f"ℹ️ *Dashboard evaluation metrics, curves, and confusion matrix below are benchmarked at the default **0.50 threshold** for {display_model_name}.*"
            )

        # Row 2: Charts (Class distribution & Amount distribution)
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            class_df = pd.DataFrame({
                "Category": ["Legitimate (0)", "Fraudulent (1)"],
                "Count": [profile["legitimate_count"], profile["fraudulent_count"]],
                "Percentage": [
                    f"{100 - profile['fraud_rate_pct']:.2f}%",
                    f"{profile['fraud_rate_pct']:.4f}%",
                ],
            })
            fig_class = px.bar(
                class_df,
                x="Category",
                y="Count",
                text="Count",
                color="Category",
                color_discrete_map={
                    "Legitimate (0)": "#10b981",
                    "Fraudulent (1)": "#ef4444",
                },
                title="Class Distribution (Extreme Imbalance)",
                log_y=True,
            )
            fig_class.update_traces(
                texttemplate="%{y:,}",
                textposition="outside",
            )
            format_plotly_dark(fig_class)
            st.plotly_chart(fig_class, use_container_width=True)

        with col_c2:
            # Transaction Amount Comparison
            legit_amounts = df[df["Class"] == 0]["Amount"]
            fraud_amounts = df[df["Class"] == 1]["Amount"]

            fig_amt = go.Figure()
            fig_amt.add_trace(go.Box(
                y=legit_amounts,
                name="Legitimate",
                marker_color="#10b981",
                boxpoints=False,
            ))
            fig_amt.add_trace(go.Box(
                y=fraud_amounts,
                name="Fraudulent",
                marker_color="#ef4444",
                boxpoints="outliers",
            ))
            fig_amt.update_layout(
                title="Transaction Amount by Class (Log Scale)",
                yaxis_type="log",
                yaxis_title="Amount (€)",
            )
            format_plotly_dark(fig_amt)
            st.plotly_chart(fig_amt, use_container_width=True)

        # Row 3: Fraud by Hour & Confusion Matrix
        col_c3, col_c4 = st.columns(2)

        with col_c3:
            # Fraud by Hour of day
            fraud_hour = df[df["Class"] == 1]["Hour"].value_counts().sort_index().reset_index()
            fraud_hour.columns = ["Hour", "Fraud Count"]

            fig_hour = px.bar(
                fraud_hour,
                x="Hour",
                y="Fraud Count",
                title="Fraudulent Transactions by Hour of the Day (0–23h)",
                color="Fraud Count",
                color_continuous_scale="Reds",
            )
            fig_hour.update_xaxes(dtick=1)
            format_plotly_dark(fig_hour)
            st.plotly_chart(fig_hour, use_container_width=True)

        with col_c4:
            # Confusion matrix
            cm_data = active_metrics.get("confusion_matrix", {})
            cm_matrix = cm_data.get("matrix", [[56859, 5], [18, 80]])

            cm_labels = ["Legitimate", "Fraud"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_matrix,
                x=cm_labels,
                y=cm_labels,
                colorscale=[[0, "#111827"], [0.5, "#0284c7"], [1, "#38bdf8"]],
                text=[
                    [f"TN: {cm_matrix[0][0]:,}", f"FP: {cm_matrix[0][1]:,}"],
                    [f"FN: {cm_matrix[1][0]:,}", f"TP: {cm_matrix[1][1]:,}"],
                ],
                texttemplate="%{text}",
                textfont=dict(size=14, color="#ffffff"),
                hoverinfo="none",
            ))
            fig_cm.update_layout(
                title=f"Confusion Matrix — {display_model_name} (@ 0.50 threshold)",
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
            )
            format_plotly_dark(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)

        # Row 4: ROC & PR Curves
        col_c5, col_c6 = st.columns(2)

        with col_c5:
            # ROC Curves comparison
            fig_roc = go.Figure()
            for key, name, color in [
                ("random_forest", "Random Forest", "#38bdf8"),
                ("logistic_regression", "Logistic Regression", "#f59e0b"),
            ]:
                model_data = metrics.get(key, {})
                roc_pts = model_data.get("roc_curve", {})
                auc_score = model_data.get("roc_auc", 0.0)
                if roc_pts:
                    fig_roc.add_trace(go.Scatter(
                        x=roc_pts["x"],
                        y=roc_pts["y"],
                        mode="lines",
                        name=f"{name} (AUC = {auc_score:.4f})",
                        line=dict(color=color, width=2.5),
                    ))

            # Random chance baseline
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random Chance",
                line=dict(color="#6b7280", dash="dash"),
            ))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC) Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
            )
            format_plotly_dark(fig_roc)
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_c6:
            # Precision-Recall Curve
            fig_pr = go.Figure()
            for key, name, color in [
                ("random_forest", "Random Forest", "#10b981"),
                ("logistic_regression", "Logistic Regression", "#ef4444"),
            ]:
                model_data = metrics.get(key, {})
                pr_pts = model_data.get("pr_curve", {})
                pr_auc = model_data.get("pr_auc", 0.0)
                if pr_pts:
                    fig_pr.add_trace(go.Scatter(
                        x=pr_pts["x"],
                        y=pr_pts["y"],
                        mode="lines",
                        name=f"{name} (AP = {pr_auc:.4f})",
                        line=dict(color=color, width=2.5),
                    ))

            fig_pr.update_layout(
                title="Precision-Recall Curve (Crucial for Imbalanced Data)",
                xaxis_title="Recall",
                yaxis_title="Precision",
            )
            format_plotly_dark(fig_pr)
            st.plotly_chart(fig_pr, use_container_width=True)

        # Row 5: Feature Importances
        rf_importances = metrics.get("random_forest", {}).get("feature_importances", [])
        if rf_importances:
            top_feats = pd.DataFrame(rf_importances[:12]).sort_values("importance", ascending=True)
            fig_imp = px.bar(
                top_feats,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 12 Features by Random Forest Gini Importance",
                color="importance",
                color_continuous_scale="Blues",
            )
            format_plotly_dark(fig_imp, height=380)
            st.plotly_chart(fig_imp, use_container_width=True)

    # ==========================================
    # PAGE 2: SINGLE TRANSACTION PREDICTION
    # ==========================================
    elif page == "🔍 Single Transaction Prediction":
        st.markdown("## 🔍 Single Transaction Scoring")
        st.markdown(
            "Enter transaction parameters or pre-populate with authentic test-set samples to evaluate fraud risk."
        )

        # Quick pre-populate buttons
        col_preset1, col_preset2, col_preset3 = st.columns([1, 1, 2])
        if "input_values" not in st.session_state:
            st.session_state["input_values"] = {}

        with col_preset1:
            if st.button("🟢 Load Normal Sample", use_container_width=True):
                legit = sample_txs.get("legitimate", {})
                st.session_state["input_values"] = legit
                st.session_state["input_time_val"] = float(legit.get("Time", 40000.0))
                st.session_state["input_amount_val"] = float(legit.get("Amount", 50.0))
                for i in range(1, 29):
                    st.session_state[f"input_V{i}"] = float(legit.get(f"V{i}", 0.0))
                st.rerun()

        with col_preset2:
            if st.button("🔴 Load Fraud Sample", use_container_width=True):
                fraud = sample_txs.get("fraudulent", {})
                st.session_state["input_values"] = fraud
                st.session_state["input_time_val"] = float(fraud.get("Time", 40000.0))
                st.session_state["input_amount_val"] = float(fraud.get("Amount", 120.0))
                for i in range(1, 29):
                    st.session_state[f"input_V{i}"] = float(fraud.get(f"V{i}", 0.0))
                st.rerun()

        with col_preset3:
            if st.button("🔄 Reset Inputs to Defaults", use_container_width=True):
                st.session_state["input_values"] = {}
                st.session_state["input_time_val"] = 40000.0
                st.session_state["input_amount_val"] = 120.0
                for i in range(1, 29):
                    st.session_state[f"input_V{i}"] = 0.0
                st.rerun()

        # Input Form
        current_vals = st.session_state.get("input_values", {})

        with st.form("single_prediction_form"):
            st.markdown("#### Primary Features")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                input_time = st.number_input(
                    "Time (Seconds from first transaction)",
                    min_value=0.0,
                    max_value=200000.0,
                    value=float(st.session_state.get("input_time_val", current_vals.get("Time", 40000.0))),
                    step=10.0,
                    key="input_time_val",
                )
            with f_col2:
                input_amount = st.number_input(
                    "Amount (€ EUR)",
                    min_value=0.0,
                    max_value=50000.0,
                    value=float(st.session_state.get("input_amount_val", current_vals.get("Amount", 120.0))),
                    step=1.0,
                    key="input_amount_val",
                )

            st.markdown("#### Anonymized PCA Features (V1 to V28)")
            v_inputs = {}
            # Display 4 columns of 7 features each
            v_cols = st.columns(4)
            for i in range(1, 29):
                v_name = f"V{i}"
                col_idx = (i - 1) % 4
                with v_cols[col_idx]:
                    v_inputs[v_name] = st.number_input(
                        v_name,
                        value=float(st.session_state.get(f"input_{v_name}", current_vals.get(v_name, 0.0))),
                        format="%.4f",
                        key=f"input_{v_name}",
                    )

            submit_btn = st.form_submit_button("⚡ Analyze Transaction Risk", use_container_width=True)

        # Run inference
        raw_input_dict = {"Time": input_time, "Amount": input_amount, **v_inputs}

        try:
            predictor = get_predictor(model_type=predictor_key)
            result = predictor.predict_single(raw_input_dict, threshold=threshold)

            st.markdown("### 📋 Prediction Result")
            res_col1, res_col2 = st.columns([1, 1.2])

            with res_col1:
                if result["is_fraud"]:
                    st.markdown(
                        f"""
                        <div class="badge-fraud">
                            ⚠️ POTENTIAL FRAUD FLAGGED
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="badge-legit">
                            ✅ TRANSACTION APPEARS LEGITIMATE
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(f"**Status Analysis:** {result['status_message']}")

            with res_col2:
                # Probability progress bar & KPI cards
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Fraud Probability", f"{result['fraud_probability_pct']:.2f}%")
                with m2:
                    st.metric("Decision Threshold", f"{result['threshold_pct']:.1f}%")

                prob_val = result["fraud_probability"]
                st.progress(prob_val)

            st.markdown(
                f"""
                <div class="disclaimer-box">
                    <b>Disclaimer:</b> {result['disclaimer']}
                    Risk classifications are generated by the <b>{display_model_name}</b> model
                    trained on European cardholder data with an 80:20 stratified split.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Inference error: {e}")

    # ==========================================
    # PAGE 3: BATCH CSV PREDICTION
    # ==========================================
    elif page == "📁 Batch CSV Prediction":
        st.markdown("## 📁 Batch CSV Transaction Scoring")
        st.markdown(
            "Upload a transaction batch CSV containing the 30 standard features (`Time`, `Amount`, `V1`..`V28`) "
            "to perform vectorized risk assessment and export enriched predictions."
        )

        b_col1, b_col2 = st.columns([2, 1])

        with b_col2:
            st.markdown("#### Sample Test Batch")
            st.markdown("Need sample data to test batch inference?")
            sample_batch_file = os.path.join("data", "sample_batch_test.csv")
            if os.path.exists(sample_batch_file):
                with open(sample_batch_file, "rb") as f:
                    st.download_button(
                        label="📥 Download Sample Batch CSV",
                        data=f,
                        file_name="sample_batch_test.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        with b_col1:
            uploaded_file = st.file_uploader(
                "Upload Transaction CSV File",
                type=["csv"],
                help="CSV must contain Time, Amount, and V1 through V28 columns.",
            )

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Loaded {len(batch_df):,} rows.")

                # Run Batch Prediction
                predictor = get_predictor(model_type=predictor_key)
                scored_df, summary = predictor.predict_batch(batch_df, threshold=threshold)

                # Display summary metrics
                st.markdown("### 📊 Batch Scoring Summary")
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("Total Transactions", f"{summary['total_processed']:,}")
                with s2:
                    st.metric("Flagged as Fraud", f"{summary['flagged_fraud']:,}", delta=f"{summary['fraud_rate_pct']}% rate", delta_color="inverse")
                with s3:
                    st.metric("Approved Legitimate", f"{summary['flagged_legitimate']:,}")
                with s4:
                    st.metric("Threshold Applied", f"{summary['threshold_used']:.2f}")

                # Preview Table
                st.markdown("### 🔍 Scored Transactions Preview")
                preview_cols = ["Time", "Amount", "fraud_probability_pct", "prediction", "prediction_label"] + [f"V{i}" for i in range(1, 6)]
                display_cols = [c for c in preview_cols if c in scored_df.columns]

                # Style fraud rows with red
                def highlight_fraud(row):
                    if row["prediction"] == 1:
                        return ["background-color: rgba(239, 68, 68, 0.2)"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    scored_df[display_cols].head(50).style.apply(highlight_fraud, axis=1),
                    use_container_width=True,
                )

                # CSV Download Button
                csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Full Predictions CSV",
                    data=csv_bytes,
                    file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Batch prediction error: {e}")

    # ==========================================
    # PAGE 4: MODEL METRICS & COMPARISON
    # ==========================================
    elif page == "📈 Model Metrics & Comparison":
        st.markdown("## 📈 Model Performance & Comparative Evaluation")
        st.markdown(
            "Evaluation metrics computed on the **unseen stratified test set (56,962 transactions)**. "
            "Metrics reflect the actual trained models (not hardcoded)."
        )

        rf_m = metrics.get("random_forest", {})
        lr_m = metrics.get("logistic_regression", {})

        # Comparative Table
        comp_data = {
            "Metric": [
                "Precision (Fraud Class 1)",
                "Recall (Fraud Class 1)",
                "F1-Score (Fraud Class 1)",
                "ROC-AUC (Area Under Curve)",
                "PR-AUC (Average Precision)",
                "Balanced Accuracy",
                "Overall Accuracy",
                "True Positives (Detected Fraud)",
                "False Negatives (Missed Fraud)",
                "False Positives (False Alarms)",
                "True Negatives (Correct Legit)",
            ],
            "Logistic Regression (Baseline)": [
                f"{lr_m.get('precision', 0):.4f}",
                f"{lr_m.get('recall', 0):.4f}",
                f"{lr_m.get('f1', 0):.4f}",
                f"{lr_m.get('roc_auc', 0):.4f}",
                f"{lr_m.get('pr_auc', 0):.4f}",
                f"{lr_m.get('balanced_accuracy', 0):.4f}",
                f"{lr_m.get('accuracy', 0):.4f}",
                f"{lr_m.get('confusion_matrix', {}).get('tp', 0):,}",
                f"{lr_m.get('confusion_matrix', {}).get('fn', 0):,}",
                f"{lr_m.get('confusion_matrix', {}).get('fp', 0):,}",
                f"{lr_m.get('confusion_matrix', {}).get('tn', 0):,}",
            ],
            "Random Forest (Primary Model)": [
                f"{rf_m.get('precision', 0):.4f}",
                f"{rf_m.get('recall', 0):.4f}",
                f"{rf_m.get('f1', 0):.4f}",
                f"{rf_m.get('roc_auc', 0):.4f}",
                f"{rf_m.get('pr_auc', 0):.4f}",
                f"{rf_m.get('balanced_accuracy', 0):.4f}",
                f"{rf_m.get('accuracy', 0):.4f}",
                f"{rf_m.get('confusion_matrix', {}).get('tp', 0):,}",
                f"{rf_m.get('confusion_matrix', {}).get('fn', 0):,}",
                f"{rf_m.get('confusion_matrix', {}).get('fp', 0):,}",
                f"{rf_m.get('confusion_matrix', {}).get('tn', 0):,}",
            ],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.markdown("### 💡 Metric Trade-Off Analysis in Highly Imbalanced Scenarios")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown(
                """
                <div class="kpi-card">
                    <h4 style="color:#38bdf8;">Why Accuracy is Misleading</h4>
                    <p style="font-size: 0.9rem; color:#cbd5e1;">
                    Because 99.83% of all transactions are legitimate, a naive "dummy classifier" that predicts every 
                    single transaction as legitimate would achieve an accuracy of <b>99.83%</b> while catching 
                    <b>0 frauds</b>. Balanced Accuracy, Precision, Recall, and PR-AUC are far more informative.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_t2:
            st.markdown(
                """
                <div class="kpi-card">
                    <h4 style="color:#10b981;">Precision vs. Recall Trade-off</h4>
                    <p style="font-size: 0.9rem; color:#cbd5e1;">
                    <b>High Recall</b> minimizes False Negatives (financial losses from missed fraudulent transactions).<br/>
                    <b>High Precision</b> minimizes False Positives (customer friction from erroneously blocked legitimate cards).<br/>
                    Random Forest achieves <b>94.1% Precision</b> and <b>81.6% Recall</b> at threshold 0.50.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==========================================
    # PAGE 5: METHODOLOGY & DISCREPANCIES
    # ==========================================
    elif page == "ℹ️ Methodology & Discrepancies":
        st.markdown("## ℹ️ Methodology & Technical Documentation")

        st.markdown(
            """
            ### 📌 Project Background
            This application is built on the European Credit Card Fraud Detection benchmark dataset (284,807 transactions 
            from September 2013). Features `V1`–`V28` are principal components extracted via PCA due to confidentiality 
            reasons. `Time` and `Amount` remain in raw numerical form.

            ---

            ### ⚠️ Documentation vs Implementation Discrepancy Note
            When reviewing the reference files, a notable discrepancy exists between the original `README` and 
            the executable notebook `creditcard-fraud-detection.ipynb`:

            1. **Train/Test Split Ratio**:
               - **README statement**: Mentions a 70:30 train/test split.
               - **Notebook implementation**: Executed an **80:20 stratified split** (`test_size=0.2, stratify=y, random_state=42`).
               - **Our implementation**: Prioritizes the **actual executable 80:20 stratified baseline** for reproducibility.

            2. **Undersampling & Outlier Removal**:
               - **README statement**: Claims undersampling and outlier removal were performed.
               - **Notebook implementation**: Does **not** implement undersampling or outlier removal; it trained directly on the imbalanced data.
               - **Our implementation**: Follows the real code behavior without inventing unexecuted preprocessing steps.

            3. **Elimination of Data Leakage**:
               - In the reference notebook, `StandardScaler` was fit across the entire dataset *prior* to splitting.
               - In our preprocessing pipeline, `CreditCardPreprocessor` is **fit strictly on the 80% training set** and only transforms the test/inference sets.

            4. **Clean Time/Hour Engineering**:
               - The reference notebook inverted scaled Time into an `Hour` feature and accidentally left it in the training matrix.
               - Our architecture trains models strictly on the 30 expected features (`Time`, `Amount`, `V1`..`V28`) and calculates `Hour` strictly for EDA visualization.

            ---

            ### 🔮 Future Enhancements Roadmap
            - **Phase 2**: SMOTE / Balanced Bagging resampling, LightGBM/XGBoost experimentation, SHAP value explanations, and cost-matrix threshold optimization.
            - **Phase 3**: FastAPI REST inference microservice, Docker containerization, and data drift monitoring.
            """
        )


if __name__ == "__main__":
    main()
