import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "model")
PLOT_DIR     = os.path.join(BASE_DIR, "shap_plots")
MODEL_PATH   = os.path.join(MODEL_DIR, "churn_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
COLS_PATH    = os.path.join(MODEL_DIR, "feature_cols.pkl")
EXPLAINER_PATH = os.path.join(MODEL_DIR, "shap_explainer.pkl")

@st.cache_resource
def load_artefacts():
    model     = joblib.load(MODEL_PATH)
    scaler    = joblib.load(SCALER_PATH)
    feat_cols = joblib.load(COLS_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    return model, scaler, feat_cols, explainer

def build_feature_vector(inputs: dict, feat_cols: list) -> pd.DataFrame:
    d = inputs.copy()

    d["gender"]           = 1 if d["gender"] == "Male" else 0
    d["Partner"]          = 1 if d["Partner"] == "Yes" else 0
    d["Dependents"]       = 1 if d["Dependents"] == "Yes" else 0
    d["PhoneService"]     = 1 if d["PhoneService"] == "Yes" else 0
    d["PaperlessBilling"] = 1 if d["PaperlessBilling"] == "Yes" else 0

    for col in ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        d[col] = 1 if d[col] == "Yes" else 0

    d["charge_ratio"] = d["MonthlyCharges"] / (d["TotalCharges"] + 1)
    d["num_services"] = (
        d["OnlineSecurity"] + d["OnlineBackup"] + d["DeviceProtection"] +
        d["TechSupport"] + d["StreamingTV"] + d["StreamingMovies"]
    )
    d["is_monthly"] = 1 if d["Contract"] == "Month-to-month" else 0

    for val in ["DSL", "Fiber optic", "No"]:
        d[f"InternetService_{val}"] = 1 if d["InternetService"] == val else 0

    for val in ["Month-to-month", "One year", "Two year"]:
        d[f"Contract_{val}"] = 1 if d["Contract"] == val else 0

    for val in ["Bank transfer (automatic)", "Credit card (automatic)",
                "Electronic check", "Mailed check"]:
        d[f"PaymentMethod_{val}"] = 1 if d["PaymentMethod"] == val else 0

    for col in ["InternetService", "Contract", "PaymentMethod"]:
        d.pop(col, None)

    row = pd.DataFrame([d])
    for col in feat_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feat_cols]
    return row

def shap_waterfall_fig(explainer, X_scaled: np.ndarray, feat_cols: list):
    sv = explainer.shap_values(X_scaled)
    explanation = shap.Explanation(
        values=sv[0],
        base_values=explainer.expected_value,
        data=X_scaled[0],
        feature_names=feat_cols,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title("Why this prediction? (SHAP waterfall)", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig

def main():
    st.title("📉 Customer Churn Predictor")
    st.markdown(
        "Enter a customer's profile in the sidebar, then click **Predict** "
        "to see the churn probability and a SHAP explanation of *why* the "
        "model made that decision."
    )
    st.divider()

    if not os.path.exists(MODEL_PATH):
        st.error("🚨 Model not found. Please run `python train_model.py` first.")
        st.stop()

    model, scaler, feat_cols, explainer = load_artefacts()

    st.sidebar.header("🧑 Customer Profile")

    with st.sidebar:
        st.subheader("Demographics")
        gender       = st.selectbox("Gender",      ["Male", "Female"])
        senior       = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner      = st.selectbox("Partner",     ["Yes", "No"])
        dependents   = st.selectbox("Dependents",  ["No", "Yes"])

        st.subheader("Account")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method  = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_charges   = st.slider("Total Charges ($)",   0.0, 8700.0,
                                    float(tenure * monthly_charges), step=10.0)

        st.subheader("Phone Services")
        phone_service  = st.selectbox("Phone Service",    ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])

        st.subheader("Internet Services")
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security  = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
        online_backup    = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
        device_prot      = st.selectbox("Device Protection",["No", "Yes", "No internet service"])
        tech_support     = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
        streaming_tv     = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        predict_btn = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

    if predict_btn:
        inputs = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_prot, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }

        row      = build_feature_vector(inputs, feat_cols)
        row_s    = scaler.transform(row)
        prob     = model.predict_proba(row_s)[0][1]
        pred     = int(prob >= 0.5)

        col1, col2, col3 = st.columns(3)
        with col1:
            color = "🔴" if pred else "🟢"
            st.metric("Prediction", f"{color} {'CHURN' if pred else 'STAY'}")
        with col2:
            st.metric("Churn Probability", f"{prob:.1%}")
        with col3:
            risk = "High" if prob > 0.65 else ("Medium" if prob > 0.35 else "Low")
            st.metric("Risk Level", risk)

        st.subheader("Churn Probability")
        bar_color = "#e74c3c" if prob > 0.65 else ("#f39c12" if prob > 0.35 else "#2ecc71")
        st.markdown(
            f"""
            <div style='background:#eee;border-radius:8px;height:22px;margin-bottom:16px;'>
              <div style='width:{prob*100:.1f}%;background:{bar_color};
                          border-radius:8px;height:22px;transition:width 0.5s;'>
              </div>
            </div>
            <p style='text-align:center;font-size:0.9rem;color:#666;'>{prob*100:.1f}% chance of churning</p>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.subheader("🔍 Why did the model predict this?")
        st.caption(
            "**Red bars** push the prediction *towards* churn.  "
            "**Blue bars** push it *away* from churn.  "
            "The longer the bar, the bigger the impact."
        )
        fig = shap_waterfall_fig(explainer, row_s, feat_cols)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.divider()

        st.subheader("📊 Overall Model Performance")
        tab1, tab2, tab3 = st.tabs(["Evaluation", "SHAP Summary", "Feature Importance"])

        with tab1:
            path = os.path.join(PLOT_DIR, "evaluation.png")
            if os.path.exists(path):
                st.image(path, use_column_width=True)
        with tab2:
            path = os.path.join(PLOT_DIR, "shap_summary.png")
            if os.path.exists(path):
                st.image(path, use_column_width=True)
            path2 = os.path.join(PLOT_DIR, "shap_dependence.png")
            if os.path.exists(path2):
                st.image(path2, use_column_width=True)
        with tab3:
            path = os.path.join(PLOT_DIR, "feature_importance.png")
            if os.path.exists(path):
                st.image(path, use_column_width=True)

    else:
        st.info("👈  Fill in the customer profile in the sidebar, then click **Predict Churn**.")

        st.subheader("About this project")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**🤖 Model**\n\nXGBoost with class-imbalance correction via `scale_pos_weight`")
        with cols[1]:
            st.markdown("**📐 Evaluation**\n\nROC-AUC, accuracy, 5-fold cross-validation")
        with cols[2]:
            st.markdown("**💡 Explainability**\n\nSHAP TreeExplainer — per-prediction waterfall charts")

        if os.path.exists(os.path.join(PLOT_DIR, "shap_summary.png")):
            st.subheader("Overall SHAP Feature Impact")
            st.image(os.path.join(PLOT_DIR, "shap_summary.png"), use_column_width=True)


if __name__ == "__main__":
    main()
