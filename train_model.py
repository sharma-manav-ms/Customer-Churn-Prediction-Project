git import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
PLOT_DIR    = os.path.join(BASE_DIR, "shap_plots")

for d in [DATA_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

DATA_PATH   = os.path.join(DATA_DIR,  "telco_churn.csv")
MODEL_PATH  = os.path.join(MODEL_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
COLS_PATH   = os.path.join(MODEL_DIR, "feature_cols.pkl")

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print("⬇  Downloading Telco dataset ...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
        print(f"   Saved to {DATA_PATH}")
    else:
        print(f"✅ Dataset found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape}  |  Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")
    return df

def preprocess(df: pd.DataFrame):
    df = df.copy()

    df.drop(columns=["customerID"], inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    df["charge_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    premium_services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[premium_services].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    df["is_monthly"] = (df["Contract"] == "Month-to-month").astype(int)

    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling"
    ]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    no_service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in no_service_cols:
        df[col] = df[col].map({"No": 0, "Yes": 1, "No phone service": 0, "No internet service": 0})

    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=False)

    return df

def train(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\n🔄 Training XGBoost ...")
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, scaler, X_train_s, X_test_s, y_train, y_test, X_test, X.columns.tolist()

def evaluate(model, X_test_s, y_test, feature_names):
    y_pred  = model.predict(X_test_s)
    y_prob  = model.predict_proba(X_test_s)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n📊 Test Accuracy : {acc:.4f}")
    print(f"   Test ROC-AUC  : {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Stay', 'Churn'])}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stay", "Churn"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], color="steelblue")
    axes[1].plot([0,1],[0,1],"k--", linewidth=0.8)
    axes[1].set_title(f"ROC Curve (AUC = {roc_auc:.3f})")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "evaluation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")

    importance = pd.Series(model.feature_importances_, index=feature_names)
    top20 = importance.nlargest(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    top20.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Top 20 Feature Importances (XGBoost)", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")

def generate_shap(model, X_train_s, X_test_s, feature_names):
    print("\n🔍 Generating SHAP explanations ...")
    feature_names = list(feature_names)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s)

    X_test_df = pd.DataFrame(X_test_s, columns=feature_names)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test_df,
        plot_type="dot",
        max_display=20,
        show=False,
        color_bar=True,
    )
    plt.title("SHAP Summary — Feature Impact on Churn Prediction", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", max_display=20, show=False)
    plt.title("Mean |SHAP| — Average Feature Contribution", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")

    top_feature_idx = np.abs(shap_values).mean(axis=0).argmax()
    top_feature     = feature_names[top_feature_idx]
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        top_feature_idx, shap_values, X_test_df,
        ax=ax, show=False, dot_size=20
    )
    plt.title(f"SHAP Dependence — '{top_feature}'", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "shap_dependence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")

    explainer_path = os.path.join(MODEL_DIR, "shap_explainer.pkl")
    joblib.dump(explainer, explainer_path)
    print(f"💾 Saved SHAP explainer: {explainer_path}")

    return explainer

def save_artefacts(model, scaler, feature_cols):
    joblib.dump(model,        MODEL_PATH)
    joblib.dump(scaler,       SCALER_PATH)
    joblib.dump(feature_cols, COLS_PATH)
    print(f"\n✅ Model    → {MODEL_PATH}")
    print(f"✅ Scaler   → {SCALER_PATH}")
    print(f"✅ Columns  → {COLS_PATH}")

if __name__ == "__main__":
    print("=" * 55)
    print("  Customer Churn Prediction — Training Pipeline")
    print("=" * 55)

    df_raw = load_data()

    print("\n🔧 Preprocessing ...")
    df_clean = preprocess(df_raw)
    print(f"   Final shape after encoding: {df_clean.shape}")

    model, scaler, X_train_s, X_test_s, y_train, y_test, X_test, feature_cols = train(df_clean)

    evaluate(model, X_test_s, y_test, feature_cols)

    generate_shap(model, X_train_s, X_test_s, feature_cols)

    save_artefacts(model, scaler, feature_cols)

    print("\n🎉 Done!  Now run:  streamlit run app.py")
    print("=" * 55)
