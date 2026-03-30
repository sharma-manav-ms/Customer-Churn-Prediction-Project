# Customer Churn Prediction

A machine learning project for predicting telecom customer churn using XGBoost, featuring SHAP explainability and an interactive Streamlit web application.

## Features

- **Predictive Modeling**: XGBoost classifier with class imbalance handling
- **Explainability**: SHAP-based feature attribution with waterfall plots
- **Web Interface**: Streamlit app for real-time predictions
- **Feature Engineering**: Custom features like charge ratio and service counts
- **Evaluation**: Comprehensive metrics including ROC-AUC and cross-validation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sharma-manav-ms/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the Model** (run once):
   ```bash
   python train_model.py
   ```
   This downloads the dataset, preprocesses data, trains the model, and generates evaluation plots.

2. **Run the Web App**:
   ```bash
   streamlit run app.py
   ```
   Access the application at `http://localhost:8501` to input customer profiles and view predictions with SHAP explanations.

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── telco_churn.csv          # Auto-downloaded dataset
├── model/
│   ├── churn_model.pkl          # Trained XGBoost model
│   ├── scaler.pkl               # Feature scaler
│   ├── feature_cols.pkl         # Feature column order
│   └── shap_explainer.pkl       # SHAP explainer object
├── shap_plots/
│   ├── evaluation.png           # Model evaluation plots
│   ├── shap_summary.png         # SHAP summary plot
│   ├── shap_bar.png             # SHAP bar plot
│   └── shap_dependence.png      # SHAP dependence plot
├── train_model.py               # Training pipeline
├── app.py                       # Streamlit application
├── requirements.txt             # Python dependencies
└── README.md
```

## Technologies

- **Python 3.x**
- **XGBoost** for classification
- **SHAP** for model explainability
- **Streamlit** for web interface
- **Scikit-learn** for preprocessing and evaluation
- **Pandas & NumPy** for data manipulation
- **Matplotlib** for plotting

## Dataset

The project uses the [IBM Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) from Kaggle, containing 7,043 customer records with 20 features including demographics, service subscriptions, and billing information.

## Model Performance

- **Test Accuracy**: ~81%
- **Test ROC-AUC**: ~86%
- **Cross-Validation ROC-AUC** (5-fold): ~84%

## Author

**Manav Sharma**  
[LinkedIn](https://www.linkedin.com/in/manav-sharma-682b0021b) 
[GitHub](https://github.com/sharma-manavms)
