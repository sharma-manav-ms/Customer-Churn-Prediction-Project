# 🚀 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges for subscription-based businesses. Losing customers directly impacts revenue and growth.
This project aims to **predict whether a customer will churn** so that companies can take proactive measures to retain them.

---

## 💡 Solution Overview

This project uses a **Machine Learning model (XGBoost)** to predict churn probability based on customer data.
It also integrates **SHAP (SHapley Additive exPlanations)** to explain *why* a prediction was made.

An interactive **Streamlit web app** allows users to input customer details and get real-time predictions.

---

## ⚙️ Features

* 🔍 Predict churn probability
* 📊 Risk classification (Low / Medium / High)
* 🧠 Model explainability using SHAP
* 📈 Visual insights with SHAP summary & waterfall plots
* 🎯 User-friendly Streamlit interface

---

## 🛠 Tech Stack

* **Programming:** Python
* **Libraries:** Scikit-learn, XGBoost, SHAP, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment/UI:** Streamlit

---

## 📊 Model Performance

* **ROC-AUC Score:** 0.84
* **Validation:** 5-Fold Cross Validation
* **Handling Imbalance:** scale_pos_weight in XGBoost

---

## ⚖️ Model Comparison

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | 0.78    |
| Random Forest       | 0.82    |
| XGBoost             | 0.84    |

---

## 🔍 Key Insights

* Customers with **month-to-month contracts** are more likely to churn
* **Fiber optic users** show higher churn probability
* **Online security services** reduce churn risk
* Higher monthly charges can increase churn likelihood

---

## 📈 Business Recommendations

* 🎯 Encourage customers to switch to long-term contracts
* 💰 Offer discounts or retention offers to high-risk users
* 🔒 Promote add-on services like online security
* 📞 Target high-risk customers with personalized outreach

---

## 🧠 Model Explainability (SHAP)

* 🔴 Red bars → Increase churn probability
* 🔵 Blue bars → Decrease churn probability
* 📏 Longer bars → Greater impact on prediction

This helps in understanding the **reason behind each prediction**, making the model transparent and trustworthy.

---

## 📸 Screenshots

### 🔹 Application Interface

(Add your screenshot here)

### 🔹 Prediction Output

(Add your screenshot here)

### 🔹 SHAP Explanation

(Add your screenshot here)

### 🔹 Model Evaluation

(Add your screenshot here)

---

## 🚀 Live Demo

👉 (Add your Streamlit deployment link here)

---

## 📈 Business Recommendations
* Encourage long-term contracts to reduce churn
* Offer incentives for high-risk customers
* Promote online security services to improve retention

---

## 🛠 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/sharma-manav-ms/Customer-Churn-Prediction-Project.git
cd Customer-Churn-Prediction-Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Customer-Churn-Prediction-Project/
│
├── app.py
├── model/
├── data/
├── notebooks/
├── requirements.txt
├── README.md
```

---

## 📌 Future Improvements

* 🌐 Deploy the app online (Streamlit Cloud)
* 📊 Add more model comparisons
* ⚡ Improve UI/UX design
* 🔄 Real-time data integration

---

## 👨‍💻 Author

**Manav Sharma**
📌 Aspiring Data Scientist | Machine Learning Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!
