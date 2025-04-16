# Loan-Default-Prediction-System

# 🧠 Loan Default Prediction System

A comprehensive **Loan Default Prediction System** built using Machine Learning models like **XGBoost**, **Logistic Regression**, **Random Forest**, **KNN**, and enhanced with a **Psychological Risk Survey** (PRS) component.

---

## 🌐 Overview

This project leverages both **transactional data** and **psychological indicators** to predict the likelihood of a customer defaulting on a loan. It is implemented in two key phases:

- **Phase 1**: Traditional Machine Learning pipeline
- **Phase 2**: Psychological Survey-based Risk Scoring

✅ **Overall Model Accuracy: 99.77%**

---

## 📊 Phase 1: Machine Learning Pipeline

### Components:

1. **Data Pipeline**
   - Raw data ingestion (loan info, customer profile, transaction history)
   - Cleaning, missing value handling
   - Feature engineering (e.g., Debt-to-Income ratio, credit score)
   - Train-Test split via temporal split or k-fold validation

2. **Model Training & Individual Accuracies**
   - ✅ **XGBoost** – Gradient Boosted Trees — **100% Accuracy**
   - ✅ **Logistic Regression** – Linear Classifier — **94% Accuracy**
   - ✅ **Random Forest** – Ensemble of Decision Trees — **100% Accuracy**
   - ✅ **K-Nearest Neighbors** – Distance-Based Classifier — **94% Accuracy**

3. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, AUC-ROC
   - Model comparison & selection

4. **Deployment**
   - Model serialization (`joblib`, `SavedModel`)
   - API with Flask / FastAPI
   - Dashboard for real-time predictions

---

## 🔍 Phase 2: Psychological Survey Component (Triggered under Specific Conditions)

> **Condition**: If model shows **<40% acceptance** and **>60% rejection**, the PRS component is activated.

### Psychological Risk Score (PRS) Flow:
- **Survey Design**: 10 questions targeting 6 psychological parameters
- **Parameters**:
  - FD: Financial Discipline  
  - RT: Risk Tolerance  
  - LS: Lifestyle Habits  
  - DA: Decision Agility  
  - SM: Spending Mindset  
  - PB: Peer Behavior
- **Scoring**:
  - Likert Scale 1–5
  - Normalized and aggregated to form PRS (range 0–100)
- **Risk Classification**:
  - High Risk (<60)  
  - Medium Risk (60–79)  
  - Low Risk (80+)
- **Integration**:
  - PRS score added to model as a new feature

---


## 🛠 Tech Stack

- **Languages**: Python
- **Libraries**: scikit-learn, XGBoost, pandas, NumPy
- **API**: Flask / FastAPI
- **Visualization**: PlantUML
- **Deployment**: Docker (optional), REST API
- **Frontend (Optional)**: Streamlit / Dash / React

---

## 🚀 Future Enhancements

- Integrate additional NLP sentiment from customer interactions
- Add model explainability (SHAP / LIME)
- Expand psychological survey with adaptive questioning (based on responses)

---

If you like this project, feel free to ⭐ the repo and contribute!

---


![image](https://github.com/user-attachments/assets/0c4fa5f3-6f4a-4b41-85d8-1dd01d2a8e71)

