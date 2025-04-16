import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Loan Default Prediction", layout="centered", page_icon="💳")

# ---- Create and Train Simple Model ----
@st.cache_resource
def create_simple_model():
    # Create synthetic training data based on reasonable assumptions
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    income = np.random.normal(5000, 2000, n_samples)  # Monthly income with mean $5000
    age = np.random.randint(18, 70, n_samples)
    loan_amount = np.random.normal(200000, 100000, n_samples)  # Loan amount with mean $200,000
    term = np.random.choice([120, 180, 240, 360], n_samples)  # Common loan terms in months
    
    # Create features array
    X = np.column_stack([income, age, loan_amount, term])
    
    # Generate target variable (default probability increases with higher loan-to-income ratio and term)
    default_prob = 1 / (1 + np.exp(-(loan_amount/(income*12) + term/360 - 5)))
    y = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

model, scaler = create_simple_model()

# ---- Custom CSS Styling ----
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #e0eafc, #cfdef3);
            padding: 20px;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 12px;
        }
        .stTextInput>div>input {
            padding: 10px;
            border-radius: 8px;
        }
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #333333;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #555555;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown('<p class="title">💳 Advanced Loan Default Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict whether a borrower is likely to default on their loan using AI.</p>', unsafe_allow_html=True)

# ---- Input Form ----
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Monthly Income ($)", min_value=0.0, step=100.0)
        age = st.number_input("Age (years)", min_value=18, max_value=100)

    with col2:
        loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, step=500.0)
        term = st.number_input("Loan Term (months)", min_value=1, max_value=360)

    submitted = st.form_submit_button("🔍 Predict")

# ---- Prediction ----
if submitted:
    # Scale the input data using the same scaler used for training
    input_data = np.array([income, age, loan_amount, term]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    # ---- Display Result ----
    st.markdown("### 🔎 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The applicant is **likely to default** on the loan. (Probability: {probability:.1%})")
    else:
        st.success(f"✅ The applicant is **not likely to default** on the loan. (Probability: {probability:.1%})")

    # ---- Show Summary ----
    with st.expander("📊 View Input Summary"):
        st.write({
            "Monthly Income": f"${income:,.2f}",
            "Age": f"{age} years",
            "Loan Amount": f"${loan_amount:,.2f}",
            "Term": f"{term} months",
            "Default Risk": f"{probability:.1%}"
        })
