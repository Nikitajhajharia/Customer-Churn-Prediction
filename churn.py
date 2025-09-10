import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ----------------- Load Dataset -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # Clean data
    df = df.dropna()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

df = load_data()

st.title("📱 Customer Churn Prediction App")
st.write("Predict whether a telecom customer will churn or not.")

# ----------------- EDA -----------------
st.subheader("Exploratory Data Analysis")

# Churn distribution
fig, ax = plt.subplots()
sns.countplot(x="Churn", data=df, palette="Set2", ax=ax)
st.pyplot(fig)

# Charges distribution
fig, ax = plt.subplots()
sns.histplot(df["MonthlyCharges"], bins=30, kde=True, ax=ax, color="purple")
st.pyplot(fig)

# Churn by Contract
fig, ax = plt.subplots()
sns.countplot(x="Contract", hue="Churn", data=df, ax=ax, palette="coolwarm")
plt.xticks(rotation=15)
st.pyplot(fig)

# ----------------- Data Prep -----------------
X = df.drop(["customerID", "Churn"], axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------- Models -----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

st.subheader("Model Performance")
performance = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)
    performance[name] = {"Accuracy": acc, "ROC-AUC": auc}

perf_df = pd.DataFrame(performance).T
st.dataframe(perf_df)

# ----------------- Prediction Form -----------------
st.subheader("🔮 Predict Churn for a New Customer")

def user_input():
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges", 20, 120, 70)
    total = tenure * monthly
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    return pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total],
        "Contract_" + contract: [1],
        "InternetService_" + internet: [1],
        "PaymentMethod_" + payment: [1]
    })

input_df = user_input()
input_full = pd.DataFrame(columns=X.columns)
for col in input_full.columns:
    input_full[col] = 0
for col in input_df.columns:
    if col in input_full.columns:
        input_full[col] = input_df[col].values

best_model = models["Gradient Boosting"]
pred_proba = best_model.predict_proba(scaler.transform(input_full))[0][1]

st.write(f"### Churn Probability: {pred_proba:.2f}")
if pred_proba > 0.5:
    st.error("⚠️ This customer is likely to churn!")
else:
    st.success("✅ This customer is likely to stay.")
