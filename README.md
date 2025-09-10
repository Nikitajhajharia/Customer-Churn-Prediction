# Customer-Churn-Prediction

A machine learning project to predict whether a telecom customer will **churn** (leave the service) or not.  
Built with **Python, Scikit-learn, and Streamlit**.  

---

## 📊 Dataset
- Source: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- Rows: ~7,000 customers  
- Features: Demographics, services (internet, phone, TV), account details (contract, charges, payment method)  
- Target: `Churn` (Yes / No)  

---

## 🔍 Exploratory Data Analysis (EDA)
Key Insights:
- ~26% of customers churned
- **Month-to-Month contracts** → highest churn
- **Higher Monthly Charges** → higher churn risk
- **Short-term customers** are more likely to churn

(See charts in `/notebooks/churn_analysis.ipynb` and `Customer_Churn_Prediction_with_Charts.pptx`)  

---

## 🤖 Models Used
- Logistic Regression  
- Random Forest  
- Gradient Boosting (**Best Model**)  

| Model                 | Accuracy | ROC-AUC |
|------------------------|----------|---------|
| Logistic Regression    | 78%      | 0.80    |
| Random Forest          | 85%      | 0.87    |
| Gradient Boosting      | **88%**  | **0.90** |

---

## 🚀 Streamlit Web App
An interactive app to explore churn predictions.  

### Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
