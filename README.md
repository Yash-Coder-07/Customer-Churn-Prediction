#  Telco Customer Churn Prediction 
**Predicting customer attrition using machine learning**  
# Overview
A machine learning solution to predict telecom customer churn, achieving **84% cross-validation accuracy** with Random Forest. Addresses class imbalance using **SMOTE** and compares multiple models (XGBoost, Decision Tree).

## 📊 Key Results
| Metric                  | Score |
|-------------------------|-------|
| **Test Accuracy**       | 78%   |
| **Non-Churn Precision** | 85%   |
| **Churn Recall**        | 59%   |

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Pickle

**📈 Methodology**
1. Data Preprocessing
     Handled missing values in TotalCharges
     Encoded categorical features (LabelEncoder)

2. Exploratory Analysis (EDA)
    Analyzed feature distributions (tenure, charges)
    Identified churn drivers (contract type, internet service)

3. Modeling
    Applied SMOTE for class imbalance
    Evaluated Random Forest (best performance)
    Saved model with pickle for deployment

💼 Business Impact
1. Identified top 3 churn drivers:
     - Month-to-month contracts
     - Fiber optic internet users
     - Customers with high monthly charges
Potential to reduce churn by 27% with targeted retention campaigns

## 🌟 Live Demo  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-c7poefg4v2nkp8jyzgaivh.streamlit.app/)
