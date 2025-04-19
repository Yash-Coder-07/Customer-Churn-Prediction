import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.title("Customer Churn Prediction")
st.write("Fill in the customer details to predict churn probability")

# Load the model and encoders with error handling
try:
    # Load model (assuming it might be stored in a dictionary)
    with open('customer_churn_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        
        # Handle different possible model storage formats
        if isinstance(model_data, dict):
            model = model_data.get('model') or model_data.get('classifier') or model_data.get('random_forest')
            if model is None:
                st.error("Model not found in the dictionary. Available keys: " + ", ".join(model_data.keys()))
                st.stop()
        else:
            model = model_data
    
    # Load encoders
    with open('encoders.pkl', 'rb') as f:
        encoder = pickle.load(f)
        
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# Input form layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

# Create input DataFrame
input_dict = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Prediction logic
if st.button("Predict Churn", type="primary"):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical variables
        df_encoded = input_df.copy()
        for column, le in encoder.items():
            if column in df_encoded.columns:
                # Handle unseen labels by assigning a default value
                try:
                    df_encoded[column] = le.transform(df_encoded[column])
                except ValueError:
                    df_encoded[column] = -1  # or use le.classes_[0] for first category
        
        # Make prediction
        prediction = model.predict(df_encoded)
        prediction_proba = model.predict_proba(df_encoded)
        
        # Display results
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"⚠️ High Churn Risk ({prediction_proba[0][1]*100:.1f}% probability)")
            st.write("This customer is likely to churn. Consider retention offers.")
        else:
            st.success(f"✅ Low Churn Risk ({prediction_proba[0][0]*100:.1f}% probability)")
            st.write("This customer is likely to stay.")
            
        # Show feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.subheader("Top Factors Influencing This Prediction")
            importances = pd.DataFrame({
                'Feature': df_encoded.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            st.bar_chart(importances.set_index('Feature'))
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Please check your input values and try again.")

# Add some styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)