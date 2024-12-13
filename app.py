import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f2f6;
        padding: 10px;
        text-align: center;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .high-risk {
        background-color: #ff4b4b33;
        border: 2px solid #FF4B4B;
    }
    .low-risk {
        background-color: #00ff0033;
        border: 2px solid #00FF00;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
model = tf.keras.models.load_model('notebooks/models/model.h5')

# Load the encoders and scaler
with open('notebooks/models/preprocessors/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('notebooks/models/preprocessors/onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('notebooks/models/preprocessors/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Header
st.title('🔄 Customer Churn Prediction')
st.markdown('### Predict customer churn probability using AI')

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader('Personal Information')
    geography = st.selectbox('📍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    tenure = st.slider('⏳ Tenure (years)', 0, 10, 2)

with col2:
    st.subheader('Financial Information')
    balance = st.number_input('💰 Balance', min_value=0.0, format="%.2f")
    credit_score = st.number_input('📊 Credit Score', min_value=300, max_value=850, value=650)
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, format="%.2f")
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card', ['No', 'Yes'])
    is_active_member = st.selectbox('✅ Is Active Member', ['No', 'Yes'])

# Convert Yes/No to 1/0
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Add a predict button
if st.button('Predict Churn Probability'):
    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    
    # Display prediction
    st.markdown("### Prediction Results")
    
    if prediction_proba > 0.5:
        st.markdown(f"""
        <div class="prediction-box high-risk">
            <h2>⚠️ High Churn Risk</h2>
            <h3>Probability: {prediction_proba:.2%}</h3>
            <p>This customer is likely to churn.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box low-risk">
            <h2>✅ Low Churn Risk</h2>
            <h3>Probability: {prediction_proba:.2%}</h3>
            <p>This customer is likely to stay. Continue maintaining good relationship.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer with GitHub icon
st.markdown("""
<div class="footer">
    <a href="https://github.com/Abdisamad001" target="_blank">
        <svg height="32" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="32" data-view-component="true">
            <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
        </svg>
    </a>
    <p>Built by Abdisamad Omar</p>
</div>
""", unsafe_allow_html=True)