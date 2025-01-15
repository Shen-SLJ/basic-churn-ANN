from typing import cast

import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from keras.src.saving import load_model

from core.ChurnPredictor import ChurnPredictor
from utils.IOUtils import IOUtils

# Variables
model = load_model('model.keras')
label_encoder_gender = cast(LabelEncoder, IOUtils.pickle_load_file('label_encoder_gender.pkl'))
onehot_encoder_geo = cast(OneHotEncoder, IOUtils.pickle_load_file('onehot_encoder_geo.pkl'))
scaler = cast(StandardScaler, IOUtils.pickle_load_file('scaler.pkl'))

# Website input fields
st.set_page_config(page_title="Basic-Churn-ANN")

st.title("Basic-Churn-ANN")
st.caption(
    "Basic neural network trained to solve binary classification problem of whether a customer will exit based on input data."
)

credit_score = st.number_input('Credit score', min_value=0, max_value=999)
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 95)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.checkbox('Has credit card')
is_active_member = st.checkbox('Is active')
estimated_salary = st.number_input('Estimated salary', min_value=0)

# Predict
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': 1 if has_credit_card else 0,
    'IsActiveMember': 1 if is_active_member else 0,
    'EstimatedSalary': estimated_salary
}

predictor = ChurnPredictor()
likelihood = predictor.predict_likelihood(input_data)

# Write prediction to website
st.divider()
st.write(f"Likelihood: **{likelihood}**")
st.write(f"{"Likely **won't** churn" if likelihood < 0.50 else "Likely **will** churn"}")
