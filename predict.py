import pickle
from typing import cast

import pandas as pd

from keras.src.saving import load_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# fake example input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

if __name__ == '__main__':
    model = load_model('model.keras')

    with open('label_encoder_gender.pkl', 'br') as file:
        label_encoder_gender = cast(LabelEncoder, pickle.load(file))

    with open('onehot_encoder_geo.pkl', 'br') as file:
        onehot_encoder_geo = cast(OneHotEncoder, pickle.load(file))

    with open('scalar.pkl', 'br') as file:
        scalar = cast(StandardScaler, pickle.load(file))

    input_data_df = pd.DataFrame([input_data])

    # geo pre-process
    input_onehot_geo = onehot_encoder_geo.transform(input_data_df[['Geography']])
    input_onehot_geo_df = pd.DataFrame(input_onehot_geo.toarray(), columns=onehot_encoder_geo.get_feature_names_out())
    input_data_df.drop(['Geography'], axis=1, inplace=True)
    input_data_df = pd.concat([input_data_df, input_onehot_geo_df], axis=1)

    # gender pre-process
    input_data_df['Gender'] = label_encoder_gender.transform(input_data_df['Gender'])

    # scale all x-values
    input_data_scaled = scalar.transform(input_data_df)

    #predict
    prediction = model.predict(input_data_scaled)
    prediction_probability = prediction[0][0]

    print(f"Prediction: {prediction_probability}")
    if prediction > 0.5:
        print("Likely to churn")
    else:
        print("Not likely to churn")