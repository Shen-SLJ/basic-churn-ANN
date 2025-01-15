import pandas as pd

from keras import Model
from keras.src.saving import load_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from typing import cast, Any

from utils.ChurnDataPreprocessingUtils import ChurnDataPreprocessingUtils
from utils.IOUtils import IOUtils
from utils.PandaUtils import PandaUtils

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


class ChurnPredictor:
    model_filename = 'model.keras'
    label_encoder_gender_filename = 'label_encoder_gender.pkl'
    onehot_encoder_geo_filename = 'onehot_encoder_geo.pkl'
    scaler_filename = 'scaler.pkl'

    def __init__(self):
        self.__model = cast(Model, load_model(self.model_filename))
        self.__label_encoder_gender = cast(LabelEncoder, IOUtils.pickle_load_file(self.label_encoder_gender_filename))
        self.__onehot_encoder_geo = cast(OneHotEncoder, IOUtils.pickle_load_file(self.onehot_encoder_geo_filename))
        self.__scaler = cast(StandardScaler, IOUtils.pickle_load_file(self.scaler_filename))

        self.__x = None

    def predict_likelihood(self, x: dict[str, Any]) -> float:
        self.__x = x
        self.__preprocess_x()

        prediction = self.__model.predict(self.__x)
        prediction_probability = prediction[0][0]

        return prediction_probability

    def __preprocess_x(self):
        self.__x = PandaUtils.dataframe_from_dict(self.__x)
        self.__x = ChurnDataPreprocessingUtils.df_with_ohe_geography(self.__x, self.__onehot_encoder_geo)
        self.__x['Gender'] = ChurnDataPreprocessingUtils.label_encoded_gender_series_from_df(self.__x,
                                                                                             self.__label_encoder_gender)
        self.__x = ChurnDataPreprocessingUtils.df_standardized(self.__x, self.__scaler)


if __name__ == '__main__':
    churn_predictor = ChurnPredictor()
    prediction_probability = churn_predictor.predict_likelihood(input_data)

    print(f"Prediction: {prediction_probability}")
    if prediction_probability > 0.5:
        print("Likely to churn")
    else:
        print("Not likely to churn")
