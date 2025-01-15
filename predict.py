import pandas as pd

from keras import Model
from keras.src.saving import load_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from typing import cast, Any
from utils.IOUtils import IOUtils

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
        self.__convert_x_to_df()
        self.__convert_geography_to_ohe()
        self.__convert_gender_to_label_encoding()
        self.__standardize_x_values()

    def __convert_x_to_df(self):
        self.__x = pd.DataFrame([self.__x])

    def __convert_geography_to_ohe(self):
        ohe_geography_df = self.__ohe_geography_df()
        self.__x.drop('Geography', axis=1, inplace=True)
        self.__x = pd.concat([self.__x, ohe_geography_df], axis=1)

    def __ohe_geography_df(self) -> pd.DataFrame:
        ohe_geography = self.__onehot_encoder_geo.transform(self.__x[['Geography']])
        ohe_geography_df = pd.DataFrame(ohe_geography.toarray(),
                                        columns=self.__onehot_encoder_geo.get_feature_names_out())
        return ohe_geography_df

    def __convert_gender_to_label_encoding(self):
        self.__x['Gender'] = self.__label_encoder_gender.transform(self.__x['Gender'])

    def __standardize_x_values(self):
        self.__x = self.__scaler.transform(self.__x)


if __name__ == '__main__':
    churn_predictor = ChurnPredictor()
    prediction_probability = churn_predictor.predict_likelihood(input_data)

    print(f"Prediction: {prediction_probability}")
    if prediction_probability > 0.5:
        print("Likely to churn")
    else:
        print("Not likely to churn")
