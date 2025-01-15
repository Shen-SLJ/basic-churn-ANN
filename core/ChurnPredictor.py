from typing import cast, Any

from keras import Model
from keras.src.saving import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from utils.ChurnDataPreprocessingUtils import ChurnDataPreprocessingUtils
from utils.IOUtils import IOUtils
from utils.PandaUtils import PandaUtils


class ChurnPredictor:
    FILEPATH_MODEL = 'model/model.keras'
    FILEPATH_LABEL_ENCODER_GENDER = 'preprocessors/label_encoder_gender.pkl'
    FILEPATH_ONEHOT_ENCODER_GEO = 'preprocessors/onehot_encoder_geo.pkl'
    FILEPATH_SCALER = 'preprocessors/scaler.pkl'

    def __init__(self):
        self.__model = cast(Model, load_model(self.FILEPATH_MODEL))
        self.__label_encoder_gender = cast(LabelEncoder, IOUtils.pickle_load_file(self.FILEPATH_LABEL_ENCODER_GENDER))
        self.__onehot_encoder_geo = cast(OneHotEncoder, IOUtils.pickle_load_file(self.FILEPATH_ONEHOT_ENCODER_GEO))
        self.__scaler = cast(StandardScaler, IOUtils.pickle_load_file(self.FILEPATH_SCALER))

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
