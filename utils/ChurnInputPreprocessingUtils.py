import numpy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


class ChurnInputPreprocessingUtils:
    @staticmethod
    def X_with_ohe_geography(X: pd.DataFrame, onehot_encoder_geo: OneHotEncoder) -> pd.DataFrame:
        ohe_geography_df = ChurnInputPreprocessingUtils.__ohe_geography_df(X[['Geography']], onehot_encoder_geo)

        X.drop('Geography', axis=1, inplace=True)
        X_geo_ohe = pd.concat([X, ohe_geography_df], axis=1)

        return X_geo_ohe

    @staticmethod
    def __ohe_geography_df(geo_df: pd.DataFrame, onehot_encoder_geo: OneHotEncoder) -> pd.DataFrame:
        ohe_geography = onehot_encoder_geo.transform(geo_df)
        ohe_geography_df = pd.DataFrame(ohe_geography.toarray(), columns=onehot_encoder_geo.get_feature_names_out())

        return ohe_geography_df

    @staticmethod
    def X_with_label_encoded_gender(X: pd.DataFrame, label_encoder_gender: LabelEncoder) -> pd.Series:
        label_encoded_gender = label_encoder_gender.transform(X['Gender'])

        return label_encoded_gender

    @staticmethod
    def X_standardized(X: pd.DataFrame, scaler: StandardScaler) -> numpy.ndarray:
        return scaler.transform(X)
