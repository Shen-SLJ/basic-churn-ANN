import numpy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


class ChurnDataPreprocessingUtils:
    @staticmethod
    def df_with_ohe_geography(df: pd.DataFrame, onehot_encoder_geo: OneHotEncoder) -> pd.DataFrame:
        ohe_geography_df = ChurnDataPreprocessingUtils.__ohe_geography_df(df[['Geography']], onehot_encoder_geo)

        df.drop('Geography', axis=1, inplace=True)
        df_geo_ohe = pd.concat([df, ohe_geography_df], axis=1)

        return df_geo_ohe

    @staticmethod
    def __ohe_geography_df(geo_df: pd.DataFrame, onehot_encoder_geo: OneHotEncoder) -> pd.DataFrame:
        ohe_geography = onehot_encoder_geo.transform(geo_df)
        ohe_geography_df = pd.DataFrame(ohe_geography.toarray(), columns=onehot_encoder_geo.get_feature_names_out())

        return ohe_geography_df

    @staticmethod
    def label_encoded_gender_series_from_df(df: pd.DataFrame, label_encoder_gender: LabelEncoder) -> pd.Series:
        label_encoded_gender = label_encoder_gender.transform(df['Gender'])

        return label_encoded_gender

    @staticmethod
    def df_standardized(df: pd.DataFrame, scaler: StandardScaler) -> numpy.ndarray:
        return scaler.transform(df)
