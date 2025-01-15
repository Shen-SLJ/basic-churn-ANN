import datetime
from typing import Any

import pandas as pd
from keras import Sequential
from keras.src.callbacks import TensorBoard, EarlyStopping
from keras.src.layers import Dense
from keras.src.losses import BinaryCrossentropy
from keras.src.optimizers import Adam
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle


class ChurnDataPreprocessor:
    y_feature = 'Exited'

    standardizer_dump_filename = 'scalar.pkl'
    label_encoder_gender_dump_filename = 'label_encoder_gender.pkl'
    onehot_encoder_geo_dump_filename = 'onehot_encoder_geo.pkl'

    def __init__(self, dump_preprocessors=True, split_test_size=0.2, split_random_state=42):
        self.__data: DataFrame = pd.read_csv('Churn_Modelling.csv')
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None

        self.__should_dump = dump_preprocessors
        self.__split_test_size = split_test_size
        self.__split_random_state = split_random_state

        self.__standardizer = StandardScaler()
        self.__label_encoder_gender = LabelEncoder()
        self.__one_hot_encoder_geo = OneHotEncoder()

    def run(self) -> None:
        self.__drop_useless_columns()
        self.__replace_gender_with_label_encoding()
        self.__replace_geography_with_ohe()
        self.__create_train_test_split_data()
        self.__standardize_x_data()
        self.__dump_preprocessors_if_should_dump()

    def x_train(self):
        return self.__x_train

    def x_test(self):
        return self.__x_test

    def y_train(self):
        return self.__y_train

    def y_test(self):
        return self.__y_test

    def __dump_preprocessors_if_should_dump(self):
        if self.__should_dump:
            self.__dump_preprocessors()

    def __dump_preprocessors(self) -> None:
        IOUtils.pickle_dump_object(self.__standardizer, self.standardizer_dump_filename)
        IOUtils.pickle_dump_object(self.__label_encoder_gender, self.label_encoder_gender_dump_filename)
        IOUtils.pickle_dump_object(self.__one_hot_encoder_geo, self.onehot_encoder_geo_dump_filename)

    def __create_train_test_split_data(self) -> None:
        x = self.__data.drop(self.y_feature, axis=1)
        y = self.__data[self.y_feature]
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(
            x, y, test_size=self.__split_test_size, random_state=self.__split_random_state
        )

    def __standardize_x_data(self) -> None:
        self.__standardizer.fit(self.__x_train)

        self.__x_train = self.__standardizer.transform(self.__x_train)
        self.__x_test = self.__standardizer.transform(self.__x_test)

    def __drop_useless_columns(self) -> None:
        self.__data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    def __replace_gender_with_label_encoding(self) -> None:
        self.__data['Gender'] = self.__label_encoder_gender.fit_transform(self.__data['Gender'])

    def __geo_ohe_as_dataframe(self) -> pd.DataFrame:
        geo_ohe = self.__one_hot_encoder_geo.fit_transform(self.__data[['Geography']])
        geo_ohe_columns = self.__one_hot_encoder_geo.get_feature_names_out()

        out = pd.DataFrame(data=geo_ohe.toarray(), columns=geo_ohe_columns)

        return out

    def __replace_geography_with_ohe(self) -> None:
        geo_ohe_df = self.__geo_ohe_as_dataframe()

        self.__data.drop(labels=['Geography'], axis=1, inplace=True)
        self.__data = pd.concat([self.__data, geo_ohe_df], axis=1, copy=False)


class IOUtils:
    @staticmethod
    def pickle_dump_object(obj: Any, filename: str) -> None:
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)


if __name__ == '__main__':
    data_preprocessor = ChurnDataPreprocessor()
    data_preprocessor.run()

    x_train = data_preprocessor.x_train()
    x_test = data_preprocessor.x_test()
    y_train = data_preprocessor.y_train()
    y_test = data_preprocessor.y_test()

    # Building the ANN model
    model = Sequential(
        layers=[
            Dense(units=64, activation='relu', input_shape=(x_train.shape[1],)),
            Dense(units=32, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ]
    )
    optimizer = Adam(learning_rate=0.01)
    loss = BinaryCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        callbacks=[tensorboard_callback, early_stopping_callback]
    )

    # Save the model
    model.save('model.keras')
