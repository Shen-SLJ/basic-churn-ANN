import datetime
import pandas as pd

from utils.ChurnDataPreprocessingUtils import ChurnDataPreprocessingUtils
from utils.IOUtils import IOUtils
from keras import Sequential
from keras.src.callbacks import TensorBoard, EarlyStopping
from keras.src.layers import Dense
from keras.src.losses import BinaryCrossentropy
from keras.src.optimizers import Adam
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


class ChurnDatasetPreprocessor:
    y_feature = 'Exited'

    churn_dataset_filename = 'Churn_Modelling.csv'
    standardizer_dump_filename = 'scaler.pkl'
    label_encoder_gender_dump_filename = 'label_encoder_gender.pkl'
    onehot_encoder_geo_dump_filename = 'onehot_encoder_geo.pkl'

    def __init__(self, dump_preprocessors=True, split_test_size=0.2, split_random_state=42):
        self.__data: DataFrame = pd.read_csv(self.churn_dataset_filename)
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
        self.__fit_encoders()
        self.__convert_categorical_vals_to_numerical()
        self.__create_train_test_split_data()
        self.__fit_standardizer()
        self.__standardize_train_test_data()
        self.__dump_preprocessors_if_should_dump()

    def x_train(self):
        return self.__x_train

    def x_test(self):
        return self.__x_test

    def y_train(self):
        return self.__y_train

    def y_test(self):
        return self.__y_test

    def __drop_useless_columns(self) -> None:
        self.__data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    def __fit_encoders(self):
        self.__one_hot_encoder_geo.fit(self.__data[['Geography']])
        self.__label_encoder_gender.fit(self.__data['Gender'])

    def __fit_standardizer(self):
        self.__standardizer.fit(self.__x_train)

    def __convert_categorical_vals_to_numerical(self):
        self.__data = ChurnDataPreprocessingUtils.df_with_ohe_geography(self.__data, self.__one_hot_encoder_geo)
        self.__data['Gender'] = ChurnDataPreprocessingUtils.label_encoded_gender_series_from_df(self.__data,
                                                                                                self.__label_encoder_gender)

    def __standardize_train_test_data(self):
        self.__x_train = ChurnDataPreprocessingUtils.df_standardized(self.__x_train, self.__standardizer)
        self.__x_test = ChurnDataPreprocessingUtils.df_standardized(self.__x_test, self.__standardizer)

    def __create_train_test_split_data(self) -> None:
        x = self.__data.drop(self.y_feature, axis=1)
        y = self.__data[self.y_feature]
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(
            x, y, test_size=self.__split_test_size, random_state=self.__split_random_state
        )

    def __dump_preprocessors_if_should_dump(self):
        if self.__should_dump:
            self.__dump_preprocessors()

    def __dump_preprocessors(self) -> None:
        IOUtils.pickle_dump_object(self.__standardizer, self.standardizer_dump_filename)
        IOUtils.pickle_dump_object(self.__label_encoder_gender, self.label_encoder_gender_dump_filename)
        IOUtils.pickle_dump_object(self.__one_hot_encoder_geo, self.onehot_encoder_geo_dump_filename)


if __name__ == '__main__':
    data_preprocessor = ChurnDatasetPreprocessor()
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
