import datetime
import keras_tuner as kt
from keras_tuner import HyperParameters

from core.ChurnDatasetPreprocessor import ChurnDatasetPreprocessor
from keras import Sequential, Model
from keras.src.callbacks import TensorBoard, EarlyStopping
from keras.src.layers import Dense, InputLayer

MODEL_SAVE_FILEPATH = 'model/model.keras'


class ChurnHyperModel(kt.HyperModel):
    def __init__(self, x_train):
        super().__init__()

        self.__x_train = x_train

    def build(self, hp: HyperParameters) -> Sequential:
        model = Sequential()
        n_layers = self.__hp_n_layers(hp)
        neurons_per_layer = self.__hp_neurons_per_layer(hp)
        loss = self.__hp_loss(hp)

        self.__add_layers_to_model(model=model, n_layers=n_layers, neurons_per_layer=neurons_per_layer)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

        return model

    def __add_layers_to_model(self, model: Sequential, n_layers: int, neurons_per_layer: int) -> None:
        model.add(InputLayer(input_shape=(self.__x_train.shape[1],)))

        for _ in range(n_layers):
            model.add(
                Dense(neurons_per_layer, activation='relu')
            )

        model.add(Dense(1, activation='sigmoid'))

    @staticmethod
    def __hp_n_layers(hp: HyperParameters) -> HyperParameters.Choice:
        return hp.Choice('layers', [1, 2])

    @staticmethod
    def __hp_neurons_per_layer(hp: HyperParameters) -> HyperParameters.Choice:
        return hp.Choice('neurons', [8, 16, 32, 64])

    @staticmethod
    def __hp_loss(hp: HyperParameters) -> HyperParameters.Choice:
        return hp.Choice('loss', ['binary_crossentropy', 'binary_focal_crossentropy'])


if __name__ == '__main__':
    data_preprocessor = ChurnDatasetPreprocessor().run()

    x_train = data_preprocessor.x_train()
    x_test = data_preprocessor.x_test()
    y_train = data_preprocessor.y_train()
    y_test = data_preprocessor.y_test()

    hypermodel = ChurnHyperModel(x_train=x_train)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    objective = kt.Objective(name='val_accuracy', direction='max')
    tuner = kt.GridSearch(hypermodel=hypermodel, objective=objective, project_name='tuning')
    tuner.search(
        x=x_train,
        y=y_train,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping_callback],
    )

    print(f"Best n_layers: {tuner.get_best_hyperparameters()[0].get('layers')}")
    print(f"Best neurons per layer: {tuner.get_best_hyperparameters()[0].get('neurons')}")
    print(f"Best loss function: {tuner.get_best_hyperparameters()[0].get('loss')}")

    # Callbacks
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # # Train the model
    # history = model.fit(
    #     x=x_train,
    #     y=y_train,
    #     validation_data=(x_test, y_test),
    #     epochs=100,
    #     callbacks=[tensorboard_callback, early_stopping_callback]
    # )
    #
    # # Save the model
    # model.save(MODEL_SAVE_FILEPATH)
