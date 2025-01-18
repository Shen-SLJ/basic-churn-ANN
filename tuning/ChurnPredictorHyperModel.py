import keras_tuner as kt
from keras import Sequential
from keras.src.layers import InputLayer, Dense
from keras_tuner import HyperParameters


class ChurnPredictorHyperModel(kt.HyperModel):
    HP_N_LAYERS_NAME = 'layers'
    HP_NEURONS_PER_LAYER_NAME = 'neurons'
    HP_LOSS_NAME = 'loss'

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
        return hp.Choice(name=ChurnPredictorHyperModel.HP_N_LAYERS_NAME, values=[1, 2])

    @staticmethod
    def __hp_neurons_per_layer(hp: HyperParameters) -> HyperParameters.Choice:
        return hp.Choice(name=ChurnPredictorHyperModel.HP_NEURONS_PER_LAYER_NAME, values=[8, 16, 32, 64])

    @staticmethod
    def __hp_loss(hp: HyperParameters) -> HyperParameters.Choice:
        return hp.Choice(
            name=ChurnPredictorHyperModel.HP_LOSS_NAME,
            values=['binary_crossentropy', 'binary_focal_crossentropy']
        )
