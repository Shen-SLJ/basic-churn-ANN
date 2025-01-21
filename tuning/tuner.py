import keras_tuner as kt
from keras.src.callbacks import EarlyStopping
from keras_tuner import Tuner

from core.ChurnDatasetPreprocessor import ChurnDatasetPreprocessor
from tuning.ChurnPredictorHyperModel import ChurnPredictorHyperModel


def __print_best_hyperparameter(tuner: Tuner, hp_name: str) -> None:
    best_hyperparam = tuner.get_best_hyperparameters()[0].get(hp_name)

    print(f"Best value for hp {hp_name} = {best_hyperparam}")


if __name__ == '__main__':
    data_preprocessor = ChurnDatasetPreprocessor().run()

    x_train = data_preprocessor.x_train()
    x_test = data_preprocessor.x_test()
    y_train = data_preprocessor.y_train()
    y_test = data_preprocessor.y_test()

    hypermodel = ChurnPredictorHyperModel(x_train=x_train)
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    objective = kt.Objective(name='val_accuracy', direction='max')
    tuner = kt.GridSearch(hypermodel=hypermodel, objective=objective, project_name='data')

    tuner.search(
        x=x_train,
        y=y_train,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping_callback],
    )

    __print_best_hyperparameter(tuner=tuner, hp_name=ChurnPredictorHyperModel.HP_NEURONS_PER_LAYER_NAME)
    __print_best_hyperparameter(tuner=tuner, hp_name=ChurnPredictorHyperModel.HP_N_LAYERS_NAME)
    __print_best_hyperparameter(tuner=tuner, hp_name=ChurnPredictorHyperModel.HP_LOSS_NAME)
