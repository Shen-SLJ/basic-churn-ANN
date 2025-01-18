# Trainer training on the best parameters found via hyper-tuning

import datetime

from keras import Sequential
from keras.src.callbacks import TensorBoard, EarlyStopping
from keras.src.layers import Dense

from core.ChurnDatasetPreprocessor import ChurnDatasetPreprocessor

MODEL_SAVE_FILEPATH = 'model/model.keras'
LOSS_FN_NAME = 'binary_crossentropy'
OPTIMIZER_NAME = 'adam'

if __name__ == '__main__':
    data_preprocessor = ChurnDatasetPreprocessor().run()

    x_train = data_preprocessor.x_train()
    x_test = data_preprocessor.x_test()
    y_train = data_preprocessor.y_train()
    y_test = data_preprocessor.y_test()

    # Create model
    model = Sequential([
        Dense(units=64, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(loss=LOSS_FN_NAME, optimizer=OPTIMIZER_NAME, metrics=['accuracy'])

    # Callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        callbacks=[tensorboard_callback, early_stopping_callback]
    )

    # Save the model
    model.save(MODEL_SAVE_FILEPATH)
