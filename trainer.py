import datetime

from core.ChurnDatasetPreprocessor import ChurnDatasetPreprocessor
from keras import Sequential
from keras.src.callbacks import TensorBoard, EarlyStopping
from keras.src.layers import Dense
from keras.src.losses import BinaryCrossentropy
from keras.src.optimizers import Adam

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
