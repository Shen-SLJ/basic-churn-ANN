from tuning import x_train, x_test, y_train, y_test, early_stopping_callback, tuner

MODEL_SAVE_FILEPATH = 'model/model.keras'

if __name__ == '__main__':
    pass
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
