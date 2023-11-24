from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers.legacy import Adam
from datetime import datetime

import keras_tuner
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import model_base as mb


#
# def create_model(X_train, learning_rate=0.001, n_layers=2, n_nodes=64, dropout_rate=0.1):
#     model = Sequential()
#     for i in range(n_layers):
#         if i == 0:
#             model.add(Dense(n_nodes, input_dim=X_train.shape[1], activation='relu'))
#         else:
#             model.add(Dense(n_nodes, activation='relu'))
#         model.add(Dropout(dropout_rate))
#     model.add(Dense(1, activation='linear'))
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='mean_squared_error')
#     return model
#
#
# def ann_tune_and_evaluate(df):
#     n_iter_search = 10
#     random_state = 42
#
#     scaler = MinMaxScaler()
#     # Define your features and target variable
#     train_data, validation_data, test_data = mb.split_data(df)
#     # Extract the features
#     X_train, X_val, X_test = mb.scale_features(train_data, validation_data, test_data, scaler)
#     # Extract the target variable
#     y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)
#
#     print(f'Started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
#
#     # Define the hyperparameter space
#     learning_rates = [0.001, 0.01, 0.1]
#     n_layers_options = [2, 3]
#     n_nodes_options = [32, 64]
#     dropout_rates = [0.1, 0.2]
#
#     best_val_loss = float('inf')
#     best_hyperparams = {}
#
#     # Hyperparameter tuning loop
#     for lr in learning_rates:
#         for n_layers in n_layers_options:
#             for n_nodes in n_nodes_options:
#                 for dropout_rate in dropout_rates:
#                     model = create_model(X_train, lr, n_layers, n_nodes, dropout_rate)
#                     model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1)
#                     val_loss = model.evaluate(X_val, y_val, verbose=1)
#                     if val_loss < best_val_loss:
#                         best_val_loss = val_loss
#                         best_hyperparams = {'learning_rate': lr, 'n_layers': n_layers, 'n_nodes': n_nodes,
#                                             'dropout_rate': dropout_rate}
#
#     print("Best Hyperparameters:")
#     print(best_hyperparams)
#     print(f"Validation Loss with Best Hyperparameters: {best_val_loss}")
#     best_model = create_model(X_val, **best_hyperparams)
#     best_model.fit(X_val, y_val, epochs=100, batch_size=32, verbose=1)
#     test_loss = best_model.evaluate(X_test, y_test, verbose=0)
#     print(f"Test Loss: {test_loss}")
#
#     print(f'Finished {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
#     # Return the best estimator and the MSE scores
#     # return random_search.best_estimator_
#

# Define the model building function
def ann_build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(
            layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(1, activation="linear"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model

def ann_tune_and_get_best_model(df ,epochs=10, max_trials=5, executions_per_trial=2):
    # Split the data
    train_data, validation_data, test_data = mb.split_data(df)

    # Scale the features
    X_train, X_val, X_test = mb.scale_features(train_data, validation_data, test_data, MinMaxScaler())

    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    # Initialize the tuner
    tuner = keras_tuner.RandomSearch(
        hypermodel=ann_build_model,
        objective=keras_tuner.Objective("val_mean_absolute_error", direction="min"),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        overwrite=True,
        directory="my_dir",
        project_name="irp_ann",
    )

    # Perform the hyperparameter search
    tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

    # Get the best model
    best_model = tuner.get_best_models()[0]

    # Print the best model
    print("===== Best Model =====")
    print(best_model)

    # Print the tuner's results summary
    tuner.results_summary()

    return best_model


def ann_train_and_evolve(df):
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    # Normalize the features
    X_train, X_val, X_test = mb.scale_features(train_data, validation_data, test_data, MinMaxScaler())
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    # Define the parameters
    params = {
        'learning_rate': 0.00014992878789921055,  # 0.0037702242108628697,
        'n_layers': 3,
        'n_nodes': 32,
        'dropout_rate': 0.25,
        'epoch': 10
    }

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=2,
                                                      mode='min')

    # Create the ANN model with the specified parameters
    model = Sequential()

    #     # Add the input layer and the first hidden layer
    #     model.add(Dense(params['n_nodes'], input_dim=X_train.shape[1], activation='relu'))
    #     model.add(Dropout(params['dropout_rate']))

    #     # Add the second hidden layer
    #     model.add(Dense(params['n_nodes'], activation='relu'))
    #     model.add(Dropout(params['dropout_rate']))

    #  # Add the second hidden layer
    # model.add(Dense(params['n_nodes'], activation='relu'))
    # model.add(Dropout(params['dropout_rate']))

    # Add the input layer and the first hidden layer
    model.add(Dense(448, input_dim=X_train.shape[1], activation='relu'))
    # model.add(Dropout(params['dropout_rate']))

    # Add the second hidden layer
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(params['dropout_rate']))

    # Add the second hidden layer
    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(params['dropout_rate']))

    # Add the output layer
    model.add(Dense(1, activation='linear'))

    # Compile the model
    optimizer = Adam(learning_rate=params['learning_rate'])
    # optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # ANN model architecture
    # Train the model
    model.fit(X_train, y_train, epochs=params['epoch'], batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=1)
    print(f'Model Loss on Test Data: {loss}')

    # VALIDATION Prediction and Evolution
    y_val_pred = model.predict(X_val)

    print(y_val_pred)

    # Validation Error Metric
    mb.evolve_error_metrics(y_val, y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)

    # TEST Prediction and Evolution
    y_test_pred = model.predict(X_test)

    print(y_test_pred)

    # Test Error Metric
    mb.evolve_error_metrics(y_test, y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)

    # Plot
    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')


FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'
ALL = ['Start', 'NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value', 'PM2.5-Value']


def create_sequences(data, timesteps=1):
    X, y = [], []
    for i in range(len(data) - timesteps):
        # Select the feature columns for the current sequence
        X.append(data.iloc[i:i + timesteps][FEATURES].values)
        # Select the target value for the corresponding sequence
        y.append(data.iloc[i + timesteps][TARGET])
    return np.array(X), np.array(y)


def create_lstm_model(X_train, learning_rate=0.001, units=50, dropout_rate=0.1):
    print(f'X_train: {X_train.shape[1]} and {X_train.shape[2]}')
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model


def lstm_tune_and_evaluate(df2):
    df = df2[ALL]
    print(df.head())
    df['Start'] = pd.to_datetime(df['Start'])
    df.set_index('Start', inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    # Define your features and target variable
    train_data, validation_data, test_data = mb.split_data(df)

    print(f'create_sequences')
    X_train, y_train = create_sequences(train_data)
    X_val, y_val = create_sequences(validation_data)
    X_test, y_test = create_sequences(test_data)

    print(f'Extract the features')

    print(f'Started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Define the hyperparameter space
    learning_rates = [0.001, 0.01, 0.1]
    dropout_rates = [0.1, 0.2, 0.3]

    best_val_loss = float('inf')
    best_hyperparams = {}

    for units in [100, 150]:
        for dropout_rate in dropout_rates:
            for lr in learning_rates:
                # Build the LSTM model
                # model = Sequential()
                # model.add(LSTM(units=units, input_shape=(X_train.shape[1], X_train.shape[2])))
                # model.add(Dropout(dropout_rate))
                # model.add(Dense(1))
                # model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
                model = create_lstm_model(X_train, lr, units, dropout_rate)
                # Fit the model
                history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val),
                                    verbose=1)
                # Evaluate the model
                val_loss = history.history['val_loss'][-1]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_hyperparams = {'units': units, 'dropout_rate': dropout_rate, 'learning_rate': lr}

    print("Best Hyperparameters:")
    print(best_hyperparams)
    print(f"Validation Loss with Best Hyperparameters: {best_val_loss}")
    best_model = create_lstm_model(X_val, **best_hyperparams)
    best_model.fit(X_val, y_val, epochs=100, batch_size=32, verbose=1)
    test_loss = best_model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss}")

    print(f'Finished {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # Return the best estimator and the MSE scores
    # return random_search.best_estimator_

# def lstm_train_and_evolve(df):
#     train_data, validation_data, test_data = mb.split_data(df)
#     # Extract the features
#     # Normalize the features
#     X_train, X_val, X_test =  mb.scale_features(train_data, validation_data, test_data, MinMaxScaler())
#     # Extract the target variable
#     y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)
#
#     # LSTM model architecture
#     model = Sequential()
#     model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1, activation='linear'))
#
#     # Compile the model
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#
#     # Train the model
#     model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1)
#
#     # Evaluate the model
#     loss = model.evaluate(X_test, y_test, verbose=0)
#     print(f'Model Loss on Test Data: {loss}')
#
#     # VALIDATION Prediction and Evolution
#     y_val_pred = model.predict(X_val)
#
#     print(y_val_pred)
#
#     # Validation Error Metric
#     mb.evolve_error_metrics(y_val, y_val_pred)
#     mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)
#
#     # TEST Prediction and Evolution
#     y_test_pred = model.predict(X_test)
#
#     print(y_test_pred)
#
#     # Test Error Metric
#     mb.evolve_error_metrics(y_test, y_test_pred)
#     mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)
#
#     # Plot
#     mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
#     mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')
