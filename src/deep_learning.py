import random

import keras_tuner as kt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

import model_base as mb


def plot_actual_vs_predicted(actual, pred, name='Name', title='Actual vs Predicted Values', xlabel='Time',
                             ylabel='Values'):
    """
    Plots actual vs. predicted values for comparison.

    Args:
        actual (Series or array-like): The actual values to plot.
        pred (Series or array-like): The predicted values to plot.
        name (str): Context name for the plot (e.g., dataset or model name).
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    plt.figure(figsize=(15, 8))
    plt.plot(actual, color='blue', marker='o', label='Actual', linestyle='-', linewidth=1)
    plt.plot(pred, color='red', marker='x', label='Predicted', linestyle='None')
    plt.title(f'{name}-{title}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def set_index_to_datetime(df, frequency='H', datetime_column_name='Start', datetime_format='%Y-%m-%d %H:%M:%S'):
    """
    Set the DataFrame's index to a datetime column with a specific format.

    Parameters:
        df (DataFrame): The DataFrame for which to set the index.
        datetime_column_name (str): The name of the datetime column.
        datetime_format (str): The format of the datetime column (default is '%Y-%m-%d %H:%M:%S').

    Returns:
        DataFrame: The DataFrame with the updated index.
    """
    if frequency != 'H':
        datetime_format = '%Y-%m-%d'
    df.index = pd.to_datetime(df[datetime_column_name], format=datetime_format)
    return df


def add_time_features(df, datetime_column_name='Seconds', frequency='H'):
    """
    Add daily, yearly, hourly sine and cosine features to a DataFrame
    based on a given datetime column.

    Parameters:
        df (DataFrame): The DataFrame to which features will be added.
        datetime_column_name (str): The name of the datetime column in the DataFrame.

    Returns:
        DataFrame: The DataFrame with added time-related features.
    """
    timestamp_s = df[datetime_column_name]

    # Define constants
    second = 1
    minute = 60 * second
    hour = 60 * minute
    day = 24 * hour
    week = 7 * day
    year = (365.2425) * day
    month = (30.44) * day  # Average number of days in a month

    # Add daily, yearly, hourly, monthly, and weekly sine and cosine features
    if frequency == 'H':
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Hour sin'] = np.sin(timestamp_s * (2 * np.pi / hour))
        df['Hour cos'] = np.cos(timestamp_s * (2 * np.pi / hour))
        df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
        df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    elif frequency == 'D':
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
        df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    elif frequency == 'W':
        df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
        df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
        df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
        df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    elif frequency == 'M':
        df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
        df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


def preprocess_time_series(df, frequency='H',
                           columns=['PM2.5-Value', 'NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']):
    """
    Preprocess a DataFrame with time series data.

    Parameters:
        df (DataFrame): The DataFrame containing the time series data.
        columns (list): List of columns to select from the DataFrame.

    Returns:
        DataFrame: The preprocessed DataFrame with added time-related features.
    """

    df_ts = df[columns]
    df_ts['Seconds'] = df_ts.index.map(pd.Timestamp.timestamp)
    df_ts = add_time_features(df_ts, 'Seconds', frequency)
    df_ts = df_ts.drop('Seconds', axis=1)

    return df_ts


def create_windows(data, window_size=24):
    """
    Creates overlapping windows from the input data.

    Args:
        data (np.ndarray): The input dataset, expected to be a NumPy array.
        window_size (int): The size of each window to create.

    Returns:
        tuple: A tuple containing two NumPy arrays, `X` and `y`. `X` contains the input features
               for each window, and `y` contains the corresponding target value for each window.
    """
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:(i + window_size), :])
        y.append(data[i + window_size, :])
    return np.array(X), np.array(y)


def preprocess_and_scale_data(df, window_size=24, train_ratio=0.6, val_ratio=0.2):
    """
    Scales the input data and splits it into training, validation, and testing sets based on provided ratios.

    Args:
        df (np.ndarray): The input dataset to preprocess.
        window_size (int): The size of the window to use for creating sequences.
        train_ratio (float): The ratio of the dataset to allocate for training.
        val_ratio (float): The ratio of the dataset to allocate for validation.

    Returns:
        tuple: Tuple containing scaled and windowed training, validation, and testing feature sets (X_train, X_val, X_test),
               corresponding target sets (y_train, y_val, y_test), and the scaler object used for inverse transformations.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = create_windows(scaled_data, window_size=window_size)

    train_size = int(len(X) * train_ratio)  # 60% of the data for training
    validation_size = int(len(X) * val_ratio)  # 20% of the data for validation
    test_size = len(X) - train_size - validation_size  # Remaining 20% for testing

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + validation_size], y[train_size:train_size + validation_size]
    X_test, y_test = X[train_size + validation_size:], y[train_size + validation_size:]

    preprocessing_summary = {
        "Total Data Points": len(df),
        "Training Data Size": len(X_train),
        "Validation Data Size": len(X_val),
        "Testing Data Size": len(X_test)
    }

    print(preprocessing_summary)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def load_data(df, frequency='H'):
    window_size = {'H': 24, 'D': 24, 'W': 24, 'M': 12}
    df = set_index_to_datetime(df, frequency)
    df = preprocess_time_series(df, frequency)
    return preprocess_and_scale_data(df, window_size=window_size[frequency])


def get_dnn_best_params(frequency='H'):
    """
    Returns the best parameters for DNN based on the specified frequency.

    Args:
    - frequency (str): The frequency for which to get the best parameters. Options are 'H', 'D', 'W', 'M'.

    Returns:
    - dict: A dictionary of the best parameters.
    """
    # Define best parameters for each frequency
    best_params = {
        'H': {
            'learning_rate': 0.00014014488528467923,
            'num_layers': 3,
            'units': [448, 256, 160],
            'activations': ['tanh', 'tanh', 'tanh', 'tanh', 'tanh'],
            'dropout': True,
        },
        'D': {
            'learning_rate': 0.0005906296261520694,
            'num_layers': 5,
            'units': [224, 32, 96, 32, 160],
            'activations': ['tanh', 'tanh', 'tanh', 'tanh', 'tanh'],
            'dropout': True,
        },
        'W': {
            'learning_rate': 0.0004077726925804003,
            'num_layers': 4,
            'units': [352, 32, 32, 32],
            'activations': ['tanh', 'tanh', 'tanh', 'tanh'],
            'dropout': False,
        },
        'M': {
            'learning_rate': 0.0040835753251513085,
            'num_layers': 2,
            'units': [32, 32],
            'activations': ['tanh', 'tanh'],
            'dropout': False,
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


## DNN
def build_and_tune_dnn_model(X_train, y_train, X_val, y_val, max_trials=5, num_epochs=10):
    """
    Build and tune an DNN model using Keras Tuner.

    Parameters:
        X_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation data.
        y_val (numpy.ndarray): Validation labels.
        max_trials (int): Maximum number of hyperparameter tuning trials.
        num_epochs (int): Number of training epochs per trial.

    Returns:
        tensorflow.keras.models.Sequential: The best-tuned LSTM model.
    """
    shape = y_train.shape[1]

    def build_dnn_model(hp):
        model = keras.Sequential()
        model.add(layers.Flatten())
        for i in range(hp.Int("num_layers", 1, 5)):
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))
        model.add(layers.Dense(shape, activation="linear"))
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        return model

    tuner = kt.BayesianOptimization(
        hypermodel=build_dnn_model,
        objective=kt.Objective('val_root_mean_squared_error', direction='min'),
        max_trials=max_trials,
        directory="my_dir",
        project_name=f"dnn_tuning_{random.randint(1, 100)}",
    )

    tuner.search(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]
    # best_model.summary()
    tuner.results_summary()
    return best_model, best_hp


def build_best_dnn_model(y_train, learning_rate=0.0004489034857354316, num_layers=3, units=[320, 32, 32],
                         activations=['relu', 'relu', 'relu'], dropout=False):
    """
    Builds a sequential DNN model based on specified parameters, including optional dropout.

    Args:
        y_train (np.ndarray): The training dataset target values, used to set the output layer size.
        learning_rate (float): The learning rate for the Adam optimizer.
        num_layers (int): The number of dense layers to include in the model.
        units (list of int): The number of neurons in each dense layer.
        activations (list of str): The activation functions for each dense layer.
        dropout (bool): Whether to include a dropout layer.
        dropout_rate (float): The rate of dropout, if dropout is enabled.

    Returns:
        A compiled TensorFlow Keras model.
    """
    model = Sequential()
    model.add(layers.Flatten())
    for i in range(num_layers):
        model.add(layers.Dense(units[i], activations[i]))
    if dropout:
        model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(y_train.shape[1], 'linear'))

    optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanAbsoluteError()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def dnn_train_and_evaluate(df, frequency='H'):
    """
    Trains and evaluates a DNN model based on the specified dataset and frequency.

    Args:
        df (DataFrame): The input dataset.
        frequency (str): The frequency of the dataset, influencing model training and evaluation.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)

    best_params = get_dnn_best_params(frequency)
    print(best_params)

    model = build_best_dnn_model(y_train, learning_rate=best_params['learning_rate'],
                                 num_layers=best_params['num_layers'],
                                 units=best_params['units'],
                                 activations=best_params['activations'],
                                 dropout=best_params['dropout'])

    cp = ModelCheckpoint(f'dnn_model_{frequency}/', save_best_only=True)

    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss for a regression problem
        patience=10,  # Number of epochs with no improvement
        verbose=1,
        restore_best_weights=True
    )
    callbacks = [cp, early_stopping] if frequency == 'H' else [cp]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=callbacks)

    model = load_model(f'dnn_model_{frequency}/')

    print(model.summary())

    # Validation
    val_predictions = model.predict(X_val)

    # Converting predictions back to original scale
    val_predicted = scaler.inverse_transform(val_predictions)
    y_val_inverse = scaler.inverse_transform(y_val)

    val_actual = y_val_inverse[:, 0]
    val_pred = val_predicted[:, 0]
    plot_actual_vs_predicted(val_actual, val_pred, "Validation")

    # Error Metric for Validation
    mb.evolve_error_metrics(val_pred, val_actual)
    mb.naive_mean_absolute_scaled_error(val_pred, val_actual)

    # Test
    test_predictions = model.predict(X_test)

    # Converting predictions back to original scale
    test_predicted = scaler.inverse_transform(test_predictions)
    y_test_inverse = scaler.inverse_transform(y_test)

    test_actual = y_test_inverse[:, 0]
    test_pred = test_predicted[:, 0]
    plot_actual_vs_predicted(test_actual, test_pred, "Test")

    # Error Metric for Validation
    mb.evolve_error_metrics(test_pred, test_actual)
    mb.naive_mean_absolute_scaled_error(test_pred, test_actual)

    mb.save_model_to_pickle(model, f'dnn_model_{frequency}.pkl')


def dnn_tune_and_evolve(df, frequency='H'):
    """
    Loads the dataset, tunes a DNN model based on the training data, and evaluates it on the validation set.

    Args:
        df (DataFrame): The dataset to be used for training and evaluation.
        frequency (str): The sampling frequency of the data, used to tailor the data loading process.

    Returns:
        tuple: A tuple containing the best DNN model and its hyperparameters after tuning.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)

    best_model, best_hp = build_and_tune_dnn_model(X_train, y_train, X_val, y_val)

    return best_model, best_hp


## LSTM

def get_lstm_best_params(frequency):
    """
    Returns the best parameters for LSTM based on the specified frequency.

    Args:
    - frequency (str): The frequency for which to get the best parameters. Options are 'H', 'D', 'W', 'M'.

    Returns:
    - dict: A dictionary of the best parameters.
    """
    # Define best parameters for each frequency
    best_params = {
        'H': {
            'learning_rate': 0.00019153080222192724,
            'num_layers': 2,
            'units': [448, 224],
            'activations': ['tanh', 'tanh'],
            'dropout': False,
        },
        'D': {
            'learning_rate': 0.0028652697828724623,
            'num_layers': 3,
            'units': [480, 384, 288],
            'activations': ['tanh', 'tanh', 'tanh'],
            'dropout': True,
        },
        'W': {
            'learning_rate': 0.0009513380195031257,
            'num_layers': 6,
            'units': [480, 160, 64, 224, 512, 384],
            'activations': ['relu', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh'],
            'dropout': False,
        },
        'M': {
            'learning_rate': 0.0013383290456424192,
            'num_layers': 5,
            'units': [288, 352, 480, 224, 192],
            'activations': ['relu', 'tanh', 'tanh', 'tanh', 'tanh'],
            'dropout': False,
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def build_and_tune_lstm_model(X_train, y_train, X_val, y_val, max_trials=5, num_epochs=10):
    """
    Build and tune an LSTM model using Keras Tuner.

    Parameters:
        X_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation data.
        y_val (numpy.ndarray): Validation labels.
        max_trials (int): Maximum number of hyperparameter tuning trials.
        num_epochs (int): Number of training epochs per trial.

    Returns:
        tensorflow.keras.models.Sequential: The best-tuned LSTM model.
    """
    shape = y_train.shape[1]

    def build_lstm_model(hp):
        model = keras.Sequential()
        model.add(layers.InputLayer((X_train.shape[1], X_train.shape[2])))
        model.add(
            layers.LSTM(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation_lstm", ["relu", "tanh"]),
            )
        )

        for i in range(hp.Int("num_layers", 1, 5)):
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )

        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))

        model.add(layers.Dense(shape, activation="linear"))
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        return model

    tuner = kt.BayesianOptimization(
        hypermodel=build_lstm_model,
        objective=kt.Objective('val_root_mean_squared_error', direction='min'),
        max_trials=max_trials,
        directory="my_dir",
        project_name=f"lstm_tuning_{random.randint(1, 100)}",
    )

    tuner.search(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]
    best_model.summary()
    tuner.results_summary()
    return best_model, best_hp


def build_best_lstm_model(X_train, y_train, learning_rate=0.0004489034857354316, num_layers=3, units=[320, 32, 32],
                          activations=['relu', 'relu', 'relu'], dropout=False):
    """
    Builds a sequential LSTM model based on specified parameters, including optional dropout.

    Args:
        y_train (np.ndarray): The training dataset target values, used to set the output layer size.
        learning_rate (float): The learning rate for the Adam optimizer.
        num_layers (int): The number of dense layers to include in the model.
        units (list of int): The number of neurons in each dense layer.
        activations (list of str): The activation functions for each dense layer.
        dropout (bool): Whether to include a dropout layer.
        dropout_rate (float): The rate of dropout, if dropout is enabled.

    Returns:
        A compiled TensorFlow Keras model.
    """
    model = Sequential()
    model.add(layers.InputLayer((X_train.shape[1], X_train.shape[2])))
    model.add(layers.LSTM(units[0], activations[0]))

    for i in range(1, num_layers):
        model.add(layers.Dense(units[i], activations[i]))

    if dropout:
        model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(y_train.shape[1], 'linear'))

    optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanAbsoluteError()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def lstm_train_and_evaluate(df, frequency='H'):
    """
    Trains and evaluates a LSTM model based on the specified dataset and frequency.

    Args:
        df (DataFrame): The input dataset.
        frequency (str): The frequency of the dataset, influencing model training and evaluation.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)

    best_params = get_lstm_best_params(frequency)
    print(best_params)

    model = build_best_lstm_model(X_train, y_train, learning_rate=best_params['learning_rate'],
                                  num_layers=best_params['num_layers'],
                                  units=best_params['units'],
                                  activations=best_params['activations'],
                                  dropout=best_params['dropout'])

    cp = ModelCheckpoint(f'lstm_model_{frequency}/', save_best_only=True)
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss for a regression problem
        patience=10,  # Number of epochs with no improvement
        verbose=1,
        restore_best_weights=True
    )
    callbacks = [cp, early_stopping] if frequency == 'H' else [cp]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=callbacks)

    model = load_model(f'lstm_model_{frequency}/')

    # Validation
    val_predictions = model.predict(X_val)

    # Converting predictions back to original scale
    val_predicted = scaler.inverse_transform(val_predictions)
    y_val_inverse = scaler.inverse_transform(y_val)

    val_actual = y_val_inverse[:, 0]
    val_pred = val_predicted[:, 0]
    plot_actual_vs_predicted(val_actual, val_pred, "Validation")

    # Error Metric for Validation
    mb.evolve_error_metrics(val_pred, val_actual)
    mb.naive_mean_absolute_scaled_error(val_pred, val_actual)

    # Test
    test_predictions = model.predict(X_test)

    # Converting predictions back to original scale
    test_predicted = scaler.inverse_transform(test_predictions)
    y_test_inverse = scaler.inverse_transform(y_test)

    test_actual = y_test_inverse[:, 0]
    test_pred = test_predicted[:, 0]
    plot_actual_vs_predicted(test_actual, test_pred, "Test")

    # Error Metric for Validation
    mb.evolve_error_metrics(test_pred, test_actual)
    mb.naive_mean_absolute_scaled_error(test_pred, test_actual)

    mb.save_model_to_pickle(model, f'lstm_model_{frequency}.pkl')


def lstm_tune_and_evolve(df, frequency='H'):
    """
    Loads the dataset, tunes a LSTM model based on the training data, and evaluates it on the validation set.

    Args:
        df (DataFrame): The dataset to be used for training and evaluation.
        frequency (str): The sampling frequency of the data, used to tailor the data loading process.

    Returns:
        tuple: A tuple containing the best LSTM model and its hyperparameters after tuning.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)

    best_model, best_hp = build_and_tune_lstm_model(X_train, y_train, X_val, y_val)

    return best_model, best_hp


## CNN


def get_cnn_best_params(frequency):
    """
    Returns the best parameters for CNN based on the specified frequency.

    Args:
    - frequency (str): The frequency for which to get the best parameters. Options are 'H', 'D', 'W', 'M'.

    Returns:
    - dict: A dictionary of the best parameters.
    """
    # Define best parameters for each frequency
    best_params = {
        'H': {
            'learning_rate': 0.00021997838024224393,
            'num_layers': 2,
            'units': [224, 256],
            'activations': ['relu', 'tanh'],
            'dropout': False,
        },
        'D': {
            'learning_rate': 0.0012918827423762096,
            'num_layers': 5,
            'units': [96, 448, 96, 512, 320],
            'activations': ['tanh', 'tanh', 'tanh', 'tanh', 'tanh'],
            'dropout': True,
        },
        'W': {
            'learning_rate': 0.0031087381681937547,
            'num_layers': 4,
            'units': [96, 32, 96, 96],
            'activations': ['tanh', 'tanh', 'tanh', 'tanh'],
            'dropout': True,
        },
        'M': {
            'learning_rate': 0.0009974518412524185,
            'num_layers': 5,
            'units': [64, 448, 256, 64, 416, ],
            'activations': ['relu', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh'],
            'dropout': True,
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def build_and_tune_cnn_model(X_train, y_train, X_val, y_val, max_trials=5, num_epochs=10):
    """
    Build and tune an CNN model using Keras Tuner.

    Parameters:
        X_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation data.
        y_val (numpy.ndarray): Validation labels.
        max_trials (int): Maximum number of hyperparameter tuning trials.
        num_epochs (int): Number of training epochs per trial.

    Returns:
        tensorflow.keras.models.Sequential: The best-tuned LSTM model.
    """
    shape = y_train.shape[1]

    def build_cnn_model(hp):
        model = keras.Sequential()
        model.add(layers.InputLayer((X_train.shape[1], X_train.shape[2])))
        model.add(
            layers.Conv1D(
                filters=hp.Int("filters", min_value=32, max_value=512, step=32),
                kernel_size=2,
                activation=hp.Choice("activation_cnn", ["relu", "tanh"]),
            )
        )
        model.add(layers.Flatten())

        for i in range(hp.Int("num_layers", 1, 5)):
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )

        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))

        model.add(layers.Dense(shape, activation="linear"))
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        return model

    tuner = kt.BayesianOptimization(
        hypermodel=build_cnn_model,
        objective=kt.Objective('val_root_mean_squared_error', direction='min'),
        max_trials=max_trials,
        directory="my_dir",
        project_name=f"cnn_tuning_{random.randint(1, 100)}",
    )

    tuner.search(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]
    best_model.summary()
    tuner.results_summary()
    return best_model, best_hp


def build_best_cnn_model(X_train, y_train, learning_rate=0.0004489034857354316, num_layers=3, units=[64, 32, 32],
                         activations=['relu', 'relu', 'relu', 'relu'], dropout=False):
    """
    Builds a sequential CNN model based on specified parameters, including optional dropout.

    Args:
        y_train (np.ndarray): The training dataset target values, used to set the output layer size.
        learning_rate (float): The learning rate for the Adam optimizer.
        num_layers (int): The number of dense layers to include in the model.
        units (list of int): The number of neurons in each dense layer.
        activations (list of str): The activation functions for each dense layer.
        dropout (bool): Whether to include a dropout layer.
        dropout_rate (float): The rate of dropout, if dropout is enabled.

    Returns:
        A compiled TensorFlow Keras model.
    """
    shape = y_train.shape[1]
    model = Sequential()
    model.add(layers.InputLayer((X_train.shape[1], X_train.shape[2])))
    model.add(layers.Conv1D(units[0], kernel_size=2, activation=activations[0]))
    model.add(layers.Flatten())

    for i in range(1, num_layers):
        model.add(layers.Dense(units[i], activations[i]))
    if dropout:
        model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(shape, 'linear'))

    optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanAbsoluteError()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def cnn_train_and_evaluate(df, frequency='H'):
    """
    Trains and evaluates a CNN model based on the specified dataset and frequency.

    Args:
        df (DataFrame): The input dataset.
        frequency (str): The frequency of the dataset, influencing model training and evaluation.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)

    best_params = get_cnn_best_params(frequency)
    print(best_params)

    model = build_best_cnn_model(X_train, y_train,
                                 learning_rate=best_params['learning_rate'],
                                 num_layers=best_params['num_layers'],
                                 units=best_params['units'],
                                 activations=best_params['activations'],
                                 dropout=best_params['dropout'])

    cp = ModelCheckpoint(f'cnn_model_{frequency}/', save_best_only=True)
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss for a regression problem
        patience=10,  # Number of epochs with no improvement
        verbose=1,
        restore_best_weights=True
    )
    callbacks = [cp, early_stopping] if frequency == 'H' else [cp]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=callbacks)

    model = load_model(f'cnn_model_{frequency}/')

    # Validation
    val_predictions = model.predict(X_val)

    # Converting predictions back to original scale
    val_predicted = scaler.inverse_transform(val_predictions)
    y_val_inverse = scaler.inverse_transform(y_val)

    val_actual = y_val_inverse[:, 0]
    val_pred = val_predicted[:, 0]
    plot_actual_vs_predicted(val_actual, val_pred, "Validation")

    # Error Metric for Validation
    mb.evolve_error_metrics(val_pred, val_actual)
    mb.naive_mean_absolute_scaled_error(val_pred, val_actual)

    # Test
    test_predictions = model.predict(X_test)

    # Converting predictions back to original scale
    test_predicted = scaler.inverse_transform(test_predictions)
    y_test_inverse = scaler.inverse_transform(y_test)

    test_actual = y_test_inverse[:, 0]
    test_pred = test_predicted[:, 0]
    plot_actual_vs_predicted(test_actual, test_pred, "Test")

    # Error Metric for Validation
    mb.evolve_error_metrics(test_pred, test_actual)
    mb.naive_mean_absolute_scaled_error(test_pred, test_actual)

    mb.save_model_to_pickle(model, f'cnn_model_{frequency}.pkl')


def cnn_tune_and_evolve(df, frequency='H'):
    """
    Loads the dataset, tunes a CNN model based on the training data, and evaluates it on the validation set.

    Args:
        df (DataFrame): The dataset to be used for training and evaluation.
        frequency (str): The sampling frequency of the data, used to tailor the data loading process.

    Returns:
        tuple: A tuple containing the best CNN model and its hyperparameters after tuning.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)
    best_model, best_hp = build_and_tune_cnn_model(X_train, y_train, X_val, y_val)
    return best_model, best_hp
