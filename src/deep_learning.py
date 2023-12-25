import random

import keras_tuner
import pandas as pd
from kerastuner.tuners import RandomSearch, BayesianOptimization
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

import model_base as mb


def plot_actual_vs_predicted(actual, pred, name='Name', title='Actual vs Predicted Values', xlabel='Time',
                             ylabel='Values'):
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


def df_to_X_y(df, window_size=24):
    df_values = df.values
    X = [df_values[i:i + window_size] for i in range(len(df_values) - window_size)]
    y = df_values[window_size:, 0]
    print(len(np.array(X)))
    return np.array(X), np.array(y)


def df_to_X_y3(df, window_size=7):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)
        label = [df_as_np[i + window_size][0], df_as_np[i + window_size][1]]
        y.append(label)
    return np.array(X), np.array(y)


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


def preprocess_and_normalize_data(df, window_size=24, train_ratio=0.6, val_ratio=0.2):
    """
    Preprocess and normalize time series data.

    Parameters:
        df (DataFrame): The DataFrame containing the time series data.
        window_size (int): The window size for creating input sequences.
        train_ratio (float): The ratio of data used for training.
        val_ratio (float): The ratio of data used for validation.

    Returns:
        numpy.ndarray: Normalized training data.
        numpy.ndarray: Normalized validation data.
        numpy.ndarray: Normalized testing data.
        numpy.ndarray: Corresponding training labels.
        numpy.ndarray: Corresponding validation labels.
        numpy.ndarray: Corresponding testing labels.
    """
    print(window_size)

    X, y = df_to_X_y(df, window_size=window_size)

    n = len(X)
    X_train, y_train = X[:int(n * train_ratio)], y[:int(n * train_ratio)]
    X_val, y_val = X[int(n * train_ratio):int(n * (train_ratio + val_ratio))], y[int(n * train_ratio):int(
        n * (train_ratio + val_ratio))]
    X_test, y_test = X[int(n * (train_ratio + val_ratio)):], y[int(n * (train_ratio + val_ratio)):]

    X_train_mean = X_train.mean()
    X_train_std = X_train.std()

    X_train_norm = (X_train - X_train_mean) / X_train_std
    X_val_norm = (X_val - X_train_mean) / X_train_std
    X_test_norm = (X_test - X_train_mean) / X_train_std

    return X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test


def create_windows(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:(i + window_size), :])
        y.append(data[i + window_size, :])
    return np.array(X), np.array(y)


def preprocess_and_scale_data(df, window_size=24, train_ratio=0.6, val_ratio=0.2):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = create_windows(scaled_data, window_size=window_size)

    train_size = int(len(X) * train_ratio)  # 60% of the data for training
    validation_size = int(len(X) * val_ratio)  # 20% of the data for validation
    test_size = len(X) - train_size - validation_size  # Remaining 20% for testing

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + validation_size], y[train_size:train_size + validation_size]
    X_test, y_test = X[train_size + validation_size:], y[train_size + validation_size:]

    # Now X_train, y_train are for training; X_val, y_val are for validation; X_test, y_test are for testing

    preprocessing_summary = {
        "Total Data Points": len(df),
        "Training Data Size": len(X_train),
        "Validation Data Size": len(X_val),
        "Testing Data Size": len(X_test)
    }

    print(preprocessing_summary)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# def load_data(df, frequency='H'):
#     window_size = {'H': 24, 'D': 30, 'W': 24, 'M': 6}
#     df = set_index_to_datetime(df, frequency)
#     df = preprocess_time_series(df, frequency)
#     return preprocess_and_normalize_data(df, window_size=window_size[frequency])

def load_data(df, frequency='H'):
    window_size = {'H': 24, 'D': 30, 'W': 52, 'M': 12}
    df = set_index_to_datetime(df, frequency)
    df = preprocess_time_series(df, frequency)
    return preprocess_and_scale_data(df, window_size=window_size[frequency])


def get_ann_best_params(frequency='H'):
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
            'learning_rate': 0.00033663431603295945,
            'num_layers': 2,
            'units': [288, 32],
            'activations': ['relu', 'relu'],
            'dropout': True,
        },
        'D': {
            'learning_rate': 0.006415517608465564,
            'num_layers': 4,
            'units': [352, 256, 256, 32],
            'activations': ['relu', 'relu', 'relu', 'relu'],
            'dropout': False,
        },
        'W': {
            'learning_rate': 0.004533188169061783,
            'num_layers': 3,
            'units': [224, 224, 256],
            'activations': ['relu', 'relu', 'relu'],
            'dropout': False,
        },
        'M': {
            'learning_rate': 0.000889004387467539,
            'num_layers': 5,
            'units': [288, 64, 320, 320, 448],
            'activations': ['relu', 'relu', 'relu', 'relu', 'relu'],
            'dropout': False,
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


## ANN
def build_and_tune_ann_model(X_train, y_train, X_val, y_val, max_trials=5, num_epochs=10, frequency='H'):
    """
    Build and tune an ANN model using Keras Tuner.

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

    def build_ann_model(hp):
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

    tuner = BayesianOptimization(
        hypermodel=build_ann_model,
        objective=keras_tuner.Objective('val_root_mean_squared_error', direction='min'),
        max_trials=max_trials,
        directory="my_dir",
        project_name=f"ann_tuning_{random.randint(1, 100)}",
    )

    tuner.search(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]
    # best_model.summary()
    tuner.results_summary()
    return best_model, best_hp


def build_best_ann_model(y_train, learning_rate=0.0004489034857354316, num_layers=3, units=[320, 32, 32],
                         activations=['relu', 'relu', 'relu'], dropout=False):
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


def ann_train_and_evaluate(df, frequency='H'):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)

    best_params = get_ann_best_params(frequency)
    print(best_params)

    model = build_best_ann_model(y_train, learning_rate=best_params['learning_rate'],
                                 num_layers=best_params['num_layers'],
                                 units=best_params['units'],
                                 activations=best_params['activations'],
                                 dropout=best_params['dropout'])

    cp = ModelCheckpoint(f'ann_model_{frequency}/', save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

    model = load_model(f'ann_model_{frequency}/')

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


def ann_tune_and_evolve(df, frequency='H'):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)

    best_model, best_hp = build_and_tune_ann_model(X_train, y_train, X_val, y_val)

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
            'learning_rate': 0.0017115621539278423,
            'num_layers': 6,
            'units': [64, 384, 480, 160, 32, 32],
            'activations': ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
            'dropout': False,
        },
        'D': {
            'learning_rate': 0.002830345538814358,
            'num_layers': 5,
            'units': [96, 288, 256, 448, 384],
            'activations': ['relu', 'relu', 'relu', 'relu', 'relu'],
            'dropout': True,
        },
        'W': {
            'learning_rate': 0.0017202351384356584,
            'num_layers': 2,
            'units': [224, 320],
            'activations': ['tanh', 'relu'],
            'dropout': True,
        },
        'M': {
            'learning_rate': 0.001601227482635775,
            'num_layers': 4,
            'units': [480, 192, 192, 288],
            'activations': ['relu', 'relu', 'relu', 'relu'],
            'dropout': False,
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def build_and_tune_lstm_model(X_train, y_train, X_val, y_val, max_trials=5, num_epochs=10, frequency='H'):
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

    tuner = BayesianOptimization(
        hypermodel=build_lstm_model,
        objective=keras_tuner.Objective('val_root_mean_squared_error', direction='min'),
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
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)

    best_params = get_lstm_best_params(frequency)
    print(best_params)

    model = build_best_lstm_model(X_train, y_train, learning_rate=best_params['learning_rate'],
                                  num_layers=best_params['num_layers'],
                                  units=best_params['units'],
                                  activations=best_params['activations'],
                                  dropout=best_params['dropout'])

    cp = ModelCheckpoint(f'lstm_model_{frequency}/', save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

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


def lstm_tune_and_evolve(df, frequency='H'):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)

    best_model, best_hp = build_and_tune_lstm_model(X_train, y_train, X_val, y_val)

    return best_model, best_hp


## CNN


def get_cnn_best_params(frequency):
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
            'learning_rate': 0.00010374652025290634,
            'num_layers': 6,
            'units': [352, 448, 416, 64, 288, 288],
            'activations': ['tanh', 'relu', 'relu', 'relu', 'relu', 'relu'],
            'dropout': True,
        },
        'D': {
            'learning_rate': 0.006172427728586157,
            'num_layers': 4,
            'units': [224, 32, 512, 224],
            'activations': ['tanh', 'relu', 'relu', 'relu'],
            'dropout': False,
        },
        'W': {
            'learning_rate': 0.002717326166807044,
            'num_layers': 3,
            'units': [128, 32, 512],
            'activations': ['tanh', 'relu', 'relu'],
            'dropout': True,
        },
        'M': {
            'learning_rate': 0.002539774299219148,
            'num_layers': 5,
            'units': [352, 352, 448, 416, 352],
            'activations': ['tanh', 'relu', 'relu', 'relu', 'relu'],
            'dropout': True,
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def build_and_tune_cnn_model(X_train, y_train, X_val, y_val, max_trials=5, num_epochs=10, frequency='H'):
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

    tuner = BayesianOptimization(
        hypermodel=build_cnn_model,
        objective=keras_tuner.Objective('val_root_mean_squared_error', direction='min'),
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
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

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


def cnn_tune_and_evolve(df, frequency='H'):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(df, frequency)

    best_model, best_hp = build_and_tune_cnn_model(X_train, y_train, X_val, y_val)

    return best_model, best_hp


def cnn_lstm(df):
    print(len(df))
    # df.index = pd.to_datetime(df['Start'], format='%Y-%m-%d %H:%M:%S')
    df.index = pd.to_datetime(df['Start'], format='%Y-%m-%d')
    df_ts = df[['PM2.5-Value', 'NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']]
    df_ts['Seconds'] = df_ts.index.map(pd.Timestamp.timestamp)

    timestamp_s = df_ts['Seconds']

    # Define constants
    second = 1
    minute = 60 * second
    hour = 60 * minute
    day = 24 * hour
    week = 7 * day
    year = (365.2425) * day
    month = (30.44) * day  # Average number of days in a month
    # df_ts['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    # df_ts['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    # df_ts['Hour sin'] = np.sin(timestamp_s * (2 * np.pi / hour))
    # df_ts['Hour cos'] = np.cos(timestamp_s * (2 * np.pi / hour))
    # df_ts['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    # df_ts['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    df_ts['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
    df_ts['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    df_ts['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df_ts['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    df_ts = df_ts.drop('Seconds', axis=1)
    df_new = df_ts

    print(df_new.info())

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_new)
    print(scaled_data.shape)

    # Function to create time series windows
    def create_windows(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size - 1):
            X.append(data[i:(i + window_size), :])
            y.append(data[i + window_size, :])
        return np.array(X), np.array(y)

    window_size = 6
    X, y = create_windows(scaled_data, window_size)
    print(X.shape)

    train_size = int(len(X) * 0.6)  # 60% of the data for training
    validation_size = int(len(X) * 0.2)  # 20% of the data for validation
    test_size = len(X) - train_size - validation_size  # Remaining 20% for testing

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + validation_size], y[train_size:train_size + validation_size]
    X_test, y_test = X[train_size + validation_size:], y[train_size + validation_size:]

    # Now X_train, y_train are for training; X_val, y_val are for validation; X_test, y_test are for testing

    preprocessing_summary = {
        "Total Data Points": len(df_new),
        "Training Data Size": len(X_train),
        "Validation Data Size": len(X_val),
        "Testing Data Size": len(X_test)
    }

    print(preprocessing_summary)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)

    model = Sequential()
    # model.add(layers.InputLayer((X_train.shape[1], X_train.shape[2])))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, 'relu'))
    model.add(layers.Dense(32, 'relu'))
    model.add(layers.Dense(y_train.shape[1], 'linear'))

    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanAbsoluteError()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

    val_predictions = model.predict(X_val)

    # Converting predictions back to original scale
    predicted = scaler.inverse_transform(val_predictions)
    y_val_inverse = scaler.inverse_transform(y_val)

    print(predicted.shape)

    actual = y_val_inverse[:, 0]
    pred = predicted[:, 0]

    plt.figure(figsize=(12, 6))
    plt.plot(actual, color='blue', marker='o', label='Actual', linestyle='-', linewidth=1)
    plt.plot(pred, color='red', marker='x', label='Predicted', linestyle='None')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    # Error Metric for Validation
    mb.evolve_error_metrics(pred, actual)
    mb.naive_mean_absolute_scaled_error(pred, actual)
