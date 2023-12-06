from tensorflow.keras.models import Sequential
from datetime import datetime

import keras_tuner
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import model_base as mb
import keras_tuner
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import random


def set_index_to_datetime(df, datetime_column_name='Start', datetime_format='%Y-%m-%d %H:%M:%S'):
    """
    Set the DataFrame's index to a datetime column with a specific format.

    Parameters:
        df (DataFrame): The DataFrame for which to set the index.
        datetime_column_name (str): The name of the datetime column.
        datetime_format (str): The format of the datetime column (default is '%Y-%m-%d %H:%M:%S').

    Returns:
        DataFrame: The DataFrame with the updated index.
    """

    df.index = pd.to_datetime(df[datetime_column_name], format=datetime_format)
    return df


def df_to_X_y(df, window_size=24):
    df_values = df.values
    X = [df_values[i:i + window_size] for i in range(len(df_values) - window_size)]
    y = df_values[window_size:, 0]
    print(len(np.array(X)))
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
                           columns=['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value', 'PM2.5-Value']):
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


def ann_split_data(df):
    """
    Preprocess and normalize time series data.

    Parameters:
        df (DataFrame): The DataFrame containing the time series data.
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

    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    # Normalize the features
    X_train_norm, X_val_norm, X_test_norm = mb.scale_features(train_data, validation_data, test_data, MinMaxScaler())
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    return X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test


def load_data(df, frequency='H'):
    window_size = {'H': 24, 'D': 30, 'W': 24, 'M': 6}
    df = set_index_to_datetime(df)
    df = preprocess_time_series(df, frequency)
    return preprocess_and_normalize_data(df, window_size=window_size[frequency])


def ann_load_data(df, frequency='H'):
    df = set_index_to_datetime(df)
    df = preprocess_time_series(df, frequency)
    return ann_split_data(df)


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
            'learning_rate': 0.0019995992408133965,
            'num_layers': 5,
            'units': [224, 320, 384, 320, 288],
            'activations': ['tanh', 'tanh', 'tanh', 'tanh', 'tanh'],
            'dropout': True,
        },
        'D': {
            'learning_rate': 0.00034353533937987133,
            'num_layers': 3,
            'units': [64, 160, 32],
            'activations': ['tanh', 'tanh', 'tanh'],
            'dropout': False,
        },
        'W': {
            'learning_rate': 0.0008264439766792489,
            'num_layers': 2,
            'units': [352, 32],
            'activations': ['tanh', 'tanh'],
            'dropout': False,
        },
        'M': {
            'learning_rate': 0.00015312486598661134,
            'num_layers': 4,
            'units': [192, 480, 224, 32],
            'activations': ['tanh', 'tanh', 'tanh', 'tanh'],
            'dropout': True,
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


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
            'learning_rate': 0.0010932764246458526,
            'num_layers': 2,
            'units': [224, 32],
            'activations': ['relu', 'relu'],
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
            'learning_rate': 0.002269170549520706,
            'num_layers': 6,
            'units': [352, 416, 96, 160, 32, 32],
            'activations': ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
            'dropout': True,
        },
        'M': {
            'learning_rate': 0.0015604152272701418,
            'num_layers': 5,
            'units': [192, 352, 416, 32, 512],
            'activations': ['relu', 'relu', 'relu', 'relu', 'relu'],
            'dropout': True,
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


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
            'learning_rate': 0.0004489034857354316,
            'num_layers': 3,
            'filters': 64,
            'units': [320, 32, 32],
            'activations': ['relu', 'relu', 'relu'],
            'dropout': False,
        },
        'D': {
            'learning_rate': 0.00098949266270698,
            'num_layers': 3,
            'filters': 64,
            'units': [224, 224, 224],
            'activations': ['relu', 'relu', 'relu'],
            'dropout': True,
        },
        'W': {
            'learning_rate': 0.00098949266270698,
            'num_layers': 3,
            'filters': 64,
            'units': [224, 224, 224],
            'activations': ['relu', 'relu', 'relu'],
            'dropout': True,
        },
        'M': {
            'learning_rate': 0.00098949266270698,
            'num_layers': 3,
            'filters': 64,
            'units': [224, 224, 224],
            'activations': ['relu', 'relu', 'relu'],
            'dropout': True,
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

    def build_ann_model(hp):
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

    tuner = RandomSearch(
        hypermodel=build_ann_model,
        objective="val_mean_absolute_error",
        max_trials=max_trials,
        directory="my_dir",
        project_name=f"ann_tuning_{random.randint(1, 100)}",
    )

    tuner.search(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]
    tuner.results_summary()
    return best_model, best_hp


def build_best_ann_model(learning_rate=0.0004489034857354316, num_layers=3, units=[320, 32, 32],
                         activations=['relu', 'relu', 'relu'], dropout=False):
    model = Sequential()

    for i in range(num_layers):
        model.add(layers.Dense(units[i], activations[i]))
    if dropout:
        model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(1, 'linear'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanAbsoluteError()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def ann_train_and_evaluate(df, frequency='H'):
    X_train, X_val, X_test, y_train, y_val, y_test = ann_load_data(df, frequency)

    best_params = get_ann_best_params(frequency)
    print(best_params)

    model = build_best_ann_model(learning_rate=best_params['learning_rate'],
                                 num_layers=best_params['num_layers'],
                                 units=best_params['units'],
                                 activations=best_params['activations'],
                                 dropout=best_params['dropout'])

    rd_num = random.randint(1, 100)
    cp = ModelCheckpoint(f'ann_model_{frequency}_{rd_num}/', save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

    model = load_model(f'ann_model_{frequency}_{rd_num}/')

    print(model.summary())

    # Validation
    val_predictions = model.predict(X_val).flatten()

    val_results = pd.DataFrame(data={'Predictions': val_predictions, 'Actuals': y_val})

    # Error Metric for Validation
    mb.evolve_error_metrics(val_results['Predictions'], val_results['Actuals'])
    mb.naive_mean_absolute_scaled_error(val_results['Predictions'], val_results['Actuals'])

    # Test
    test_predictions = model.predict(X_test).flatten()
    test_results = pd.DataFrame(data={'Predictions': test_predictions, 'Actuals': y_test})

    # Error Metric for Test
    mb.evolve_error_metrics(test_results['Predictions'], test_results['Actuals'])
    mb.naive_mean_absolute_scaled_error(test_results['Predictions'], test_results['Actuals'])

    # Plot Validation
    mb.plot_pm_true_predict_dl(val_results['Actuals'], val_results['Predictions'], 'Validation')

    # Plot Test
    mb.plot_pm_true_predict_dl(test_results['Actuals'], test_results['Predictions'], 'Test')


def ann_tune_and_evolve(df, frequency='H'):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(df, frequency)

    best_model, best_hp = build_and_tune_ann_model(X_train, y_train, X_val, y_val)

    return best_model, best_hp


## LSTM
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

        model.add(layers.Dense(1, activation="linear"))
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        return model

    tuner = RandomSearch(
        hypermodel=build_lstm_model,
        objective="val_mean_absolute_error",
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


def build_best_lstm_model(X_train, learning_rate=0.0004489034857354316, num_layers=3, units=[320, 32, 32],
                          activations=['relu', 'relu', 'relu'], dropout=False):
    model = Sequential()
    model.add(layers.InputLayer((X_train.shape[1], X_train.shape[2])))
    model.add(layers.LSTM(units[0], activations[0]))

    for i in range(1, num_layers):
        model.add(layers.Dense(units[i], activations[i]))

    if dropout:
        model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(1, 'linear'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanAbsoluteError()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def lstm_train_and_evaluate(df, frequency='H'):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(df, frequency)

    best_params = get_lstm_best_params(frequency)
    print(best_params)

    model = build_best_lstm_model(X_train, learning_rate=best_params['learning_rate'],
                                  num_layers=best_params['num_layers'],
                                  units=best_params['units'],
                                  activations=best_params['activations'],
                                  dropout=best_params['dropout'])
    rd_num = random.randint(1, 100)
    cp = ModelCheckpoint(f'lstm_model_{frequency}_{rd_num}/', save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

    model = load_model(f'lstm_model_{frequency}_{rd_num}/')
    # Validation
    val_predictions = model.predict(X_val).flatten()
    val_results = pd.DataFrame(data={'Predictions': val_predictions, 'Actuals': y_val})

    # Error Metric for Validation
    mb.evolve_error_metrics(val_results['Predictions'], val_results['Actuals'])
    mb.naive_mean_absolute_scaled_error(val_results['Predictions'], val_results['Actuals'])

    # Test
    test_predictions = model.predict(X_test).flatten()
    test_results = pd.DataFrame(data={'Predictions': test_predictions, 'Actuals': y_test})

    # Error Metric for Test
    mb.evolve_error_metrics(test_results['Predictions'], test_results['Actuals'])
    mb.naive_mean_absolute_scaled_error(test_results['Predictions'], test_results['Actuals'])

    # Plot Validation
    mb.plot_pm_true_predict_dl(val_results['Actuals'], val_results['Predictions'], 'Validation')

    # Plot Test
    mb.plot_pm_true_predict_dl(test_results['Actuals'], test_results['Predictions'], 'Test')


def lstm_tune_and_evolve(df, frequency='H'):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(df, frequency)

    best_model, best_hp = build_and_tune_lstm_model(X_train, y_train, X_val, y_val)

    return best_model, best_hp


## CNN
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

    def build_cnn_model(hp):
        model = keras.Sequential()
        model.add(layers.InputLayer((X_train.shape[1], X_train.shape[2])))
        model.add(
            layers.Conv1D(
                filters=hp.Int("filters", min_value=32, max_value=512, step=32),
                kernel_size=3,
                activation=hp.Choice("activation_cnn", ["relu", "tanh"]),
            )
        )
        model.add(Flatten())

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

    tuner = RandomSearch(
        hypermodel=build_cnn_model,
        objective="val_mean_absolute_error",
        max_trials=max_trials,
        directory="my_dir",
        project_name=f"cnn_tuning_{random.randint(1, 100)}",
    )

    tuner.search(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]
    tuner.results_summary()
    return best_model, best_hp


def build_best_cnn_model(X_train, learning_rate=0.0004489034857354316, num_layers=3, units=[64, 32, 32],
                         activations=['relu', 'relu', 'relu', 'relu'], dropout=False):
    model = Sequential()
    model.add(layers.InputLayer((X_train.shape[1], X_train.shape[2])))
    model.add(layers.Conv1D(filters=units[0], kernel_size=3, activation=activations[0]))

    for i in range(1, num_layers):
        model.add(layers.Dense(units[i], activations[i]))
    if dropout:
        model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(1, 'linear'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanAbsoluteError()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def cn_train_and_evaluate(df, frequency='H'):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(df, frequency)

    best_params = get_lstm_best_params(frequency)
    print(best_params)

    model = build_best_cnn_model(X_train, learning_rate=best_params['learning_rate'],
                                 num_layers=best_params['num_layers'],
                                 units=best_params['units'],
                                 activations=best_params['activations'],
                                 dropout=best_params['dropout'])
    rd_num = random.randint(1, 100)
    cp = ModelCheckpoint(f'cnn_model_{frequency}_{rd_num}/', save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

    model = load_model(f'cnn_model_{frequency}_{rd_num}/')
    # Validation
    val_predictions = model.predict(X_val).flatten()
    val_results = pd.DataFrame(data={'Predictions': val_predictions, 'Actuals': y_val})

    # Error Metric for Validation
    mb.evolve_error_metrics(val_results['Predictions'], val_results['Actuals'])
    mb.naive_mean_absolute_scaled_error(val_results['Predictions'], val_results['Actuals'])

    # Test
    test_predictions = model.predict(X_test).flatten()
    test_results = pd.DataFrame(data={'Predictions': test_predictions, 'Actuals': y_test})

    # Error Metric for Test
    mb.evolve_error_metrics(test_results['Predictions'], test_results['Actuals'])
    mb.naive_mean_absolute_scaled_error(test_results['Predictions'], test_results['Actuals'])

    # Plot Validation
    mb.plot_pm_true_predict_dl(val_results['Actuals'], val_results['Predictions'], 'Validation')

    # Plot Test
    mb.plot_pm_true_predict_dl(test_results['Actuals'], test_results['Predictions'], 'Test')


def cnn_tune_and_evolve(df, frequency='H'):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(df, frequency)

    best_model, best_hp = build_and_tune_lstm_model(X_train, y_train, X_val, y_val)

    return best_model, best_hp
