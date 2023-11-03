import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def set_start_inde(df, index_col):
    return df.set_index(index_col, inplace=True)


def decompose_time_series(df, column, period):
    decomposition = seasonal_decompose(df[column], model='additive', period=period)
    df['trend'] = decomposition.trend
    df['seasonal'] = decomposition.seasonal
    df['residual'] = decomposition.resid
    df['detrended'] = df[column] - df['trend']
    df['deseasonalized'] = df[column] - df['seasonal']
    return df


def prepare_data(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]
    return X, y


def split_data(X, y, train_size, val_size):
    train_end = int(len(X) * train_size)
    val_end = train_end + int(len(X) * val_size)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train):
    """
    Trains a Linear Regression model using the provided training data.

    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target.

    Returns:
    model: The trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute the Mean Absolute Percentage Error (MAPE)

    Parameters:
    y_true (np.array): Actual values.
    y_pred (np.array): Predicted values.

    Returns:
    mape (float): The MAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    """
    Compute the Mean Absolute Scaled Error (MASE)

    Parameters:
    y_true (np.array): Actual values.
    y_pred (np.array): Predicted values.
    y_train (np.array): Training data used for modeling.

    Returns:
    mase (float): The MASE value.
    """
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d


def evaluate_model(model, X_val, y_val, df_val, y_train):
    """
    Evaluates the trained model using validation data and calculates various metrics.

    Parameters:
    model: The trained Linear Regression model.
    X_val (pd.DataFrame): The validation features.
    y_val (pd.Series): The validation target.
    df_val (pd.DataFrame): DataFrame containing 'trend' and 'seasonal' columns for the validation data.
    y_train (pd.Series): The training target for calculating MASE.

    Returns:
    metrics (dict): A dictionary containing various evaluation metrics.
    y_val_pred (pd.Series): Predicted values on the validation set.
    """
    y_val_pred = model.predict(X_val)
    # Reapply trend and seasonality if they were removed
    y_val_pred += df_val['trend'] + df_val['seasonal']
    mse = mean_squared_error(y_val + df_val['trend'] + df_val['seasonal'], y_val_pred)
    mae = mean_absolute_error(y_val + df_val['trend'] + df_val['seasonal'], y_val_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_val + df_val['trend'] + df_val['seasonal'], y_val_pred)
    mase = mean_absolute_scaled_error(y_val + df_val['trend'] + df_val['seasonal'], y_val_pred,
                                      y_train + df_val.loc[y_train.index, 'trend'])

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase
    }

    return metrics, y_val_pred
