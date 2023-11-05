import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'


def set_start_index(df, index_col):
    return df.set_index(index_col, inplace=True)


def define_target_features(df, feature_columns, target_column):
    # Separate the features and target
    x = df[FEATURES]
    y = df[TARGET]
    return x, y


def split_data(x, y, train_size, val_size):
    train_end = int(len(x) * train_size)
    val_end = train_end + int(len(x) * val_size)
    x_train, y_train = x.iloc[:train_end], y.iloc[:train_end]
    x_val, y_val = x.iloc[train_end:val_end], y.iloc[train_end:val_end]
    x_test, y_test = x.iloc[val_end:], y.iloc[val_end:]
    return x_train, x_val, x_test, y_train, y_val, y_test


def extract_target(train_data, validation_data, test_data):
    # Extract the target variable
    y_train = train_data[TARGET]
    y_val = validation_data[TARGET]
    y_test = test_data[TARGET]
    return y_train, y_val, y_test


def scale_features(train_data, validation_data, test_data):
    # Scale the features
    scaler = StandardScaler()
    scaler.fit(train_data[FEATURES])  # Fit only on training data

    # Scale the datasets
    x_train_scaled = scaler.transform(train_data[FEATURES])
    x_val_scaled = scaler.transform(validation_data[FEATURES])
    x_test_scaled = scaler.transform(test_data[FEATURES])
    return x_train_scaled, x_val_scaled, x_test_scaled


def init_pca():
    return PCA(n_components=0.95)  # Adjust based on the explained variance






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


def evaluate_model(model, x_val, y_val, df_val, y_train):
    """
    Evaluates the trained model using validation data and calculates various metrics.

    Parameters:
    model: The trained Linear Regression model.
    x_val (pd.DataFrame): The validation features.
    y_val (pd.Series): The validation target.
    df_val (pd.DataFrame): DataFrame containing 'trend' and 'seasonal' columns for the validation data.
    y_train (pd.Series): The training target for calculating MASE.

    Returns:
    metrics (dict): A dictionary containing various evaluation metrics.
    y_val_pred (pd.Series): Predicted values on the validation set.
    """
    y_val_pred = model.predict(x_val)
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
