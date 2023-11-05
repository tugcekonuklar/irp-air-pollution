import os

os.chdir('..')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'

def get_cleaned_df() -> pd.DataFrame:
    return pd.read_csv('data/CLEAN_MERGED_DE_DEBB021.csv')


def set_start_time_index(df) -> pd.DataFrame:
    return df.set_index('Start_Timestamp', inplace=True)


def split_data(df):
    train_ratio = 0.6
    validation_ratio = 0.2

    # Calculate the indices for the splits
    train_end = int(len(df) * train_ratio)
    validation_end = train_end + int(len(df) * validation_ratio)

    # Split the dataset
    train_data = df[:train_end]
    validation_data = df[train_end:validation_end]
    test_data = df[validation_end:]

    print(f"Training set size: {train_data.shape[0]}")
    print(f"Validation set size: {validation_data.shape[0]}")
    print(f"Test set size: {test_data.shape[0]}")
    return train_data, validation_data, test_data


def train(model, x_train, y_train):
    model.fit(x_train, y_train)


def predict(model, x):
    return model.predict(x)


def naive_mean_absolute_scaled_error(y, y_pred):
    """
    Calculate the Mean Absolute Scaled Error (MASE) for forecasts with Naive Approach.

    Parameters:
    y_true (array): Actual observed values.
    y_pred (array): Forecasted values, same length as y_true.
    y_naive (array): Naive (benchmark) forecast values, same length as y_true.

    Returns:
    float: The MASE value.
    """

    # Create naive forecasts for validation and test sets
    y_naive = np.roll(y, 1)

    # Calculate MAE for the naive forecasts
    mae_naive = mean_absolute_error(y[1:], y_naive[1:])

    # Calculate MASE for validation and test sets
    mae = mean_absolute_error(y, y_pred)
    mase = mae / mae_naive

    print(f"Validation MASE: {mase}")
    return mase


def evolve_error_metrics(y_true, y_pred):
    """
    Calculate and print the test metrics: MAE, MSE, RMSE, and MAPE.

    Parameters:
    y_true (array): Actual observed values.
    y_pred (array): Predicted values, same length as y_true.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }
    return metrics


def plot_pm_actual_predict(df, y_pred):
    # Plotting the actual vs predicted values for validation set
    plt.figure(figsize=(15, 5))
    y_val = df[TARGET]
    # Actual values - using blue color with a line marker
    plt.plot(df.index, y_val, color='blue', marker='o', label='Actual', linestyle='-', linewidth=1)

    # Predicted values - using red color with a cross marker
    plt.plot(df.index, y_pred, color='red', marker='x', label='Predicted', linestyle='None')

    plt.title('Validation Set - Actual vs Predicted PM2.5')
    plt.xlabel('Date')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.show()
