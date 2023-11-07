import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import TimeSeriesSplit

FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'


def get_cleaned_df() -> pd.DataFrame:
    return pd.read_csv('../../data/CLEAN_MERGED_DE_DEBB021.csv')


def get_cleaned_datetime_df() -> pd.DataFrame:
    return pd.read_csv('../../data/DATETIME_CLEAN_MERGED_DE_DEBB021.csv')


def set_start_index(df, index_col):
    return df.set_index(index_col, inplace=True)


def define_target_features(df):
    # Separate the features and target
    x = df[FEATURES]
    y = df[TARGET]
    return x, y


def extract_target(train_data, validation_data, test_data):
    # Extract the target variable
    y_train = train_data[TARGET]
    y_val = validation_data[TARGET]
    y_test = test_data[TARGET]
    return y_train, y_val, y_test


def extract_features(train_data, validation_data, test_data):
    return train_data[FEATURES], validation_data[FEATURES], test_data[FEATURES]


def set_start_time_index(df) -> pd.DataFrame:
    return df.set_index('Start_Timestamp', inplace=True)


def init_pca():
    return PCA(n_components=0.95)  # Adjust based on the explained variance


def scale_features(train_data, validation_data, test_data):
    # Scale the features
    scaler = StandardScaler()
    scaler.fit(train_data[FEATURES])  # Fit only on training data

    # Scale the datasets
    x_train_scaled = scaler.transform(train_data[FEATURES])
    x_val_scaled = scaler.transform(validation_data[FEATURES])
    x_test_scaled = scaler.transform(test_data[FEATURES])
    return x_train_scaled, x_val_scaled, x_test_scaled


# def pca_transform(pca, df):
#     return pca.fit_transform(df)


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


#
# def split_time_series_dataframe(df, target_column_name, n_splits=5):
#     """
#     Splits the DataFrame into train, validation, and test sets using TimeSeriesSplit.
#
#     Parameters:
#     df: DataFrame
#         The complete dataset containing features and the target column.
#     target_column_name: str
#         The name of the target variable column.
#     n_splits: int, default=5
#         Number of splits for the TimeSeriesSplit.
#
#     Returns:
#     df_train, df_validation, df_test: tuple of DataFrames
#         Train, validation, and test DataFrame splits.
#     """
#     X = df.drop(columns=[target_column_name])
#     y = df[target_column_name]
#
#     tscv = TimeSeriesSplit(n_splits=n_splits)
#
#     train_indices, validation_indices, test_indices = [], [], []
#
#     for i, (train_index, test_index) in enumerate(tscv.split(X, y)):
#         if i == 0:
#             train_indices = train_index
#         elif i == 1:
#             validation_indices = test_index
#         else:
#             test_indices.extend(test_index)
#
#     df_train = df.iloc[train_indices]
#     df_validation = df.iloc[validation_indices]
#     df_test = df.iloc[test_indices]
#
#     return df_train, df_validation, df_test
#
#
# def split_time_series_data(df):
#     return split_time_series_dataframe(df, TARGET, n_splits=5)

def train(model, x_train, y_train):
    model.fit(x_train, y_train)


def predict(model, x):
    return model.predict(x)


def naive_mean_absolute_scaled_error(y_true, y_pred):
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
    y_naive = np.roll(y_true, 1)

    # Calculate MAE for the naive forecasts
    mae_naive = mean_absolute_error(y_true[1:], y_naive[1:])

    # Calculate MASE for validation and test sets
    mae = mean_absolute_error(y_true, y_pred)
    mase = mae / mae_naive

    print(f"MASE: {mase}")
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


def plot_pm_true_predict(df, y_pred, name):
    # Plotting the actual vs predicted values for validation set
    plt.figure(figsize=(15, 5))
    y_val = df[TARGET]
    # Actual values - using blue color with a line marker
    plt.plot(df.index, y_val, color='blue', marker='o', label='Actual', linestyle='-', linewidth=1)

    # Predicted values - using red color with a cross marker
    plt.plot(df.index, y_pred, color='red', marker='x', label='Predicted', linestyle='None')

    plt.title(f'{name} Set - Actual vs Predicted PM2.5')
    plt.xlabel('Date')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid()
    plt.show()


def plot_time_series(train_data, y_train, test_data, y_test, test_predictions_mean, y_test_pred, name):
    """
    Plots the time series data including training, validation sets and predictions with confidence intervals.

    Parameters:
    train_data (DataFrame): The training dataset with a DateTimeIndex.
    y_train (Series): The training data target values.
    test_data (DataFrame): The test dataset with a DateTimeIndex.
    y_test (Series): The test data target values.
    test_predictions_mean (Series): The predicted mean values for the tested data.
    y_test_pred (PredictionResults): The prediction results object that has a `conf_int` method for confidence intervals.
    """
    plt.figure(figsize=(15,5))

    plt.title(f'{name} Set - Actual vs Predicted PM2.5')
    plt.xlabel('Date')
    plt.ylabel('PM2.5')

    plt.plot(train_data.index, y_train, label='Train')
    plt.plot(test_data.index, y_test, label=name, marker='o', linestyle='-', color='green')
    plt.plot(test_data.index, test_predictions_mean, label='Predictions', marker='x', linestyle='None' , color='red')
    plt.fill_between(test_data.index,
                     y_test_pred.conf_int().iloc[:, 0],
                     y_test_pred.conf_int().iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.legend()
    plt.show()
