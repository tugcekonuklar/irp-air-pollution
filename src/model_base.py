import warnings
import os
import pickle
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'

HOURLY_DATETIME_FILE_PATH = '../data/HOURLY_DATETIME_CLEAN_MERGED_DE_DEBB021.csv'
HOURLY_TIMESTAMP_FILE_PATH = '../data/HOURLY_TIMESTAMP_CLEAN_MERGED_DE_DEBB021.csv'

DAILY_DATETIME_FILE_PATH = '../data/DAILY_DATETIME_CLEAN_MERGED_DE_DEBB021.csv'
DAILY_TIMESTAMP_FILE_PATH = '../data/DAILY_TIMESTAMP_CLEAN_MERGED_DE_DEBB021.csv'

WEEKLY_DATETIME_FILE_PATH = '../data/WEEKLY_DATETIME_CLEAN_MERGED_DE_DEBB021.csv'
WEEKLY_TIMESTAMP_FILE_PATH = '../data/WEEKLY_TIMESTAMP_CLEAN_MERGED_DE_DEBB021.csv'

MONTHLY_DATETIME_FILE_PATH = '../data/MONTLY_DATETIME_CLEAN_MERGED_DE_DEBB021.csv'
MONTHLY_TIMESTAMP_FILE_PATH = '../data/MONTLY_TIMESTAMP_CLEAN_MERGED_DE_DEBB021.csv'



def save_model_to_pickle(model, file_path):
    """
    Saves a machine learning model to a file using pickle.

    Args:
        model: The machine learning model to be saved.
        file_path (str): The path (including file name) where the model should be saved.
    """
    with open(f'../models/{file_path}', 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {file_path}")

def convert_parquet_to_csv(input_folder="../data/parquet", output_folder="../data/csv"):
    """
    Converts all Parquet files in the input_folder to CSV files in the output_folder.

    Parameters:
    - input_folder: Folder containing Parquet files.
    - output_folder: Destination folder for CSV files.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".parquet"):
            # Full paths for the input Parquet and output CSV files
            parquet_file = os.path.join(input_folder, filename)
            csv_file = os.path.join(output_folder, filename.replace(".parquet", ".csv"))

            try:
                # Read the Parquet file and convert to a DataFrame
                df = pq.read_table(parquet_file).to_pandas()

                # Write the DataFrame to a CSV file
                df.to_csv(csv_file, index=False)
                print(f"Successfully converted {parquet_file} to {csv_file}.")
            except Exception as e:
                print(f"Error converting {parquet_file}: {e}")


def move_columns_to_front(df, cols_to_move):
    """
    Move specified columns to the front of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to reorder.
    cols_to_move (list of str): The column names to move to the front.

    Returns:
    pd.DataFrame: The DataFrame with columns reordered.
    """
    # Filter out the columns to move from the original column list
    other_cols = [col for col in df.columns if col not in cols_to_move]
    # Create the new column order
    new_order = cols_to_move + other_cols
    # Reindex the DataFrame with the new column order
    return df[new_order]


def prepare_datetime_and_reorder(df, date_cols):
    """
    Convert specified string date columns to datetime, then to Unix timestamp,
    and finally move the timestamp columns to the beginning of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    date_cols (list of str): The names of the date columns to convert.

    Returns:
    pd.DataFrame: The processed DataFrame with datetime conversions and reordered columns.
    """
    timestamp_cols = []

    # Convert date columns to datetime and then to Unix timestamp
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        timestamp_col = col + '_Timestamp'
        df[timestamp_col] = df[col].astype('int64') // 10 ** 9
        timestamp_cols.append(timestamp_col)

    # Use the separate method to move the timestamp columns to the beginning
    return move_columns_to_front(df, timestamp_cols)


def drop_df_columns(data_frame: pd.DataFrame, columns_to_drop: list = None) -> pd.DataFrame:
    """
    Drops specified columns from a dataframe and returns a new dataframe.

    Args:
    - data_frame (pd.DataFrame): The original dataframe.
    - columns_to_drop (list): List of columns to drop. Defaults to the specified list.

    Returns:
    - pd.DataFrame: New dataframe with specified columns dropped.
    """
    # Drop columns that exist in the dataframe
    cols_to_drop = [col for col in columns_to_drop if col in data_frame.columns]

    return data_frame.drop(cols_to_drop, axis=1)


def read_all_air_pollutant_csv():
    """
    Reads multiple CSV files into a dictionary of DataFrames.

    Args:
    - file_paths (dict): Dictionary where keys are descriptive names and values are file paths.

    Returns:
    - dict: Dictionary of DataFrames.
    """

    file_paths = {
        'pm25': '../data/csv/PM2.5_DE_DEBB021.csv',
        'pm10': '../data/csv/PM10_DE_DEBB021.csv',
        'no2': '../data/csv/NO2_DE_DEBB021.csv',
        'o3': '../data/csv/O3_DE_DEBB021.csv',
        'so2': '../data/csv/SO2_DE_DEBB021.csv'
    }
    dataframes = {}
    for key, path in file_paths.items():
        try:
            dataframes[key] = pd.read_csv(path)
        except FileNotFoundError as e:
            print(f"Error reading {path}: {e}")
            continue

    return dataframes['pm25'], dataframes['pm10'], dataframes['no2'], dataframes['o3'], dataframes['so2']


def read_frequency_data(file_paths):
    """
    Reads frequency data from specified file paths.

    Args:
    - file_paths (dict): Dictionary containing file paths for hourly, daily, weekly, and monthly data.

    Returns:
    - Tuple of DataFrames: (df_hourly, df_daily, df_weekly, df_monthly)
    """
    try:
        df_hourly = pd.read_csv(file_paths['hourly'])
        df_daily = pd.read_csv(file_paths['daily'])
        df_weekly = pd.read_csv(file_paths['weekly'])
        df_monthly = pd.read_csv(file_paths['monthly'])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error reading files: {e}")

    return df_hourly, df_daily, df_weekly, df_monthly


def read_date_freq():
    """
    Reads date-frequency data from predefined file paths.

    This function maps various date frequencies (hourly, daily, weekly, monthly)
    to their respective file paths and reads the data using the read_frequency_data function.

    Returns:
        The result of the read_frequency_data function call with the mapped file paths.
    """
    return read_frequency_data({
        'hourly': HOURLY_DATETIME_FILE_PATH,
        'daily': DAILY_DATETIME_FILE_PATH,
        'weekly': WEEKLY_DATETIME_FILE_PATH,
        'monthly': MONTHLY_DATETIME_FILE_PATH
    })


def read_timestamp_freq():
    """
    Reads timestamp frequency data from predefined file paths.

    This function maps various timestamp frequencies (hourly, daily, weekly, monthly)
    to their respective file paths and reads the data using the read_frequency_data function.

    Returns:
        The result of the read_frequency_data function call with the mapped file paths.
    """
    return read_frequency_data({
        'hourly': HOURLY_TIMESTAMP_FILE_PATH,
        'daily': DAILY_TIMESTAMP_FILE_PATH,
        'weekly': WEEKLY_TIMESTAMP_FILE_PATH,
        'monthly': MONTHLY_TIMESTAMP_FILE_PATH
    })


def read_hourly_datetime_df():
    return pd.read_csv(HOURLY_DATETIME_FILE_PATH)


def process_and_save_freq_data(date_file_path, timestamp_file_path, resample='D', drop_columns='Start'):
    """
    Processes hourly data to resample, adds a timestamp column, and saves to two CSV files.

    Args:
    - date_file_path (str): Path to save the daily data CSV file with date.
    - timestamp_file_path (str): Path to save the daily data CSV file with timestamp.
    - drop_columns (list of str): Optional. List of column names to drop before saving the timestamp file.
    """

    # Get hourly cleaned data
    df = read_hourly_datetime_df()

    # Convert 'Start' column to datetime and set it as index
    df['Start'] = pd.to_datetime(df['Start'])
    df = df.set_index('Start')

    df = df.drop(columns=['PM2.5-Unit', 'PM10-Unit', 'NO2-Unit', 'O3-Unit', 'SO2-Unit'])

    # Resample data and take the median
    df_resample = df.resample(resample).median()

    # Add a 'Start_Timestamp' column with Unix timestamp in seconds
    df_resample['Start_Timestamp'] = df_resample.index.view('int64') // 10 ** 9

    # Save the processed data to a CSV file with date
    df_resample.to_csv(date_file_path, index=True)

    df_resample = pd.read_csv(date_file_path)

    # Drop specified columns if provided
    if drop_columns:
        df_resample.drop(columns=drop_columns, inplace=True)

    # Save the processed data to a CSV file with timestamp
    df_resample.to_csv(timestamp_file_path, index=False)


def process_and_save_hourly_data(df, date_file_path, timestamp_file_path):
    """
    Processes hourly data by dropping specific columns and reordering, then saves the data to CSV files.

    Args:
    - df (pd.DataFrame): DataFrame containing the hourly data.
    - date_file_path (str): File path to save the hourly data with date.
    - timestamp_file_path (str): File path to save the hourly data with timestamp.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if not isinstance(date_file_path, str) or not isinstance(timestamp_file_path, str):
        raise TypeError("File paths must be strings")

    # Drop unnecessary columns and prepare datetime
    df = drop_df_columns(df, ['End', 'End_Timestamp'])
    df = prepare_datetime_and_reorder(df, ['Start'])

    # Save the DataFrame with the 'Start' column
    df.to_csv(date_file_path, index=False)

    # Drop the 'Start' column for the timestamp file
    df.drop(columns='Start', inplace=True)

    # Save the processed data to a CSV file with timestamp
    df.to_csv(timestamp_file_path, index=False)


def process_date_freq_data(df):
    """
    Processes data for different frequencies and saves to specified file paths.

    Args:
    - df (pd.DataFrame): DataFrame containing the data to be processed.
    - file_paths (dict): Dictionary containing file paths for each frequency.
    """

    # Example usage
    file_paths = {
        'H': {'datetime': HOURLY_DATETIME_FILE_PATH, 'timestamp': HOURLY_TIMESTAMP_FILE_PATH},
        'D': {'datetime': DAILY_DATETIME_FILE_PATH, 'timestamp': DAILY_TIMESTAMP_FILE_PATH},
        'W': {'datetime': WEEKLY_DATETIME_FILE_PATH, 'timestamp': WEEKLY_TIMESTAMP_FILE_PATH},
        'M': {'datetime': MONTHLY_DATETIME_FILE_PATH, 'timestamp': MONTHLY_TIMESTAMP_FILE_PATH}
    }

    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    date_freq = ['H', 'D', 'W', 'M']

    for freq in date_freq:
        datetime_path = file_paths[freq]['datetime']
        timestamp_path = file_paths[freq]['timestamp']

        if freq == 'H':
            # Assuming process_and_save_hourly_data is defined to handle hourly data
            process_and_save_hourly_data(df, datetime_path, timestamp_path)
        else:
            # Assuming process_and_save_freq_data is defined to handle D, W, M frequencies
            process_and_save_freq_data(datetime_path, timestamp_path, resample=freq)


def set_start_index(df, index_col):
    """
    Sets the specified column as the DataFrame index in-place.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        index_col (str): The column name to set as the new index.

    Returns:
        None: The operation modifies the DataFrame in-place and does not return a value.
    """
    df.set_index(index_col, inplace=True)


def define_target_features(df):
    """
    Separates the specified features and target from the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        tuple: A tuple where the first element is the target series (y) and the
               second element is the features DataFrame (X).
    """
    x = df[FEATURES]
    y = df[TARGET]
    return y, x


def extract_target(train_data, validation_data, test_data):
    """
     Extracts the target column from training, validation, and test datasets.

     Args:
         train_data (pandas.DataFrame): The training dataset.
         validation_data (pandas.DataFrame): The validation dataset.
         test_data (pandas.DataFrame): The test dataset.

     Returns:
         tuple: A tuple containing the target series from the training, validation,
                and test datasets in that order (y_train, y_val, y_test).
     """
    y_train = train_data[TARGET]
    y_val = validation_data[TARGET]
    y_test = test_data[TARGET]
    return y_train, y_val, y_test


def extract_features(train_data, validation_data, test_data):
    """
    Extracts specified features from the training, validation, and test datasets.

    Args:
        train_data (pandas.DataFrame): The training dataset.
        validation_data (pandas.DataFrame): The validation dataset.
        test_data (pandas.DataFrame): The test dataset.

    Returns:
        tuple: A tuple containing the features extracted from the training,
               validation, and test datasets, respectively.
    """
    return train_data[FEATURES], validation_data[FEATURES], test_data[FEATURES]


def set_start_time_index(df) -> pd.DataFrame:
    """
        Sets the 'Start_Timestamp' column as the DataFrame's index.

        Args:
            df (pd.DataFrame): The DataFrame to modify.

        Returns:
            pd.DataFrame: A new DataFrame with 'Start_Timestamp' set as the index.
        """
    return df.set_index('Start_Timestamp', inplace=True)


def set_start_date_time_index(df) -> pd.DataFrame:
    """
        Sets the 'Start_Timestamp' column as the DataFrame's index.

        Args:
            df (pd.DataFrame): The DataFrame to modify.

        Returns:
            pd.DataFrame: A new DataFrame with 'Start' set as the index.
        """
    df['Start'] = pd.to_datetime(df['Start'])
    return set_start_index(df, 'Start')


def scale_features(train_data, validation_data, test_data, scaler=StandardScaler()):
    """
    Scales the specified features in the training, validation, and test datasets.

    Args:
        train_data (pandas.DataFrame): The training dataset.
        validation_data (pandas.DataFrame): The validation dataset.
        test_data (pandas.DataFrame): The test dataset.
        scaler (sklearn.preprocessing.StandardScaler): An instance of a scaler to use. Defaults to StandardScaler()

    Returns:
        tuple: A tuple containing the scaled features for the training, validation, and test datasets.
    """
    # Scale the features
    scaler.fit(train_data[FEATURES])  # Fit only on training data

    # Scale the datasets
    x_train_scaled = scaler.transform(train_data[FEATURES])
    x_val_scaled = scaler.transform(validation_data[FEATURES])
    x_test_scaled = scaler.transform(test_data[FEATURES])
    return x_train_scaled, x_val_scaled, x_test_scaled


def split_data(df, train_ratio=0.6, validation_ratio=0.2):
    """
    Splits the dataset into training, validation, and test sets based on provided ratios.

    Args:
        df (pandas.DataFrame): The dataset to split.
        train_ratio (float): The proportion of the dataset to include in the train split.
        validation_ratio (float): The proportion of the dataset to include in the validation split.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.

    Raises:
        ValueError: If the sum of the ratios exceeds 1.
    """

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
    bias = np.mean(y_pred - y_true)  # Calculating bias

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"Bias: {bias:.4f}")

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Bias': bias
    }
    return metrics


def plot_pm_true_predict(df, y_pred, name):
    """
    Plots the actual versus predicted PM2.5 values.

    Args:
        df (DataFrame): DataFrame containing the actual values and timestamps.
        y_pred (array): Array of predicted values.
        name (str): Name of the dataset (e.g., 'Validation', 'Test') to use in the plot title.

    This function creates a line plot displaying both the actual and predicted PM2.5
    values over time, facilitating the visual comparison of model performance.
    """
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
    plt.grid(True)  # More explicit call to enable the grid
    plt.show()


def plot_pm_true_index_predict(y_actual, y_pred, name):
    """
    Plots the actual vs predicted PM2.5 values for a given dataset.

    Args:
        y_actual (pd.Series): A pandas Series containing the actual PM2.5 values with dates as the index.
        y_pred (Iterable): An iterable (e.g., list or numpy array) of predicted PM2.5 values. Must match the length of y_actual.
        name (str): The name of the dataset (e.g., 'Training', 'Validation', 'Test') to include in the plot title.

    This function creates a line plot showing actual and predicted PM2.5 values to facilitate visual comparison.
    """
    # Ensure the predicted values align with the actual values in length
    if len(y_actual) != len(y_pred):
        raise ValueError("The lengths of actual and predicted values do not match.")

    plt.figure(figsize=(15, 5))

    # Plot actual values
    plt.plot(y_actual.index, y_actual, color='blue', marker='o', label='Actual', linestyle='-', linewidth=1)

    # Plot predicted values
    plt.plot(y_actual.index, y_pred, color='red', marker='x', label='Predicted', linestyle='None')

    plt.title(f'{name} Set - Actual vs Predicted PM2.5')
    plt.xlabel('Date')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid(True)
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
    plt.figure(figsize=(15, 5))

    plt.title(f'{name} Set - Actual vs Predicted PM2.5')
    plt.xlabel('Date')
    plt.ylabel('PM2.5')

    plt.plot(train_data.index, y_train, label='Train')
    plt.plot(test_data.index, y_test, label=name, marker='o', linestyle='-', color='green')
    plt.plot(test_data.index, test_predictions_mean, label='Predictions', marker='x', linestyle='None', color='red')
    plt.fill_between(test_data.index,
                     y_test_pred.conf_int().iloc[:, 0],
                     y_test_pred.conf_int().iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.legend()
    plt.show()
