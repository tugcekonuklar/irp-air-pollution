import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.impute import KNNImputer
import numpy as np


def merge_dataframes_on_columns(dfs, columns=['Start', 'End'], how='outer'):
    """
    Merges multiple dataframes on specified columns.

    Args:
    - dfs (list of pd.DataFrame): List of dataframes to merge.
    - columns (list of str): Columns to merge on. Default is ['Start', 'End'].
    - how (str): Type of merge. Default is 'outer'.

    Returns:
    - pd.DataFrame: Merged dataframe.
    """
    if not dfs:
        raise ValueError("The list of dataframes is empty.")

    # Start with the first dataframe
    df_merged = dfs[0]

    # Iteratively merge the other dataframes
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on=columns, how=how)

    return df_merged


def rename_df_columns(data_frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Renames the dataframe columns by adding a given prefix.

    Args:
    - data_frame (pd.DataFrame): The original dataframe.
    - prefix (str): The prefix to add to each column name.

    Returns:
    - pd.DataFrame: New dataframe with columns renamed.
    """

    if not isinstance(prefix, str):
        raise ValueError("The provided prefix must be a string.")

    # Dynamically create the new column names with the given prefix
    columns = {
        'Pollutant': f'{prefix}-Pollutant',
        'Value': f'{prefix}-Value',
        'Unit': f'{prefix}-Unit',
        'Validity': f'{prefix}-Validity',
        'Verification': f'{prefix}-Verification'
    }

    # Rename columns without modifying the original dataframe
    renamed_df = data_frame.rename(columns=columns)

    return renamed_df


def drop_unused_df_columns(data_frame: pd.DataFrame, columns_to_drop: list = None) -> pd.DataFrame:
    """
    Drops specified columns from a dataframe and returns a new dataframe.

    Args:
    - data_frame (pd.DataFrame): The original dataframe.
    - columns_to_drop (list): List of columns to drop. Defaults to the specified list.

    Returns:
    - pd.DataFrame: New dataframe with specified columns dropped.
    """

    if columns_to_drop is None:
        columns_to_drop = ['Samplingpoint', 'AggType', 'ResultTime', 'DataCapture', 'FkObservationLog']

    # Drop columns that exist in the dataframe
    cols_to_drop = [col for col in columns_to_drop if col in data_frame.columns]

    return data_frame.drop(cols_to_drop, axis=1)


def get_clean_merged_data() -> pd.DataFrame:
    return pd.read_csv('../data/eea/main/CLEAN_MERGED_DE_DEBB021.csv')


def get_stationarity(timeseries, window=12, visualize=True):
    """
    Check the stationarity of a timeseries using rolling stats and the Augmented Dickey-Fuller test.

    Args:
    - timeseries (pd.Series): The time series data.
    - window (int): Rolling window size. Default is 12.
    - visualize (bool): If True, plots rolling statistics. Default is True.

    Returns:
    - dict: A dictionary containing the ADF result and the p-value.
    """
    # Rolling statistics
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()

    # Visualization of rolling statistics
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(timeseries, color='blue', label='Original')
        plt.plot(rolling_mean, color='red', label='Rolling Mean')
        plt.plot(rolling_std, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

    # Dickey–Fuller test
    result = adfuller(timeseries)
    adf_output = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4]
    }
    return adf_output


def plot_differencing(df, order=2):
    """
    Plots the original time series, its ACF and PACF, and their differentiated versions up to the specified order.

    Args:
    - df (pd.Series): The time series data.
    - order (int): The maximum differentiation order. Default is 2.

    Returns:
    - None: Displays the plots.
    """

    # Plotting parameters
    plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

    # Setup the figure and axes
    fig, axes = plt.subplots(order + 1, 3, sharex=True)

    # Original Series
    axes[0, 0].plot(df)
    axes[0, 0].set_title('Original Series')
    plot_acf(df, ax=axes[0, 1])
    plot_pacf(df, ax=axes[0, 2])

    # Differencing
    for i in range(1, order + 1):
        diff = df.diff().dropna() if i == 1 else df.diff(i).dropna()
        axes[i, 0].plot(diff)
        axes[i, 0].set_title(f'{i}st Order Differencing')
        plot_acf(diff, ax=axes[i, 1])
        plot_pacf(diff, ax=axes[i, 2])

    # Display the plots
    plt.tight_layout()
    plt.show()


def plot_time_series_decomposition(df, column_name, period=1, figsize=(30, 100)):
    """
    Plots the decomposition of a time series into its trend, seasonality, and residuals.

    Args:
    - df (pd.DataFrame): The dataframe containing the time series data.
    - column_name (str): The name of the column containing the time series data to be decomposed.
    - period (int): The period for the seasonal decomposition. Default is 1.
    - figsize (tuple): Figure size for the plot. Default is (30, 100).

    Returns:
    - None: Displays the decomposition plot.
    """

    # Set the figure size
    plt.figure(figsize=figsize)

    # Decompose the series
    decomposed_series = seasonal_decompose(df[column_name], period=period)

    # Plot the decomposed series
    decomposed_series.plot()
    plt.show()


# Use KNN to fill missing values
def impute_missing_with_knn(df, column_name, n_neighbors=5, weights='distance'):
    """
    Imputes missing values in the specified column of the dataframe using KNN.

    Args:
    - df (pd.DataFrame): The dataframe with missing values.
    - column_name (str): The column in which to impute missing values.
    - n_neighbors (int): Number of neighbors for KNN. Default is 3.
    - weights (str): Weight function used in prediction for KNN. Default is 'distance'.

    Returns:
    - pd.DataFrame: DataFrame with missing values imputed.
    """

    df.replace([-999], np.nan, inplace=True)

    # Reshape the data
    X = df[[column_name]]

    # Initialize KNN imputer
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)

    # Fit and transform the data
    df[column_name] = imputer.fit_transform(X)

    return df

def plot_timeseries(df, column, xlabel="Date", ylabel=None, title=None, figsize=(30, 20)):
    """
    Plots a timeseries from a given dataframe column.

    Args:
    - df (pd.DataFrame): The dataframe containing the data.
    - column (str): The name of the column to plot.
    - xlabel (str): The label for the x-axis. Default is "Date".
    - ylabel (str): The label for the y-axis. If None, it uses the column name. Default is None.
    - title (str): The title of the plot. If None, it uses "Column Name Over Time". Default is None.
    - figsize (tuple): Figure size for the plot. Default is (30, 20).

    Returns:
    - None: Displays the plot.
    """
    plt.figure(figsize=figsize)
    plt.plot(df[column])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else column)
    plt.title(title if title else f"{column} Over Time")
    plt.show()