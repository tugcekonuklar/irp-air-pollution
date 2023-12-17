import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import model_base as mb


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

    df_merged = df_merged.sort_values(by='Start')
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

    return mb.drop_df_columns(data_frame, columns_to_drop)


def get_clean_merged_data() -> pd.DataFrame:
    return pd.read_csv('data/CLEAN_MERGED_DE_DEBB021.csv')


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
        plt.figure(figsize=(8, 6))
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
    plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 120})

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


def plot_time_series_decomposition(df, column_name, period=1, figsize=(10, 6)):
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
def impute_missing_with_knn(df, column_names, n_neighbors=5, weights='distance'):
    """
    Imputes missing values in the specified columns of the dataframe using KNN.

    Args:
    - df (pd.DataFrame): The dataframe with missing values.
    - column_names (list): The list of columns in which to impute missing values.
    - n_neighbors (int): Number of neighbors for KNN. Default is 5.
    - weights (str): Weight function used in prediction for KNN. Default is 'distance'.

    Returns:
    - pd.DataFrame: DataFrame with missing values imputed.
    """

    df.replace([-999], np.nan, inplace=True)

    # Initialize KNN imputer
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)

    # Apply imputation for each column in the list
    for column_name in column_names:
        # Reshape the data
        X = df[[column_name]]

        # Fit and transform the data
        df[[column_name]] = imputer.fit_transform(X)

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


def from_datetime_to_timestamp(df, col):
    df[col] = pd.to_datetime(df[col])
    timestamp_col = col + '_Timestamp'
    df[timestamp_col] = df[col].astype('int64') // 10 ** 9
    return df


def check_seasonality_and_trend(df, column_name='PM2.5-Value', freq='H'):
    # Check if DataFrame has a DateTimeIndex with frequency set
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame({
            'Error': ['DataFrame index is not a DateTimeIndex']
        })
    if df.index.freq is None:
        try:
            df = df.asfreq(freq)
            if df.index.freq is None:
                return pd.DataFrame({
                    'Error': ['Unable to set a frequency on DataFrame index']
                })
        except ValueError as e:
            return pd.DataFrame({
                'Error': [str(e)]
            })

    # Extract the time series data
    time_series = df[column_name].dropna()  # Ensure there are no NaNs in the series

    # Decompose the time series using the seasonal_decompose method
    decomposition = sm.tsa.seasonal_decompose(time_series, model='additive', period=24)  # Assuming 24-hour seasonality
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Perform Dickey-Fuller test
    dftest = adfuller(time_series.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    # Plotting the decomposed components
    fig, axes = plt.subplots(4, 1, figsize=(10, 6))
    time_series.plot(ax=axes[0], title='Original')
    axes[0].set_ylabel('Original')
    trend.plot(ax=axes[1], title='Trend')
    axes[1].set_ylabel('Trend')
    seasonal.plot(ax=axes[2], title='Seasonality')
    axes[2].set_ylabel('Seasonality')
    residual.plot(ax=axes[3], title='Residuals')
    axes[3].set_ylabel('Residuals')
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        'Dickey-Fuller Test Statistic': [dfoutput['Test Statistic']],
        'p-value': [dfoutput['p-value']],
        '#Lags Used': [dfoutput['#Lags Used']],
        'Number of Observations': [dfoutput['Number of Observations Used']],
        'Critical Value 1%': [dfoutput['Critical Value (1%)']],
        'Critical Value 5%': [dfoutput['Critical Value (5%)']],
        'Critical Value 10%': [dfoutput['Critical Value (10%)']],
        'ADF Result': ['Stationary' if dfoutput['p-value'] < 0.05 else 'Non-Stationary']
    })


def analyze_skewness(df, column_name):
    """
    Calculate and visualize the skewness of a time series dataset.

    Args:
    - df (pd.DataFrame): DataFrame containing the time series data.
    - column_name (str): Name of the column with time series data.

    Returns:
    - float: Skewness of the data.
    """
    # Calculate skewness
    data_skewness = skew(df[column_name])
    print(f"Skewness of the data: {data_skewness}")

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))  # 1 row, 2 columns

    # Histogram on the first subplot
    sns.histplot(df[column_name], kde=True, ax=ax1)
    ax1.set_title('Histogram of Time Series Data')
    ax1.set_xlabel(column_name)
    ax1.set_ylabel('Frequency')

    # Density Plot on the second subplot
    sns.kdeplot(df[column_name], fill=True, ax=ax2)
    ax2.set_title('Density Plot of Time Series Data')
    ax2.set_xlabel(column_name)

    # Automatically adjust subplot params for better layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    return data_skewness


def detect_and_show_outliers(df, column_name):
    """
    Detects outliers in a time series dataset and visualizes them.

    Args:
    - df (pd.DataFrame): DataFrame containing the time series data.
    - column_name (str): Name of the column with time series data.

    Returns:
    - pd.DataFrame: DataFrame containing the outliers.
    """
    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))  # 1 row, 2 columns

    # Time Series Plot on the first subplot
    ax1.plot(df[column_name], marker='o', linestyle='-')
    ax1.set_title('Time Series Plot')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(column_name)

    # Box Plot on the second subplot
    ax2.boxplot(df[column_name], vert=False)
    ax2.set_title('Box Plot')
    ax2.set_xlabel(column_name)

    # Automatically adjust subplot params for better layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Identify outliers using IQR
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column_name] < (Q1 - 1.5 * IQR)) | (df[column_name] > (Q3 + 1.5 * IQR))]

    # print("Outliers using IQR method:")
    # print(outliers)

    return outliers


def correlation_matrix(df, figsize=(20, 8), cmap='coolwarm', title="Correlation Matrix"):
    """
    Generates a heatmap for the correlation matrix of the DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame for which to generate the correlation matrix.
    - figsize (tuple): Size of the figure (width, height).
    - cmap (str): Colormap for the heatmap.
    - title (str): Title of the plot.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap)
    plt.title(title)
    plt.show()


def pairplot(df, title="Pairplot"):
    """
    Generates a pairplot for the DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame for which to generate the pairplot.
    - title (str): Title of the plot.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    pair_plot = sns.pairplot(df)
    pair_plot.fig.suptitle(title, y=1.02)  # Adjust the title position
    plt.show()


def feature_importance(df):
    # Train a simple Random Forest model and check the feature importances. 
    # Define your features and target variable
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    # Assuming X is your feature set and y is the target variable
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.show()


def comprehensive_eda(df):
    correlation_matrix(df)
    pairplot(df)
    feature_importance(df)


def analyse_data_frame(df):
    """
    Analyzes a DataFrame for various statistical properties including autocorrelation,
    skewness, outliers, stationarity, seasonality, and trend.

    Args:
    - df (pd.DataFrame): DataFrame containing the data to be analyzed.
    """

    # Calculate and print skewness of PM2.5 values
    skewness = analyze_skewness(df, 'PM2.5-Value')
    print(f"Skewness of PM2.5-Value: {skewness}")
    if skewness > 0:
        print('There is more weight in the right tail of the distribution.')
    elif skewness < 0:
        print('There is more weight in the left tail of the distribution.')
    else:
        print('The distribution is symmetric.')

        # Detect and print information about outliers in PM2.5 values
    outliers = detect_and_show_outliers(df, 'PM2.5-Value')
    print(f'Count of outliers: {len(outliers)}')
    outlier_percentage = (len(outliers) / len(df)) * 100
    print(f'Percentage of outliers: {outlier_percentage:.2f}%')

    # Analyze autocorrelation for PM2.5 values
    df_pm25 = df[['PM2.5-Value']]
    autocorrelation_plot(df_pm25)

    # Check for stationarity in PM2.5 values
    # get_stationarity(df[['PM2.5-Value']], visualize=True)

    # Check for seasonality and trend in PM2.5 values
    # Assuming mb.get_cleaned_datetime_df() is a function to clean/prepare the DataFrame
    df['Start'] = pd.to_datetime(df['Start'])
    df = df.set_index('Start')

    result_df = check_seasonality_and_trend(df, 'PM2.5-Value')
    if 'Error' in result_df.columns:
        print(f"Error in seasonality and trend analysis: {result_df['Error'][0]}")
    else:
        print("Seasonality and Trend Analysis Results:")
        print(result_df)
