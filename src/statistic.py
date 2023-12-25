import model_base as mb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import itertools

ALL = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value', 'PM2.5-Value']
TARGET = 'PM2.5-Value'


def get_arimax_best_params(frequency='H'):
    """
    Returns the best parameters for ARIMAX based on the specified frequency.

    Args:
    - frequency (str): The frequency for which to get the best parameters. Options are 'H', 'D', 'W', 'M'.

    Returns:
    - dict: A dictionary of the best parameters.
    """
    # Define best parameters for each frequency
    best_params = {
        'H': {
            'order': (0, 0, 0),
            'seasonal_order': (0, 0, 0, 0),
        },
        'D': {
            'order': (6, 0, 1),
            'seasonal_order': (0, 0, 0, 0),
        },
        'W': {
            'order': (6, 0, 7),
            'seasonal_order': (0, 0, 0, 0),
        },
        'M': {
            'order': (3, 0, 6),
            'seasonal_order': (0, 0, 0, 0),
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def plot_pm_true_predict(y_actual, y_pred, name):
    # Plotting the actual vs predicted values
    plt.figure(figsize=(15, 5))

    # Actual values - using blue color with a line marker
    plt.plot(y_actual.index, y_actual, color='blue', marker='o', label='Actual', linestyle='-', linewidth=1)

    # Predicted values - using red color with a cross marker
    plt.plot(y_actual.index, y_pred, color='red', marker='x', label='Predicted', linestyle='None')

    plt.title(f'{name} Set - Actual vs Predicted PM2.5')
    plt.xlabel('Date')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.grid()
    plt.show()


def split_data(df):
    data = df[ALL]
    # Assuming the dataset contains additional features other than PM2.5 values
    # Standardize the features before applying PCA
    features = data.drop([TARGET], axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95)  # Retain 95% of the variance
    principal_components = pca.fit_transform(features_scaled)

    # Add the principal components as new columns to the dataset
    for i in range(principal_components.shape[1]):
        data[f'PC{i + 1}'] = principal_components[:, i]

    # Now, use these principal components as exogenous variables in the ARIMAX model
    pm25_data = data[TARGET]
    exog_data = data.drop([TARGET], axis=1)

    data_length = len(pm25_data)

    # Calculate indices for 60%, 80% of the data
    train_end = int(data_length * 0.6)
    val_end = int(data_length * 0.8)

    # Split the data
    y_train, train_exog = pm25_data.iloc[:train_end], exog_data.iloc[:train_end]
    y_val, val_exog = pm25_data.iloc[train_end:val_end], exog_data.iloc[train_end:val_end]
    y_test, test_exog = pm25_data.iloc[val_end:], exog_data.iloc[val_end:]

    print(f"Training set size: {len(y_train)}")
    print(f"Validation set size: {len(y_val)}")
    print(f"Test set size: {len(y_test)}")

    return y_train, train_exog, y_val, val_exog, y_test, test_exog


def arimax_train_and_evolve(df, frequency='H'):
    y_train, train_exog, y_val, val_exog, y_test, test_exog = split_data(df)

    # Fit the ARIMAX model with PCA components as exogenous variables on the training data
    best_params = get_arimax_best_params(frequency)
    model = SARIMAX(y_train, exog=train_exog, order=best_params['order'],
                    seasonal_order=best_params['seasonal_order'])
    model_fit = model.fit()

    # summary for model fit
    print("===== Summary ======")
    print(model_fit.summary())

    print("===== Residuals ======")
    residuals = pd.DataFrame(model_fit.resid)
    print(residuals.describe())

    print("===== Evaluation ======")
    # Make predictions on the validation set
    val_predictions = model_fit.forecast(steps=len(y_val), exog=val_exog)

    # Error Metric
    mb.evolve_error_metrics(y_val, val_predictions)
    mb.naive_mean_absolute_scaled_error(y_val, val_predictions)

    # Make predictions on the test set
    test_predictions = model_fit.forecast(steps=len(y_test), exog=test_exog)

    # Error Metric
    mb.evolve_error_metrics(y_test, test_predictions)
    mb.naive_mean_absolute_scaled_error(y_test, test_predictions)

    plot_pm_true_predict(y_val, val_predictions, 'Validation')
    plot_pm_true_predict(y_test, test_predictions, 'Test')


def tune_arimax(df, max_p=7, max_d=5, max_q=7):
    best_mae = float('inf')
    best_params = None
    y_train, train_exog, y_val, val_exog, y_test, test_exog = split_data(df)

    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        try:
            print(f'Tuning for p={p}, d={d}, q={q}')
            model = SARIMAX(y_train, exog=train_exog, order=(p, d, q))
            model_fit = model.fit(disp=False)
            val_predictions = model_fit.forecast(steps=len(y_val), exog=val_exog)
            mae = mean_absolute_error(y_val, val_predictions)

            if mae < best_mae:
                best_mae = mae
                best_params = {'order': (p, d, q), 'seasonal_order': (0, 0, 0, 0)}

        except Exception as e:
            continue

    return best_params
