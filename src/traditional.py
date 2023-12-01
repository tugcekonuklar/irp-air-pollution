import pmdarima as pm
import model_base as mb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import numpy as np
from sklearn.metrics import mean_squared_error

FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'


def init_auto_model(df):
    X, y = mb.define_target_features(df)
    return pm.auto_arima(y.values, start_p=1, start_q=1,
                         test='adf',  # use adftest to find optimal 'd'
                         max_p=10, max_q=10,  # maximum p and q
                         m=1,  # frequency of series
                         d=None,  # let model determine 'd'
                         seasonal=False,  # No Seasonality
                         start_P=0,
                         D=0,
                         trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True,
                         exogenous=X)


def init_arimax_model(df):
    # seasonal_order=(0, 0, 0, 0 means no seasonal
    X, y = mb.define_target_features(df)
    return SARIMAX(y, exog=X, order=(6, 0, 1), seasonal_order=(0, 0, 0, 0))



def init_exponential_smooting_model(target_df):
    # return SimpleExpSmoothing(df[TARGET]).fit(smoothing_level=0.2, optimized=False)
    return ExponentialSmoothing(np.asarray(target_df), trend='add', seasonal_periods=12, seasonal=None,
                                damped=False, use_boxcox=True).fit()


def arimax_train_and_evolve(df):
    train_data, validation_data, test_data = mb.split_data(df)
    # Scale the features
    X_train_scaled, X_val_scaled, X_test_scaled = mb.scale_features(train_data, validation_data, test_data)
    # Apply PCA on the scaled data
    pca = mb.init_pca()
    pca.fit(X_train_scaled)
    # Transform the datasets using the fitted PCA
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    # model creation
    model = init_arimax_model(df)
    fitted_model = model.fit()

    # evaluation
    # Make predictions on the validation set
    y_val_pred = fitted_model.get_prediction(start=y_val.index[0], end=y_val.index[-1], exog=X_val_pca)
    val_predictions_mean = y_val_pred.predicted_mean

    print(y_val_pred)
    print(val_predictions_mean)

    # Error Metric
    mb.evolve_error_metrics(y_val,val_predictions_mean)
    mb.naive_mean_absolute_scaled_error(y_val,val_predictions_mean)

    # Predict on the test set
    y_test_pred = fitted_model.get_prediction(start=y_test.index[0], end=y_test.index[-1], exog=X_test_pca)
    test_predictions_mean = y_test_pred.predicted_mean

    print(y_val_pred)
    print(test_predictions_mean)

    # Error Metric
    mb.evolve_error_metrics(y_test,test_predictions_mean)
    mb.naive_mean_absolute_scaled_error(y_test,test_predictions_mean)

    mb.plot_pm_true_predict(validation_data, val_predictions_mean, 'Validation')
    mb.plot_pm_true_predict(test_data, test_predictions_mean, 'Test')
    # mb.plot_time_series(train_data, y_train, validation_data, y_val, val_predictions_mean, y_val_pred, 'Validation')
    # mb.plot_time_series(train_data, y_train, test_data, y_test, test_predictions_mean, y_test_pred, 'Test')
    
    

def expomnential_smooting_train_and_evolve(df):
    
    df = df.resample('H').mean()  # Resample to hourly data if not already
    df = df.sort_index()

    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    # model creation
    model = init_exponential_smooting_model(y_train)

    # evaluation
    # Make predictions on the validation set
    y_val_pred = model.forecast(len(y_val))

    print(y_val_pred)

    # Error Metric
    mb.evolve_error_metrics(y_val,y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val,y_val_pred)

    # Predict on the test set
    y_test_pred = model.forecast(len(y_test))

    print(y_test_pred)

    # Error Metric
    mb.evolve_error_metrics(y_test,y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test,y_test_pred)

    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')
    # mb.plot_time_series(train_data, y_train, validation_data, y_val, val_predictions_mean, y_val_pred, 'Validation')
    # mb.plot_time_series(train_data, y_train, test_data, y_test, test_predictions_mean, y_test_pred, 'Test')
    
    
from itertools import product
from statsmodels.tsa.api import ExponentialSmoothing


def optimize_exponential_smoothing(df, trend_options=['add', 'mul', None], seasonal_options=['add', 'mul', None], seasonal_periods_options=[12], damped_options=[True, False]):
    """
    Optimize Exponential Smoothing model based on Mean Squared Error.

    :param train_data: Training dataset
    :param validation_data: Validation dataset
    :param trend_options: List of trend options
    :param seasonal_options: List of seasonal options
    :param seasonal_periods_options: List of seasonal period options
    :param damped_options: List of damped trend options
    :return: Best model configuration and its MSE
    """
    
    df = df.resample('H').mean()  # Resample to hourly data if not already
    df = df.sort_index()
    
    
    train_data, validation_data, test_data = mb.split_data(df)
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)
    
    # Cartesian product of the hyperparameter grid
    param_grid = list(product(trend_options, seasonal_options, seasonal_periods_options, damped_options))

    # Keep track of the best configuration and corresponding MSE
    best_mse = float("inf")
    best_config = None
    best_model = None

    # Perform grid search
    for params in param_grid:
        trend, seasonal, seasonal_periods, damped = params

        # Skip if both trend and seasonal are None
        if trend is None and seasonal is None:
            continue

        try:
            # Fit the model with the current set of hyperparameters
            model = ExponentialSmoothing(
                y_train, 
                seasonal_periods=seasonal_periods, 
                trend=trend, 
                seasonal=seasonal,
                damped_trend=damped,
                use_boxcox=True
            ).fit()

            # Forecast on the validation set
            val_predictions = model.forecast(len(y_val))

            # Calculate the MSE for this model configuration
            mse = mean_squared_error(y_val, val_predictions)

            # Check if this configuration gives us a lower MSE than what we've seen so far
            if mse < best_mse:
                best_mse = mse
                best_config = params
                best_model = model

        except Exception as e:
            print(f"Error with configuration {params}: {e}")

    # Output the best performing model configuration
    print(f"Best configuration: Trend: {best_config[0]}, Seasonal: {best_config[1]}, Seasonal Periods: {best_config[2]}, Damped: {best_config[3]}")
    print(f"Best MSE: {best_mse}")

    return best_model, best_mse
