import model_base as mb
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import itertools
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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


def init_linear_model():
    return LinearRegression()


def linear_train_and_evolve(df, model=init_linear_model()):
    train_data, validation_data, test_data = mb.split_data(df)

    # Scale the features
    X_train_scaled, X_val_scaled, X_test_scaled = mb.scale_features(train_data, validation_data, test_data)
    # Apply PCA on the scaled data
    pca = mb.init_pca()
    pca.fit(X_train_scaled)
    # Transform the datasets using the fitted PCA
    X_train_pca = pca.transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    model.fit(X_train_pca, y_train)
    # Make predictions on the validation set
    y_val_pred = model.predict(X_val_pca)

    #     print(y_val_pred)

    # Error Metric
    mb.evolve_error_metrics(y_val, y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)

    # Predict on the test set
    y_test_pred = model.predict(X_test_pca)

    # Error Metric
    mb.evolve_error_metrics(y_test, y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)

    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')


def get_svr_best_params(frequency):
    """
    Returns the best parameters for Hist Gradient Boosting Regressor based on the specified frequency.

    Args:
    - frequency (str): The frequency for which to get the best parameters. Options are 'H', 'D', 'W', 'M'.

    Returns:
    - dict: A dictionary of the best parameters.
    """
    # Define best parameters for each frequency
    best_params = {
        'H': {
            'C': 10.0,
            'epsilon': 1.0,
            'kernel': 'rbf',
            'gamma': 0.01
        },
        'D': {
            'C': 1.0,
            'epsilon': 0.01,
            'kernel': 'linear',
            'gamma': 1.0
        },
        'W': {
            'C': 10.0,
            'epsilon': 0.01,
            'kernel': 'linear',
            'gamma': 0.001
        },
        'M': {
            'C': 10.0,
            'epsilon': 0.01,
            'kernel': 'linear',
            'gamma': 0.001
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def init_svr_pipeline(frequency):
    best_params = get_svr_best_params(frequency)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # Retain 95% of the variance
        ('svr', SVR(C=best_params['C'], epsilon=best_params['epsilon'], kernel=best_params['kernel'],
                    gamma=best_params['gamma']))
    ])


def svr_tune_and_evaluate(df):
    n_iter_search = 10
    random_state = 42

    # Define your features and target variable
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    print(f'Started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # Define a pipeline combining a scaler, PCA, and SVR
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('svr', SVR())
    ])

    # Define the parameter space for the grid search
    param_distributions = {
        'pca__n_components': [0.85, 0.90, 0.95],
        'svr__C': np.logspace(-3, 3, 7),
        'svr__epsilon': np.logspace(-3, 0, 4),
        'svr__kernel': ['linear', 'poly', 'rbf'],
        'svr__gamma': np.logspace(-3, 1, 5)  # Relevant for 'rbf', 'poly' and 'sigmoid'
    }

    # Create the TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=5)

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions,
                                       n_iter=n_iter_search, scoring='neg_mean_squared_error',
                                       cv=tscv, n_jobs=1, random_state=random_state, verbose=1)

    print(f'Fitted {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # Fit the model on the training data
    try:
        random_search.fit(X_train, y_train)
    except Exception as ex:
        print(ex)

    # Print the best parameters and best score from the training
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {-random_search.best_score_}")

    print(f'Prediction started  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # Check the performance on the validation set
    y_val_pred = random_search.predict(X_val)
    mb.evolve_error_metrics(y_val, y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)

    # Use the best estimator to predict on the test set
    y_test_pred = random_search.best_estimator_.predict(X_test)
    mb.evolve_error_metrics(y_test, y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)

    print(f'Finished {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # Return the best estimator and the MSE scores
    return random_search.best_estimator_


def svr_train_and_evolve(df, frequency='H'):
    ## Splitting Data
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    # Initialize and train the SVR model
    pipeline = init_svr_pipeline(frequency)
    pipeline.fit(X_train, y_train)

    # # Make predictions on the validation set
    y_val_pred = pipeline.predict(X_val)

    #     print(y_val_pred)

    # # Error Metric
    mb.evolve_error_metrics(y_val, y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)

    # # Predict on the test set
    y_test_pred = pipeline.predict(X_test)

    # # Error Metric
    mb.evolve_error_metrics(y_test, y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)

    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')
