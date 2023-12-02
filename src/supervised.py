from datetime import datetime

import numpy as np
from scipy.stats import loguniform
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import model_base as mb


def init_linear_model():
    return LinearRegression()


def init_ridge_model_with_random():
    model = Ridge()
    param_distributions = {}  # {'alpha': uniform(1e-4, 1e4)}

    tscv = TimeSeriesSplit(n_splits=5)

    return RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=100, cv=tscv,
                              scoring='neg_mean_absolute_error', verbose=1, random_state=42, n_jobs=-1)


def init_lasso_model_with_random():
    model = Lasso(max_iter=10000)  # Increased max_iter for convergence with high-dimensional data

    # Define the distribution of hyperparameters to sample from
    # loguniform is useful for alpha because it spans several orders of magnitude
    param_distributions = {'alpha': loguniform(1e-4, 1e4)}

    tscv = TimeSeriesSplit(n_splits=5)

    # Set up the random search with cross-validation
    return RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=100,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )


def train_and_evolve(df, model=init_linear_model()):
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


def lasso_train_and_evolve(df, model=init_lasso_model_with_random()):
    train_and_evolve(df, model)


def ringe_train_and_evolve(df, model=init_ridge_model_with_random()):
    train_and_evolve(df, model)


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


def tune_and_evaluate_svr(df):
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

