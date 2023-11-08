from sklearn.decomposition import PCA
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
import model_base as mb
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import randint as sp_randint, uniform

FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'

GRADIENT_BOOSTING = 'gradient_boosting'
HISTOGRAM_GRADIENT_BOOSTING = 'histogram_gradient_boosting'
XGBOOST = 'xgboost'
ADABOOST = 'adaboost'

GBR_MODEL = ('gbr', GradientBoostingRegressor(learning_rate=0.1,
                                              max_depth=3,
                                              min_samples_leaf=3,
                                              min_samples_split=6,
                                              n_estimators=406))
HIST_GBR_MODEL = ('hist_gbr', HistGradientBoostingRegressor(learning_rate=0.04410482473745831,
                                                            max_depth=9,
                                                            max_iter=373,
                                                            min_samples_leaf=23))
XGB_MODEL = ('xgb', xgb.XGBRegressor(n_estimators=205,
                                     max_depth=8,
                                     learning_rate=0.10351332282682328,
                                     subsample=0.7838501639099957,
                                     colsample_bytree=0.831261142176991,
                                     min_child_weight=6))
ADA_MODEL = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),
                              learning_rate=0.6892612104349699,
                              n_estimators=237)


def init_boosting_model(algorithm: str):
    if algorithm == GRADIENT_BOOSTING:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            GBR_MODEL
        ])
    elif algorithm == HISTOGRAM_GRADIENT_BOOSTING:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            HIST_GBR_MODEL
        ])
    elif algorithm == XGBOOST:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=4)),
            XGB_MODEL
        ])
    elif algorithm == ADABOOST:
        return ADA_MODEL
    else:
        raise ValueError("Unknown algorithm enum provided!")


def get_gb_param_distribution():
    return {
        'pca__n_components': [0.95, 0.99],
        'gbr__n_estimators': sp_randint(100, 500),
        'gbr__max_depth': sp_randint(3, 10),
        'gbr__min_samples_split': sp_randint(2, 11),
        'gbr__min_samples_leaf': sp_randint(1, 11),
        'gbr__learning_rate': [0.01, 0.02, 0.05, 0.1]
    }


def get_hist_gb_param_distribution():
    return {
        'pca__n_components': [0.95, 0.99],
        'hist_gbr__max_iter': sp_randint(100, 1000),
        'hist_gbr__max_depth': sp_randint(3, 30),
        'hist_gbr__min_samples_leaf': sp_randint(20, 100),
        'hist_gbr__learning_rate': uniform(0.01, 0.2)
    }


def get_xgb_param_distribution():
    return {
        'pca__n_components': sp_randint(3, 10),
        'xgb__n_estimators': sp_randint(100, 1000),
        'xgb__max_depth': sp_randint(3, 10),
        'xgb__learning_rate': uniform(0.01, 0.3),
        'xgb__min_child_weight': sp_randint(1, 10),
        'xgb__subsample': uniform(0.5, 0.5),
        'xgb__colsample_bytree': uniform(0.5, 0.5),
    }


def get_ada_param_distribution():
    # Hyperparameters to tune
    return {
        'adaboost__n_estimators': sp_randint(50, 500),
        'adaboost__learning_rate': np.logspace(-2, 0, 100),
        'adaboost__base_estimator__max_depth': sp_randint(1, 11)
    }


def get_param_distribution_by_algorithm(algorithm: str):
    if algorithm == GRADIENT_BOOSTING:
        return get_gb_param_distribution()
    elif algorithm == HISTOGRAM_GRADIENT_BOOSTING:
        return get_hist_gb_param_distribution()
    elif algorithm == XGBOOST:
        return get_xgb_param_distribution()
    elif algorithm == ADABOOST:
        return get_ada_param_distribution()
    else:
        raise ValueError("Unknown algorithm enum provided!")


def tune_and_evaluate_boosting(df, ensemble_alg: str):
    n_iter_search = 10
    random_state = 42

    # Define your features and target variable
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    print(f'Started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    model = init_boosting_model(ensemble_alg)

    # Define the parameter space for the grid search
    param_distributions = get_param_distribution_by_algorithm(ensemble_alg)

    # Create the TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=5)

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(model, param_distributions=param_distributions,
                                       n_iter=n_iter_search, scoring='neg_mean_squared_error',
                                       cv=tscv, n_jobs=1, random_state=random_state, verbose=1)

    print(f'Fitted {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # Fit the model on the training data
    try:
        random_search.fit(X_train, y_train)
    except Exception as ex:
        print(ex)

    # Print the best parameters and best score from the training
    print(f"Best parameters for {ensemble_alg} : {random_search.best_params_}")
    print(f"Best score {ensemble_alg} : {-random_search.best_score_}")

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


def train_and_evolve(df, ensemble_alg: str):
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    model = init_boosting_model(ensemble_alg)
    # Fit the model on training data
    model.fit(X_train, y_train)

    # VALIDATION Prediction and Evolution
    y_val_pred = model.predict(X_val)

    print(y_val_pred)

    # Validation Error Metric
    mb.evolve_error_metrics(y_val, y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)

    # TEST Prediction and Evolution
    y_test_pred = model.predict(X_test)

    print(y_test_pred)

    # Test Error Metric
    mb.evolve_error_metrics(y_test, y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)

    # Plot
    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')
