from datetime import datetime

import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.stats import randint as sp_randint, uniform
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, \
    RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import model_base as mb

GRADIENT_BOOSTING = 'gradient_boosting'
HISTOGRAM_GRADIENT_BOOSTING = 'histogram_gradient_boosting'
XGBOOST = 'xgboost'
ADABOOST = 'adaboost'
CATBOOST = 'catboost'
RANDOMFOREST = 'randomforest'


def get_gbr_best_params(frequency):
    """
    Returns the best parameters for Gradient Boosting Regressor based on the specified frequency.

    Args:
    - frequency (str): The frequency for which to get the best parameters. Options are 'H', 'D', 'W', 'M'.

    Returns:
    - dict: A dictionary of the best parameters.
    """
    # Define best parameters for each frequency
    best_params = {
        'H': {
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_samples_leaf': 8,
            'min_samples_split': 6,
            'n_estimators': 202
        },
        'D': {
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_samples_leaf': 8,
            'min_samples_split': 6,
            'n_estimators': 202
        },
        'W': {
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_leaf': 6,
            'min_samples_split': 6,
            'n_estimators': 357
        },
        'M': {
            'learning_rate': 0.1,
            'max_depth': 8,
            'min_samples_leaf': 2,
            'min_samples_split': 10,
            'n_estimators': 445
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def get_hist_gbr_best_params(frequency):
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
            'learning_rate': 0.04410482473745831,
            'max_depth': 9,
            'max_iter': 373,
            'min_samples_leaf': 23
        },
        'D': {
            'learning_rate': 0.04410482473745831,
            'max_depth': 9,
            'max_iter': 373,
            'min_samples_leaf': 23
        },
        'W': {
            'learning_rate': 0.04410482473745831,
            'max_depth': 9,
            'max_iter': 373,
            'min_samples_leaf': 23
        },
        'M': {
            'learning_rate': 0.04410482473745831,
            'max_depth': 9,
            'max_iter': 373,
            'min_samples_leaf': 23
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def get_xgb_best_params(frequency):
    """
    Returns the best parameters for XGBoost Regressor based on the specified frequency.

    Args:
    - frequency (str): The frequency for which to get the best parameters. Options are 'H', 'D', 'W', 'M'.

    Returns:
    - dict: A dictionary of the best parameters.
    """
    # Define best parameters for each frequency
    best_params = {
        'H': {
            'n_estimators': 714,
            'max_depth': 7,
            'learning_rate': 0.06503043695984914,
            'subsample': 0.7229163764267956,
            'colsample_bytree': 0.8982714934301164,
            'min_child_weight': 5
        },
        'D': {
            'n_estimators': 714,
            'max_depth': 7,
            'learning_rate': 0.06503043695984914,
            'subsample': 0.7229163764267956,
            'colsample_bytree': 0.8982714934301164,
            'min_child_weight': 5
        },
        'W': {
            'n_estimators': 714,
            'max_depth': 7,
            'learning_rate': 0.06503043695984914,
            'subsample': 0.7229163764267956,
            'colsample_bytree': 0.8982714934301164,
            'min_child_weight': 5
        },
        'M': {
            'n_estimators': 205,
            'max_depth': 7,
            'learning_rate': 0.06503043695984914,
            'subsample': 0.7229163764267956,
            'colsample_bytree': 0.8982714934301164,
            'min_child_weight': 5
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def get_ada_best_params(frequency):
    """
    Returns the best parameters for XGBoost Regressor based on the specified frequency.

    Args:
    - frequency (str): The frequency for which to get the best parameters. Options are 'H', 'D', 'W', 'M'.

    Returns:
    - dict: A dictionary of the best parameters.
    """
    # Define best parameters for each frequency
    best_params = {
        'H': {
            'max_depth': 8,
            'learning_rate': 0.16297508346206444,
            'n_estimators': 70,
        },
        'D': {
            'max_depth': 8,
            'learning_rate': 0.010974987654930561,
            'n_estimators': 199,
        },
        'W': {
            'max_depth': 5,
            'learning_rate': 0.044306214575838825,
            'n_estimators': 363,
        },
        'M': {
            'max_depth': 8,
            'learning_rate': 0.010974987654930561,
            'n_estimators': 199,
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def get_cat_best_params(frequency):
    """
    Returns the best parameters for CatBoost Regressor based on the specified frequency.

    Args:
    - frequency (str): The frequency for which to get the best parameters. Options are 'H', 'D', 'W', 'M'.

    Returns:
    - dict: A dictionary of the best parameters.
    """
    # Define best parameters for each frequency
    best_params = {
        'H': {
            'l2_leaf_reg': 3,
            'learning_rate': 0.09999999999999999,
            'iterations': 200,
            'depth': 4
        },
        'D': {
            'l2_leaf_reg': 3,
            'learning_rate': 0.09999999999999999,
            'iterations': 200,
            'depth': 4
        },
        'W': {
            'l2_leaf_reg': 3,
            'learning_rate': 0.09999999999999999,
            'iterations': 200,
            'depth': 4
        },
        'M': {
            'l2_leaf_reg': 9,
            'learning_rate': 0.09999999999999999,
            'iterations': 500,
            'depth': 6
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


def get_random_forest_best_params(frequency):
    """
    Returns the best parameters for Random Forest Regressor based on the specified frequency.

    Args:
    - frequency (str): The frequency for which to get the best parameters. Options are 'H', 'D', 'W', 'M'.

    Returns:
    - dict: A dictionary of the best parameters.
    """
    # Define best parameters for each frequency
    best_params = {
        'H': {
            'max_depth': 28,
            'n_estimators': 472,
            'max_features': 'log2',
            'min_samples_leaf': 7,
            'min_samples_split': 9,
        },
        'D': {
            'max_depth': 9,
            'n_estimators': 714,
            'max_features': None,
            'min_samples_leaf': 8,
            'min_samples_split': 6,
        },
        'W': {
            'max_depth': 21,
            'n_estimators': 266,
            'max_features': 'log2',
            'min_samples_leaf': 5,
            'min_samples_split': 10,
        },
        'M': {
            'max_depth': 21,
            'n_estimators': 266,
            'max_features': 'log2',
            'min_samples_leaf': 5,
            'min_samples_split': 10,
        }
    }

    # Return the best parameters for the specified frequency
    return best_params.get(frequency, "Invalid frequency")


# best_params = {
#     'max_depth': 21,
#     'max_features': 'log2',
#     'min_samples_leaf': 5,
#     'min_samples_split': 10,
#     'n_estimators': 266
# }

def get_gbr_model(frequency):
    best_params = get_gbr_best_params(frequency)
    return ('gbr', GradientBoostingRegressor(learning_rate=best_params['learning_rate'],
                                             max_depth=best_params['max_depth'],
                                             min_samples_leaf=best_params['min_samples_leaf'],
                                             min_samples_split=best_params['min_samples_split'],
                                             n_estimators=best_params['n_estimators']))


def get_hist_gbr_model(frequency):
    best_params = get_hist_gbr_best_params(frequency)
    return ('hist_gbr', HistGradientBoostingRegressor(learning_rate=best_params['learning_rate'],
                                                      max_depth=best_params['max_depth'],
                                                      max_iter=best_params['max_iter'],
                                                      min_samples_leaf=best_params['min_samples_leaf']))


def get_xgb_model(frequency):
    best_params = get_xgb_best_params(frequency)
    return ('xgb', xgb.XGBRegressor(n_estimators=best_params['n_estimators'],
                                    max_depth=best_params['max_depth'],
                                    learning_rate=best_params['learning_rate'],
                                    subsample=best_params['subsample'],
                                    colsample_bytree=best_params['colsample_bytree'],
                                    min_child_weight=best_params['min_child_weight']))


def get_ada_model(frequency):
    best_params = get_ada_best_params(frequency)
    return AdaBoostRegressor(DecisionTreeRegressor(max_depth=best_params['max_depth']),
                             learning_rate=best_params['learning_rate'],
                             n_estimators=best_params['n_estimators'])


def get_cat_model(frequency):
    best_params = get_cat_best_params(frequency)
    return CatBoostRegressor(learning_rate=best_params['learning_rate'],
                             l2_leaf_reg=best_params['l2_leaf_reg'],
                             iterations=best_params['iterations'],
                             depth=best_params['depth'])


def get_random_forest_model(frequency):
    best_params = get_random_forest_best_params(frequency)
    return RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_leaf=best_params['min_samples_leaf'],
        min_samples_split=best_params['min_samples_split'],
        max_features=best_params['max_features'],
        # random_state=42  # Use a fixed seed for reproducibility
    )


# GBR_MODEL = ('gbr', GradientBoostingRegressor(learning_rate=0.1,
#                                               max_depth=3,
#                                               min_samples_leaf=3,
#                                               min_samples_split=6,
#                                               n_estimators=406))
# HIST_GBR_MODEL = ('hist_gbr', HistGradientBoostingRegressor(learning_rate=0.04410482473745831,
#                                                             max_depth=9,
#                                                             max_iter=373,
#                                                             min_samples_leaf=23))
# XGB_MODEL = ('xgb', xgb.XGBRegressor(n_estimators=205,
#                                      max_depth=8,
#                                      learning_rate=0.10351332282682328,
#                                      subsample=0.7838501639099957,
#                                      colsample_bytree=0.831261142176991,
#                                      min_child_weight=6))
# ADA_MODEL = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8),
#                               learning_rate=0.16297508346206444,
#                               n_estimators=70)

# CAT_BOOSTING = CatBoostRegressor(
#     learning_rate=0.1,
#     l2_leaf_reg=3,
#     iterations=200,
#     depth=4)

# RANDOM_FOREST = RandomForestRegressor(
#     n_estimators=21,
#     max_depth='log2',
#     min_samples_leaf=5,
#     min_samples_split=10,
#     max_features=266,
#     # random_state=42  # Use a fixed seed for reproducibility
# )


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


def get_catboost_distribution():
    return {
        'depth': [4, 6, 8, 10],
        'learning_rate': np.logspace(-3, 0, 10),
        'iterations': [100, 200, 500, 1000],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }


def get_randomforest_distribution():
    return {
        'n_estimators': sp_randint(100, 1000),
        'max_depth': sp_randint(3, 30),
        'min_samples_split': sp_randint(2, 11),
        'min_samples_leaf': sp_randint(1, 11),
        'max_features': ['auto', 'sqrt', 'log2', None]
    }


def init_ensemble_model(algorithm: str, frequency='H'):
    if algorithm == GRADIENT_BOOSTING:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            get_gbr_model(frequency)
        ])
    elif algorithm == HISTOGRAM_GRADIENT_BOOSTING:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            get_hist_gbr_model(frequency)
        ])
    elif algorithm == XGBOOST:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=4)),
            get_xgb_model(frequency)
        ])
    elif algorithm == ADABOOST:
        return get_ada_model(frequency)
    elif algorithm == CATBOOST:
        return get_cat_model(frequency)
    elif algorithm == RANDOMFOREST:
        return get_random_forest_model(frequency)
    else:
        raise ValueError("Unknown algorithm enum provided!")


def get_param_distribution_by_algorithm(algorithm: str):
    if algorithm == GRADIENT_BOOSTING:
        return get_gb_param_distribution()
    elif algorithm == HISTOGRAM_GRADIENT_BOOSTING:
        return get_hist_gb_param_distribution()
    elif algorithm == XGBOOST:
        return get_xgb_param_distribution()
    elif algorithm == ADABOOST:
        return get_ada_param_distribution()
    elif algorithm == CATBOOST:
        return get_catboost_distribution()
    elif algorithm == RANDOMFOREST:
        return get_randomforest_distribution()
    else:
        raise ValueError("Unknown algorithm enum provided!")


def tune_and_evaluate(df, ensemble_alg: str, frequency='H'):
    n_iter_search = 10
    random_state = 42

    # Define your features and target variable
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    print(f'Started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    model = None
    if ensemble_alg == ADABOOST:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('adaboost', AdaBoostRegressor(DecisionTreeRegressor(), random_state=42))
        ])
    elif ensemble_alg == RANDOMFOREST:
        model = RandomForestRegressor()
    else:
        model = init_ensemble_model(ensemble_alg, frequency)

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


def train_and_evolve(df, ensemble_alg: str, frequency='H'):
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    model = init_ensemble_model(ensemble_alg, frequency)
    # Fit the model on training data
    model.fit(X_train, y_train)

    # VALIDATION Prediction and Evolution
    y_val_pred = model.predict(X_val)

    # print(y_val_pred)

    # Validation Error Metric
    mb.evolve_error_metrics(y_val, y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)

    # TEST Prediction and Evolution
    y_test_pred = model.predict(X_test)

    # print(y_test_pred)

    # Test Error Metric
    mb.evolve_error_metrics(y_test, y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)

    # Plot
    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')


# best_params = {
#     'max_depth': 21,
#     'max_features': 'log2',
#     'min_samples_leaf': 5,
#     'min_samples_split': 10,
#     'n_estimators': 266
# }


# def init_random_forest_model():
#     return RandomForestRegressor(
#         n_estimators=best_params['n_estimators'],
#         max_depth=best_params['max_depth'],
#         min_samples_leaf=best_params['min_samples_leaf'],
#         min_samples_split=best_params['min_samples_split'],
#         max_features=best_params['max_features'],
#         # random_state=42  # Use a fixed seed for reproducibility
#     )


def train_and_evolve_bagging(df, frequency='H'):
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    model = get_random_forest_model(frequency)
    # Fit the model on training data
    model.fit(X_train, y_train)

    # VALIDATION Prediction and Evolution
    y_val_pred = model.predict(X_val)

    #  print(y_val_pred)

    # Validation Error Metric
    mb.evolve_error_metrics(y_val, y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)

    # TEST Prediction and Evolution
    y_test_pred = model.predict(X_test)

    # print(y_test_pred)

    # Test Error Metric
    mb.evolve_error_metrics(y_test, y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)

    # Plot
    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')
