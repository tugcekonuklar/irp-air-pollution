from datetime import datetime

import numpy as np
from scipy.stats import randint as sp_randint, uniform
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import model_base as mb
import traditional as td
from src.models.ensemble_models import GradientBoostingModel, HistGradientBoostingModel, AdaBoostingModel, RandomForestModel, \
    XGBoostModel, CatBoostModel
from enums import EnsembleModels


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


def get_param_distribution_by_algorithm(algorithm: str):
    """
    Returns the parameter distribution for a given algorithm.

    Args:
        algorithm (str): The identifier of the algorithm.

    Returns:
        dict: The parameter distribution for the specified algorithm.

    Raises:
        ValueError: If an unknown algorithm identifier is provided.
    """
    if algorithm == EnsembleModels.GRADIENT_BOOSTING:
        return get_gb_param_distribution()
    elif algorithm == EnsembleModels.HISTOGRAM_GRADIENT_BOOSTING:
        return get_hist_gb_param_distribution()
    elif algorithm == EnsembleModels.XGBOOST:
        return get_xgb_param_distribution()
    elif algorithm == EnsembleModels.ADABOOST:
        return get_ada_param_distribution()
    elif algorithm == EnsembleModels.CATBOOST:
        return get_catboost_distribution()
    elif algorithm == EnsembleModels.RANDOM_FOREST:
        return get_randomforest_distribution()
    else:
        raise ValueError("Unknown algorithm enum provided!")


def tune_and_evaluate(df, ensemble_alg: str, frequency='H'):
    """
    Tunes and evaluates an ensemble model based on specified algorithm and frequency.

    Args:
        df (DataFrame): The input dataset.
        ensemble_alg (str): The ensemble algorithm identifier (e.g., 'ADABOOST', 'RANDOMFOREST').
        frequency (str): The data frequency, used for custom model initialization.
    """
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
    if ensemble_alg == EnsembleModels.ADABOOST:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('adaboost', AdaBoostRegressor(DecisionTreeRegressor(), random_state=42))
        ])
    elif ensemble_alg == EnsembleModels.RANDOM_FOREST:
        model = RandomForestRegressor()
    else:
        model = init_ensemble_model(ensemble_alg, frequency).model

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


def init_ensemble_model(algorithm_str: str, frequency='H'):
    """
    Initializes an ensemble model based on the specified algorithm and data frequency.

    Args:
        algorithm_str (str): Identifier of the ensemble algorithm to initialize.
        frequency (str): Data frequency, used to tailor the model configuration.

    Returns:
        A configured model pipeline.

    Raises:
        ValueError: If an unknown algorithm identifier is provided.
    """
    algorithm = EnsembleModels.get_by_value(algorithm_str)
    if algorithm == EnsembleModels.GRADIENT_BOOSTING:
        return GradientBoostingModel(best_params=get_gbr_best_params(frequency))
    elif algorithm == EnsembleModels.HISTOGRAM_GRADIENT_BOOSTING:
        return HistGradientBoostingModel(best_params=get_hist_gbr_best_params(frequency))
    elif algorithm == EnsembleModels.XGBOOST:
        return XGBoostModel(best_params=get_xgb_best_params(frequency))
    elif algorithm == EnsembleModels.ADABOOST:
        return AdaBoostingModel(best_params=get_ada_best_params(frequency))
    elif algorithm == EnsembleModels.CATBOOST:
        return CatBoostModel(best_params=get_cat_best_params(frequency))
    elif algorithm == EnsembleModels.RANDOM_FOREST:
        return RandomForestModel(best_params=get_random_forest_best_params(frequency))
    else:
        raise ValueError("Unknown algorithm enum provided!")


def train_and_evolve(df, ensemble_alg: str, frequency='H'):
    """
    Trains a ensemble regressor model using various ensemble and regression techniques,
    evaluates its performance on validation and test datasets, and saves the model.

    Args:
        df (DataFrame): The input dataset.
        ensemble_alg (str): algorithm's name
        frequency (str): The frequency of the dataset, used to tailor model initialization.
    """
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)
    model = init_ensemble_model(ensemble_alg, frequency)
    # Fit the model on training data
    model.train(X_train, y_train)
    # VALIDATION Prediction and Evolution
    y_val_pred = model.evaluate(X_val, y_val)
    # TEST Prediction and Evolution
    y_test_pred = model.evaluate(X_test, y_test)
    # Plot
    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')
    # mb.save_model_to_pickle(model, f'{ensemble_alg}_model_{frequency}.pkl')


def voting_train_and_evolve(df, frequency='H'):
    """
    Trains a voting regressor model using various ensemble and regression techniques,
    evaluates its performance on validation and test datasets, and saves the model.

    Args:
        df (DataFrame): The input dataset.
        frequency (str): The frequency of the dataset, used to tailor model initialization.
    """
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)
    model_gb = init_ensemble_model(EnsembleModels.GRADIENT_BOOSTING.value, frequency).model
    model_hist_gb = init_ensemble_model(EnsembleModels.HISTOGRAM_GRADIENT_BOOSTING.value, frequency).model
    model_xgb = init_ensemble_model(EnsembleModels.XGBOOST.value, frequency).model
    model_ada = init_ensemble_model(EnsembleModels.ADABOOST.value, frequency).model
    model_cat = init_ensemble_model(EnsembleModels.CATBOOST.value, frequency).model
    model_rf = init_ensemble_model(EnsembleModels.RANDOM_FOREST.value, frequency).model
    model_svr = td.init_svr(frequency)
    model_lr = td.init_linear_model()

    # Create the voting regressor
    model = VotingRegressor(
        estimators=[('gb', model_gb),
                    ('hist_gb', model_hist_gb),
                    ('xgb', model_xgb),
                    ('ada', model_ada),
                    ('cat', model_cat),
                    ('rf', model_rf),
                    ('svr', model_svr),
                    ('mlr', model_lr)]
    )

    model.fit(X_train, y_train)

    # VALIDATION Prediction and Evolution
    y_val_pred = model.predict(X_val)

    # Validation Error Metric
    mb.evolve_error_metrics(y_val, y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)

    # TEST Prediction and Evolution
    y_test_pred = model.predict(X_test)

    # Test Error Metric
    mb.evolve_error_metrics(y_test, y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)

    # Plot
    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')

    mb.save_model_to_pickle(model, f'voting_model_{frequency}.pkl')
