import numpy as np
import model_base as mb
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor


best_params = {
    'learning_rate': 0.1,
    'l2_leaf_reg': 3,
    'iterations': 200,
    'depth': 4
}

def init_cat_boosting_model():
    return CatBoostRegressor(
        learning_rate=best_params['learning_rate'],
        l2_leaf_reg=best_params['l2_leaf_reg'],
        iterations=best_params['iterations'],
        depth=best_params['depth'])


def get_param_distribution_by_algorithm():
    return {
        'depth': [4, 6, 8, 10],
        'learning_rate': np.logspace(-3, 0, 10),
        'iterations': [100, 200, 500, 1000],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }


def tune_and_evaluate(df):
    n_iter_search = 10
    random_state = 42

    # Define your features and target variable
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    print(f'Started {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    model = CatBoostRegressor(verbose=0, thread_count=-1)
    param_distributions = get_param_distribution_by_algorithm()

    # Create the TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=5)

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        scoring='neg_mean_squared_error',
        cv=tscv,
        n_jobs=1,
        random_state=random_state,
        verbose=1
    )

    print(f'Fitted {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # Fit the model on the training data
    try:
        random_search.fit(X_train, y_train)
    except Exception as ex:
        print(ex)

    # Print the best parameters and best score from the training
    print(f"Best parameters for  : {random_search.best_params_}")
    print(f"Best score : {-random_search.best_score_}")

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


def train_and_evolve(df):
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

    model = init_cat_boosting_model()
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
