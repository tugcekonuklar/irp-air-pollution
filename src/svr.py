from sklearn.decomposition import PCA
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import model_base as mb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'


def init_svr_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # Retain 95% of the variance
        ('svr', SVR(C=1.0, epsilon=0.2))
    ])


def tune_and_evaluate_svr(df):
    n_iter_search = 20
    random_state = 42

    # Define your features and target variable
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)

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

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions,
                                       n_iter=n_iter_search, scoring='neg_mean_squared_error',
                                       cv=5, random_state=random_state)

    # Fit the model on the training data
    random_search.fit(X_train, y_train)

    # Check the performance on the validation set
    y_val_pred = random_search.predict(X_val)
    mb.evolve_error_metrics(y_val, y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val, y_val_pred)

    # Print the best parameters and best score from the training
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {-random_search.best_score_}")

    # Use the best estimator to predict on the test set
    y_test_pred = random_search.best_estimator_.predict(X_test)
    mb.evolve_error_metrics(y_test, y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test, y_test_pred)

    # Return the best estimator and the MSE scores
    return random_search.best_estimator_
