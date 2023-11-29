from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform
from sklearn.linear_model import Lasso
from scipy.stats import loguniform
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import model_base as mb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime 
import numpy as np

#
# def split_data(x, y, train_size, val_size):
#     train_end = int(len(x) * train_size)
#     val_end = train_end + int(len(x) * val_size)
#     x_train, y_train = x.iloc[:train_end], y.iloc[:train_end]
#     x_val, y_val = x.iloc[train_end:val_end], y.iloc[train_end:val_end]
#     x_test, y_test = x.iloc[val_end:], y.iloc[val_end:]
#     return x_train, x_val, x_test, y_train, y_val, y_test
#

def init_linear_model():
    return LinearRegression()


def init_ridge_model_with_random():
    model = Ridge()
    param_distributions = {} #{'alpha': uniform(1e-4, 1e4)}

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

    print(y_val_pred)

    # Error Metric
    mb.evolve_error_metrics(y_val,y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val,y_val_pred)

    # Predict on the test set
    y_test_pred = model.predict(X_test_pca)

    # Error Metric
    mb.evolve_error_metrics(y_test,y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test,y_test_pred)


    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')
    
def lasso_train_and_evolve(df, model=init_lasso_model_with_random()):
    train_and_evolve(df, model)

def ringe_train_and_evolve(df, model=init_ridge_model_with_random()):
    train_and_evolve(df, model)

def init_svr_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # Retain 95% of the variance
        ('svr', SVR(C=10.0, epsilon=0.2, kernel='rbf',gamma=0.01))
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


def svr_train_and_evolve(df):
    ## Splitting Data 
    train_data, validation_data, test_data = mb.split_data(df)
    # Extract the features
    X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
    # Extract the target variable
    y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)
    
    # Initialize and train the SVR model
    pipeline = init_svr_pipeline()
    pipeline.fit(X_train, y_train)

    # # Make predictions on the validation set
    y_val_pred = pipeline.predict(X_val)

    print(y_val_pred)

    # # Error Metric
    mb.evolve_error_metrics(y_val,y_val_pred)
    mb.naive_mean_absolute_scaled_error(y_val,y_val_pred)

    # # Predict on the test set
    y_test_pred = pipeline.predict(X_test)

    # # Error Metric
    mb.evolve_error_metrics(y_test,y_test_pred)
    mb.naive_mean_absolute_scaled_error(y_test,y_test_pred)

    mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
    mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')
    
    
    
#{'svr__kernel': 'rbf', 'svr__gamma': 0.01, 'svr__epsilon': 1.0, 'svr__C': 10.0, 'pca__n_components': 0.95}

# Training set size: 52588
# Validation set size: 17529
# Test set size: 17531
# Started 2023-11-28 02:31:14
# Fitted 2023-11-28 02:31:14
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# Best parameters: {'svr__kernel': 'rbf', 'svr__gamma': 0.01, 'svr__epsilon': 1.0, 'svr__C': 10.0, 'pca__n_components': 0.95}
# Best score: 13.651469207046095
# Prediction started  2023-11-28 03:32:07
# MAE: 1.7569
# MSE: 9.3864
# RMSE: 3.0637
# MAPE: 0.2376
# MASE: 1.6950847864461795
# MAE: 1.7923
# MSE: 7.6795
# RMSE: 2.7712
# MAPE: 0.2271
# MASE: 1.659328815750067
# Finished 2023-11-28 03:32:48