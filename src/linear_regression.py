from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform
from sklearn.linear_model import Lasso
from scipy.stats import loguniform
from sklearn.linear_model import LinearRegression

FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'


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
    # Define the distribution of hyperparameters to sample from
    # Here, we define a uniform distribution over a log scale for the alpha parameter
    param_distributions = {'alpha': uniform(1e-4, 1e4)}

    tscv = TimeSeriesSplit(n_splits=5)

    # Set up the random search with cross-validation
    # n_iter sets the number of different combinations to try
    # cv is the number of folds for cross-validation
    # scoring determines the metric used to evaluate the models
    return RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=100, cv=tscv,
                              scoring='neg_mean_absolute_error', verbose=1, random_state=42, n_jobs=-1)


def init_lasso_model_with_random():
    model = Lasso(max_iter=10000)  # Increased max_iter for convergence with high-dimensional data

    # Define the distribution of hyperparameters to sample from
    # loguniform is useful for alpha because it spans several orders of magnitude
    param_distributions = {'alpha': loguniform(1e-4, 1e0)}

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
