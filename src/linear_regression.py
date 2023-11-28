from scipy.stats import loguniform
from scipy.stats import uniform
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def init_linear_model():
    return LinearRegression()


def init_ridge_model_with_random():
    model = Ridge()

    param_distributions = {'alpha': uniform(1e-4, 1e4)}

    tscv = TimeSeriesSplit(n_splits=5)

    return RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=100, cv=tscv,
                              scoring='neg_mean_absolute_error', verbose=1, random_state=42, n_jobs=-1)


def init_lasso_model_with_random():
    model = Lasso(max_iter=10000)

    param_distributions = {'alpha': loguniform(1e-4, 1e0)}

    tscv = TimeSeriesSplit(n_splits=5)

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
