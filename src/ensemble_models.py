from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                              AdaBoostRegressor, RandomForestRegressor, VotingRegressor)
from sklearn.tree import DecisionTreeRegressor
from base_model import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
import model_base as mb


class BaseEnsembleModel(BaseModel):
    def __init__(self, name, model, best_params=None):
        self.model = model
        self.best_params = best_params
        self.best_estimator = None
        super().__init__(name, model)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mb.evolve_error_metrics(y_test, predictions)
        mb.naive_mean_absolute_scaled_error(y_test, predictions)
        return predictions

    def tune(self, X_train, y_train, param_distributions=None):
        if param_distributions is None:
            raise ValueError("Parameter distributions not defined.")
        n_iter = 10
        cv = TimeSeriesSplit(n_splits=5)
        tuner = RandomizedSearchCV(self.model, param_distributions=param_distributions,
                                   n_iter=n_iter, scoring='neg_mean_squared_error',
                                   cv=cv, n_jobs=1, random_state=42, verbose=1)

        tuner.fit(X_train, y_train)
        self.best_estimator = tuner.best_estimator_


class GradientBoostingModel(BaseEnsembleModel):
    def __init__(self, best_params=None, param_distributions=None):
        self.param_distributions = param_distributions
        self.model = self.init_model(best_params)
        super().__init__("GradientBoostingModel", self.model, best_params)

    def init_model(self, best_params):
        gbr = GradientBoostingRegressor(
            learning_rate=best_params.get('learning_rate', 0.1),
            max_depth=best_params.get('max_depth', 3),
            min_samples_leaf=best_params.get('min_samples_leaf', 1),
            min_samples_split=best_params.get('min_samples_split', 2),
            n_estimators=best_params.get('n_estimators', 100)
        )
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('gbr', gbr)
        ])
        return pipeline


class HistGradientBoostingModel(BaseEnsembleModel):
    def __init__(self, best_params=None, param_distributions=None):
        self.param_distributions = param_distributions
        self.model = self.init_model(best_params=best_params)
        super().__init__("HistGradientBoostingModel", self.model, best_params)

    def init_model(self, best_params):
        model = HistGradientBoostingRegressor(learning_rate=best_params.get('learning_rate', 0.1),
                                              max_depth=best_params.get('max_depth', 3),
                                              max_iter=best_params.get('max_iter', 100),
                                              min_samples_leaf=best_params.get('min_samples_leaf', 1))
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('hist_gbr', model)
        ])


class AdaBoostingModel(BaseEnsembleModel):
    def __init__(self, best_params=None, param_distributions=None):
        self.param_distributions = param_distributions
        self.model = self.init_model(best_params)
        super().__init__("AdaBoostingModel", self.model, best_params)

    def init_model(self, best_params):
        model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=best_params.get('max_depth', 3)),
                                  learning_rate=best_params.get('learning_rate', 0.1),
                                  n_estimators=best_params.get('n_estimators', 100))
        return Pipeline([
            ('scaler', StandardScaler()),
            ('adaboost', model)
        ])


class RandomForestModel(BaseEnsembleModel):
    def __init__(self, best_params=None, param_distributions=None):
        self.param_distributions = param_distributions
        self.model = self.init_model(best_params)
        super().__init__("RandomForestModel", self.model, best_params)

    def init_model(self, best_params):
        return RandomForestRegressor(
            n_estimators=best_params.get('n_estimators', 100),
            max_depth=best_params.get('max_depth', 3),
            min_samples_leaf=best_params.get('min_samples_leaf', 7),
            min_samples_split=best_params.get('min_samples_split', 9),
            max_features=best_params.get('max_features', 'log2')
        )


class XGBoostModel(BaseEnsembleModel):
    def __init__(self, best_params=None, param_distributions=None):
        self.param_distributions = param_distributions
        self.model = self.init_model(best_params)
        super().__init__("XGBoostModel", self.model, best_params)

    def init_model(self, best_params):
        model = XGBRegressor(n_estimators=best_params.get('n_estimators', 100),
                             max_depth=best_params.get('max_depth', 3),
                             learning_rate=best_params.get('learning_rate', 0.1),
                             subsample=best_params.get('subsample', 1),
                             colsample_bytree=best_params.get('colsample_bytree', 1),
                             min_child_weight=best_params.get('min_child_weight', 1))
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=4)),
            ('xgb', model),
        ])


class CatBoostModel(BaseEnsembleModel):
    def __init__(self, best_params=None, param_distributions=None):
        self.param_distributions = param_distributions
        self.model = self.init_model(best_params)
        super().__init__("CatBoostModel", self.model, best_params)

    def init_model(self, best_params):
        return CatBoostRegressor(learning_rate=best_params.get('learning_rate', 0.1),
                                 l2_leaf_reg=best_params.get('l2_leaf_reg', 3),
                                 iterations=best_params.get('iterations', 200),
                                 depth=best_params.get('depth', 4))
