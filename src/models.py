from sklearn.ensemble import RandomForestRegressor
import utils as utils


class BaseModel:
    def __init__(self, input_shape, output_shape, config=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.config = config
        self.model = None

    def build_model(self):
        raise NotImplementedError("Method 'build_model' not implemented")

    def train(self, X_train, y_train, epochs, batch_size, validation_data):
        raise NotImplementedError("Method 'train' not implemented")

    def evaluate(self, y, y_pred):
        raise NotImplementedError("Method 'evaluate' not implemented")

    def plot_performance(self, data, y_pred, name='Validation'):
        raise NotImplementedError("Method 'plot_performance' not implemented")

    def aggregate_models(self, models):
        raise NotImplementedError("Method 'aggregate_models' not implemented")

    def tune(self):
        raise NotImplementedError("Method 'tune' not implemented")


class RandomForestModel(BaseModel):
    def __init__(self, input_shape, output_shape, config=None):
        super().__init__(input_shape, output_shape, config)
        self.default_config = {
            'max_depth': 28,
            'n_estimators': 472,
            'max_features': 'log2',
            'min_samples_leaf': 7,
            'min_samples_split': 9,
        }
        if config is None:
            self.config = self.default_config
        else:
            self.config.update(config)

    def build_model(self):
        self.model = RandomForestRegressor(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['n_estimators'],
            max_features=self.config['max_features'],
            min_samples_leaf=self.config['min_samples_leaf'],
            min_samples_split=self.config['min_samples_split'],
        )

    def train(self, X_train, y_train, epochs=None, batch_size=None, validation_data=None):
        self.model.fit(X_train, y_train)

    def evaluate(self, y, y_pred):
        utils.evolve_error_metrics(y, y_pred)
        utils.naive_mean_absolute_scaled_error(y, y_pred)

    def plot_performance(self, data, y_pred, name='Validation'):
        utils.plot_pm_true_predict(data, y_pred, name)

    def tune(self):
        # You can implement hyperparameter tuning here
        pass
