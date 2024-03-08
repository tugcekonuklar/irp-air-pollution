
class BaseModel:
    def __init__(self, name, model=None):
        self.name = name
        self.model = model

    def train(self, X_train, y_train):
        if self.model is None:
            raise NotImplementedError("Model not initialized.")
        else:
            pass

    def predict(self, X):
        if self.model is None:
            raise NotImplementedError("Model not initialized.")
        else:
            pass

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise NotImplementedError("Model not initialized.")
        else:
            pass

    def tune(self, X_train, y_train):
        if self.model is None:
            raise NotImplementedError("Model not initialized.")
        else:
            pass
