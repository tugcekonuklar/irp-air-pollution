from datetime import datetime
from sklearn.ensemble import VotingRegressor
from ensemble_models import GradientBoostingModel, HistGradientBoostingModel, AdaBoostingModel, RandomForestModel, \
    XGBoostModel, CatBoostModel, VotingModel
from enums import EnsembleModels, Frequency
import model_base as mb
import traditional as td


class EnsembleAlgorithm:

    def __init__(self):
        self.df_hourly_ts, self.df_daily_ts, self.df_weekly_ts, self.df_monthly_ts = self.__read_data()

    def train_and_evolve(self, ensemble_alg: EnsembleModels):
        for frequency in Frequency:
            if frequency == Frequency.HOURLY:
                self.__train_and_evolve(self.df_hourly_ts, ensemble_alg, frequency)
            elif frequency == Frequency.DAILY:
                self.__train_and_evolve(self.df_daily_ts, ensemble_alg, frequency)
            elif frequency == Frequency.WEEKLY:
                self.__train_and_evolve(self.df_weekly_ts, ensemble_alg, frequency)
            elif frequency == Frequency.MONTHLY:
                self.__train_and_evolve(self.df_monthly_ts, ensemble_alg, frequency)
            else:
                raise ValueError("Invalid frequency")

    def __read_data(self) -> tuple:
        df_hourly_ts, df_daily_ts, df_weekly_ts, df_monthly_ts = mb.read_timestamp_freq()
        mb.set_start_time_index(df_hourly_ts)
        mb.set_start_time_index(df_daily_ts)
        mb.set_start_time_index(df_weekly_ts)
        mb.set_start_time_index(df_monthly_ts)
        return df_hourly_ts, df_daily_ts, df_weekly_ts, df_monthly_ts

    def __init_ensemble_model(self, algorithm: EnsembleModels, frequency=Frequency.HOURLY):
        if algorithm == EnsembleModels.GRADIENT_BOOSTING:
            return GradientBoostingModel(best_params=self.__get_gbr_best_params(frequency))
        elif algorithm == EnsembleModels.HISTOGRAM_GRADIENT_BOOSTING:
            return HistGradientBoostingModel(best_params=self.__get_hist_gbr_best_params(frequency))
        elif algorithm == EnsembleModels.XGBOOST:
            return XGBoostModel(best_params=self.__get_xgb_best_params(frequency))
        elif algorithm == EnsembleModels.ADABOOST:
            return AdaBoostingModel(best_params=self.__get_ada_best_params(frequency))
        elif algorithm == EnsembleModels.CATBOOST:
            return CatBoostModel(best_params=self.__get_cat_best_params(frequency))
        elif algorithm == EnsembleModels.RANDOM_FOREST:
            return RandomForestModel(best_params=self.__get_random_forest_best_params(frequency))
        elif algorithm == EnsembleModels.VOTING:
            return self.__init_voting_regressor(frequency)
        else:
            raise ValueError("Unknown algorithm enum provided!")

    def __train_and_evolve(self, df, ensemble_alg: EnsembleModels, frequency=Frequency.HOURLY):
        print(f"======== {frequency} ========")
        print(f"{ensemble_alg} model - Started at {datetime.now()}")
        train_data, validation_data, test_data = mb.split_data(df)
        # Extract the features
        X_train, X_val, X_test = mb.extract_features(train_data, validation_data, test_data)
        # Extract the target variable
        y_train, y_val, y_test = mb.extract_target(train_data, validation_data, test_data)
        model = self.__init_ensemble_model(ensemble_alg, frequency)
        model.train(X_train, y_train)
        # VALIDATION Prediction and Evolution
        y_val_pred = model.evaluate(X_val, y_val)
        # TEST Prediction and Evolution
        y_test_pred = model.evaluate(X_test, y_test)
        # Plot
        mb.plot_pm_true_predict(validation_data, y_val_pred, 'Validation')
        mb.plot_pm_true_predict(test_data, y_test_pred, 'Test')
        # Save the model
        # mb.save_model_to_pickle(model, f'{ensemble_alg}_model_{frequency}.pkl')

    def __get_gbr_best_params(self, frequency=Frequency.HOURLY):
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
        return best_params.get(frequency.value, "Invalid frequency")

    def __get_hist_gbr_best_params(self, frequency=Frequency.HOURLY):
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

        return best_params.get(frequency.value, "Invalid frequency")

    def __get_xgb_best_params(self, frequency=Frequency.HOURLY):
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

        return best_params.get(frequency.value, "Invalid frequency")

    def __get_ada_best_params(self, frequency=Frequency.HOURLY):
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

        return best_params.get(frequency.value, "Invalid frequency")

    def __get_cat_best_params(self, frequency=Frequency.HOURLY):
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

        return best_params.get(frequency.value, "Invalid frequency")

    def __get_random_forest_best_params(self, frequency=Frequency.HOURLY):
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
        return best_params.get(frequency.value, "Invalid frequency")

    def __init_voting_regressor(self, frequency=Frequency.HOURLY):
        model_gb = self.__init_ensemble_model(EnsembleModels.GRADIENT_BOOSTING, frequency).model
        model_hist_gb = self.__init_ensemble_model(EnsembleModels.HISTOGRAM_GRADIENT_BOOSTING, frequency).model
        model_xgb = self.__init_ensemble_model(EnsembleModels.XGBOOST, frequency).model
        model_ada = self.__init_ensemble_model(EnsembleModels.ADABOOST, frequency).model
        model_cat = self.__init_ensemble_model(EnsembleModels.CATBOOST, frequency).model
        model_rf = self.__init_ensemble_model(EnsembleModels.RANDOM_FOREST, frequency).model
        model_svr = td.init_svr(frequency.value)
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
        return VotingModel(model=model)
