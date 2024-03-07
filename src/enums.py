from enum import Enum


class Frequency(Enum):
    HOURLY = 'H'
    DAILY = 'D'
    WEEKLY = 'W'
    MONTHLY = 'M'

    @classmethod
    def get_by_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__} value")


class EnsembleModels(Enum):
    GRADIENT_BOOSTING = 'gradient_boosting'
    HISTOGRAM_GRADIENT_BOOSTING = 'histogram_gradient_boosting'
    XGBOOST = 'xgboost'
    ADABOOST = 'adaboost'
    CATBOOST = 'catboost'
    RANDOM_FOREST = 'random_forest'

    @classmethod
    def get_by_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__} value")
