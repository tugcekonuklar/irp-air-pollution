from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import numpy as np
# from statsmodels.tsa.api import SimpleExpSmoothing

FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'


def init_model(df):
    return SimpleExpSmoothing(df[TARGET]).fit(smoothing_level=0.2, optimized=False)
    # return ExponentialSmoothing(np.asarray(df[TARGET]), seasonal='add', seasonal_periods=24).fit(smoothing_level=0.2, optimized=False)
