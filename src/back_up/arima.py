import pmdarima as pm
import model_base as mb
from statsmodels.tsa.statespace.sarimax import SARIMAX

FEATURES = ['NO2-Value', 'O3-Value', 'SO2-Value', 'PM10-Value']
TARGET = 'PM2.5-Value'


def init_auto_model(df):
    X, y = mb.define_target_features(df)
    return pm.auto_arima(y.values, start_p=1, start_q=1,
                         test='adf',  # use adftest to find optimal 'd'
                         max_p=10, max_q=10,  # maximum p and q
                         m=1,  # frequency of series
                         d=None,  # let model determine 'd'
                         seasonal=False,  # No Seasonality
                         start_P=0,
                         D=0,
                         trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True,
                         exogenous=X)


def init_arimax_model(df):
    # seasonal_order=(0, 0, 0, 0 means no seasonal
    X, y = mb.define_target_features(df)
    return SARIMAX(y, exog=X, order=(5, 0, 1), seasonal_order=(0, 0, 0, 0))


