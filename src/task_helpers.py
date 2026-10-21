"""
task_helpers.py
===============

Helper functions for Task 2 of the forecasting exercise.

They take care of the statsmodels plumbing (building the models, fitting
them, forecasting) so you can focus on the modelling choices. You don't
need to edit this file, but you're welcome to read it.
"""

import os

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier, DeterministicProcess

# Settings shared by both models.
_MODEL_SETTINGS = dict(enforce_stationarity=True, enforce_invertibility=True)
_FIT_SETTINGS = dict(disp=False, method="powell", maxiter=2000)


def _require(value, name):
    """Stop with a friendly message if a notebook setting is still None."""
    if value is None:
        raise ValueError(
            f"{name} is still None. Fill it in in the notebook, then re-run the cell."
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path="data/forecast_preproc_task.csv"):
    """Read the hourly CSV saved in Task 1 and give it an hourly time index."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Finish Task 1 first: its last step saves this file."
        )
    data = pd.read_csv(path, index_col="Timestamp", parse_dates=True)
    # statsmodels needs to know how far apart the time steps are (here, 1 hour).
    data.index = pd.DatetimeIndex(data.index, freq=pd.infer_freq(data.index))
    return data


def split_train_test(data, n_test):
    """Hold out the last `n_test` rows as the test set. Everything before is training data."""
    _require(n_test, "HOURS_IN_TEST")
    train = data.iloc[:-n_test].copy()
    test = data.iloc[-n_test:].copy()
    # Re-attach the hourly frequency to each piece.
    train.index = pd.DatetimeIndex(train.index, freq=data.index.freq)
    test.index = pd.DatetimeIndex(test.index, freq=data.index.freq)
    return train, test


# ---------------------------------------------------------------------------
# Fourier terms (Method 2)
# ---------------------------------------------------------------------------

def make_fourier_terms(train_index, test_index, daily_period, weekly_period, order=3):
    """Build daily and weekly Fourier columns for the training and test periods.

    Returns two DataFrames (train, test). Each has a constant column plus
    2 * order sine/cosine columns for each period.
    """
    _require(daily_period, "DAILY_PERIOD")
    _require(weekly_period, "WEEKLY_PERIOD")

    dp = DeterministicProcess(
        index=train_index,
        constant=True,   # a column of 1s, so the model has an average level
        order=0,         # no straight-line trend
        additional_terms=[
            Fourier(period=daily_period, order=order),
            Fourier(period=weekly_period, order=order),
        ],
        drop=True,       # drop any duplicate columns
    )
    train_terms = dp.in_sample()
    # out_of_sample carries the waves on from where training stopped, so the
    # test-week columns stay in step with the training ones.
    test_terms = dp.out_of_sample(steps=len(test_index), forecast_index=test_index)
    return train_terms, test_terms


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def fit_seasonal_sarima(y_train, order, seasonal_order):
    """Method 1: SARIMAX with a seasonal part and no extra inputs."""
    if None in seasonal_order:
        _require(None, "SEASON_LENGTH")
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                    **_MODEL_SETTINGS)
    return model.fit(**_FIT_SETTINGS)


def fit_fourier_sarimax(y_train, order, fourier_train):
    """Method 2: SARIMAX with no seasonal part. The Fourier columns supply the seasonality."""
    model = SARIMAX(y_train, order=order, exog=fourier_train, **_MODEL_SETTINGS)
    return model.fit(**_FIT_SETTINGS)


def forecast(fit, test_index, fourier_test=None):
    """Forecast every hour of the test period. Returns a pandas Series."""
    prediction = fit.predict(start=test_index[0], end=test_index[-1],
                             exog=fourier_test)
    prediction.name = "forecast"
    return prediction
