import numpy as np
import pandas as pd


def get_anomaly_score(data, test_window, control_window):
    # Given a pd Series, return the anomaly score series.
    # The data given should be sorted in the time order.
    test_mean = data.rolling(test_window).mean()
    control_data = test_mean.rolling(control_window)
    return (test_mean - control_data.mean()) / control_data.std()
