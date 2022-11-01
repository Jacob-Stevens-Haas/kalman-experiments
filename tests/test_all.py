import numpy as np
from scipy import stats

import kal_exp

def test_gen_data():
    measurements, x_true, x_dot_true, H, times = kal_exp.gen_data(1, nt=20, meas_var=.1, process_var=1)

    # K-S test for normality of marginal distributions of x_true and x_dot_true
    delta_times = times[1:] - times[:-1]
    delta_x = x_true[1:]-x_true[:-1]
    delta_xdot = x_dot_true[1:]-x_dot_true[:-1]
    p_x = stats.kstest(delta_x / np.sqrt(delta_times**3/3), stats.norm.cdf).pvalue
    p_xdot = stats.kstest(delta_xdot / np.sqrt(delta_times), stats.norm.cdf).pvalue
    assert p_x > .05
    assert p_xdot > .05