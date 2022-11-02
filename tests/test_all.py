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

def test_solve():
    measurements, x_true, x_dot_true, H, times = kal_exp.gen_data(1, nt=20, meas_var=.1, process_var=1)
    x_hat, x_dot_hat, G, Qinv = kal_exp.solve(measurements, H, times, .1)

    n = len(x_true)
    reconstructed_x = np.hstack((H.T[list(range(1, 2 * n)) + [0],:], H.T)) @ np.vstack((x_dot_hat, x_hat))
    G_tilde = G[:, 2:]
    L = np.linalg.cholesky(Qinv.toarray())
    LTGinv = np.linalg.inv(L.T @ G_tilde)
    z_var = np.zeros((n, n))
    z_var[1:,1:] = H[1:, 2:] @ LTGinv @ LTGinv.T @ H[1:, 2:].T
    z_var += .1 * np.eye(n)
    solver_inverse = np.linalg.inv((G.T @ Qinv @ G).toarray() + .1 * H.T @ H)
    overall_variance = solver_inverse @ (.1 * H.T) @ z_var @ (.1 * H) @ solver_inverse.T
    L2 = np.linalg.cholesky(overall_variance)
    ind_normal = np.linalg.inv(L2.T) @ reconstructed_x