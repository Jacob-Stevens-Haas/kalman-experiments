import pytest
import scipy

import numpy as np
from scipy import stats

import kal_exp

### constants
@pytest.fixture
def sigma_z(): return .1

@pytest.fixture
def sigma_x(): return 1


def test_restack():
    x = np.array([
        [1,2],
        [3,4]
    ])
    x_dot = np.array([
        [5,6],
        [7,8]
    ])
    expected = np.array([
        [5,6],
        [1,2],
        [7,8],
        [3,4],
    ])
    result = kal_exp.restack(x, x_dot)
    np.testing.assert_array_equal(result, expected)

@pytest.fixture
def sample_data(sigma_z, sigma_x):
    return kal_exp.gen_data(1, nt=20, meas_var=sigma_z, process_var=sigma_x)


def test_gen_data_normal(sample_data, sigma_x):
    measurements, x_true, x_dot_true, H, times = sample_data

    # Kolmogorov Smirnov test for normality of marginal distributions 
    # of x_true and x_dot_true w/appropriate mean/variance
    delta_times = times[1:] - times[:-1]
    delta_x = x_true[1:] - x_true[:-1]
    delta_xdot = x_dot_true[1:] - x_dot_true[:-1]
    p_x = stats.kstest(delta_x / np.sqrt(sigma_x * delta_times**3 / 3), stats.norm.cdf).pvalue
    p_xdot = stats.kstest(delta_xdot / np.sqrt(sigma_x * delta_times), stats.norm.cdf).pvalue
    assert p_x > 0.05
    assert p_xdot > 0.05


def test_solve(sample_data, sigma_z, sigma_x):
    """Kolmogorov Smirnov test for normality of x_hat w/appropriate mean/variance

    x_hat = (G^TQ^{-1}G + H^TR^{-1}H)^{-1}H^TR^{-1}z
    z = Normal(0, R) + Hx
    x = Normal(0, G'QG'^T)
    where G' is the pseudoinverse of G
    Therefore variance of z should be:
    R + HG'QG'^TH^T
    and the variance of x_hat should be:
    (G^TQ^{-1}G + H^TR^{-1}H)^{-1}H^TR^{-1} 
        * var(z) 
        * R^{-1}H(G^TQ^{-1}G + H^TR^{-1}H)^{-1}
    so if we mutlipy x_hat by the root pseudoinverse of this, it should be
    a vector of IID standard normal samples
    """
    measurements, x_true, x_dot_true, H, times = sample_data
    x_hat, x_dot_hat, G, Qinv = kal_exp.solve(measurements, H, times, sigma_z/sigma_x)

    n = len(x_true)
    x_solve = kal_exp.restack(x_hat, x_dot_hat)
    G_dagger = np.linalg.pinv(G.toarray())
    sigma_z = 1 # since solve puts it all into Q, not R
    R = sigma_z * np.eye(len(measurements))
    Rinv = 1/sigma_z * np.eye(len(measurements))
    Q = np.linalg.inv(Qinv.toarray())
    var_x = G_dagger @ Q @ G_dagger.T
    var_z = R + H @ var_x @ H.T # at 30 timepoints, condition: 5.6e0
    hessian = G.T @ Qinv @ G + H.T @ Rinv @ H
    hess_inv = np.linalg.inv(hessian) # at 30 timepoints, condition:1.2e6
    var_x_hat = hess_inv @ H.T @ Rinv @ var_z @ Rinv @ H @ hess_inv

    # Need to form the root pseudoinverse of var_x_hat
    L_perm, D, perm = scipy.linalg.ldl(var_x_hat)
    L_p_inv = np.linalg.inv(L_perm)

    max_cond =  np.linalg.cond(hessian) ** 2 * np.linalg.cond(var_z)
    D_diag = np.diagonal(D)
    D_diag_log = np.log(np.abs(D_diag))/np.log(10)
    D_log_max = D_diag_log.max()
    cond_cutoff_exponent = D_log_max - 16 + np.log(max_cond)/np.log(10)
    valid_D_indices = D_diag_log > cond_cutoff_exponent
    D_tilde_root_inv = np.diag(D_diag[valid_D_indices] ** -.5)

    ind_normal = D_tilde_root_inv @ L_p_inv[valid_D_indices] @ x_solve
    p_x_hat = stats.kstest(ind_normal, stats.norm.cdf).pvalue    
    assert p_x_hat > 0.05
