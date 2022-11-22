import warnings
import pytest
import scipy

import numpy as np
from scipy import stats

import kal_exp

### constants
@pytest.fixture
def sigma_z():
    return 0.1


@pytest.fixture
def sigma_x():
    return 1


@pytest.fixture
def seed():
    return 5


def test_restack():
    x = np.array([[1, 2], [3, 4]])
    x_dot = np.array([[5, 6], [7, 8]])
    expected = np.array(
        [
            [5, 6],
            [1, 2],
            [7, 8],
            [3, 4],
        ]
    )
    result = kal_exp.restack(x, x_dot)
    np.testing.assert_array_equal(result, expected)


@pytest.fixture
def sample_data(sigma_z, sigma_x, seed):
    return kal_exp.gen_data(seed, nt=20, meas_var=sigma_z, process_var=sigma_x)


def test_gen_data_normal(sample_data, sigma_x, sigma_z):
    measurements, x_true, x_dot_true, H, times = sample_data

    # Kolmogorov Smirnov test for normality of marginal distributions
    # of x_true and x_dot_true w/appropriate mean/variance
    delta_times = times[1:] - times[:-1]
    delta_x = x_true[1:] - x_true[:-1] - delta_times * x_dot_true[:-1]
    delta_xdot = x_dot_true[1:] - x_dot_true[:-1]
    p_x = stats.kstest(
        delta_x / np.sqrt(sigma_x * delta_times**3 / 3), stats.norm.cdf
    ).pvalue
    p_xdot = stats.kstest(
        delta_xdot / np.sqrt(sigma_x * delta_times), stats.norm.cdf
    ).pvalue
    assert p_x > 0.05
    assert p_xdot > 0.05

    x_orig = kal_exp.restack(x_true, x_dot_true)
    dz = measurements - H @ x_orig
    # prematrix = root_pinv(sigma_z * np.eye(len(dz)))
    p_z = stats.kstest(sigma_z**-.5 * dz, stats.norm.cdf).pvalue
    assert p_z > .05



def test_solve(sample_data, sigma_z, sigma_x):
    measurements, x_true, x_dot_true, H, times = sample_data
    x_hat, x_dot_hat, G, Qinv = kal_exp.solve(measurements, H, times, sigma_z, sigma_x)
    x_true_stack = kal_exp.restack(x_true, x_dot_true)
    x_hat_stack = kal_exp.restack(x_hat, x_dot_hat)
    err = x_hat_stack - x_true_stack
    # x_hat is hessian_inv @ H.T @ Rinv @ z
    # z is H @ x
    # x_hat - x is (hessian_inv @ H.T @ Rinv @ H - I)x
    # x is normal, mean 0, variance G_dagger @ Q @ G_dagger.T
    # we write this as x is pinv(prematrix) @ N(0, I)
    # Thus x_hat - x is (hessian_inv @ H.T @ Rinv @ H - I) @ pinv(prematrix) @ N(0, I)
    #   aka: pinv(superprematrix) @ N(0, I)
    # E[x_hat - x] = 0
    # E[(x_hat-x)(x_hat-x).T] = pinv(superprematrix) @ pinv(superprematrix).T
    # Thus, superprematrix @ (x_hat - x) \sim N(0, I)
    T = len(x_true)
    G_dagger = np.linalg.pinv(G.toarray())
    R = sigma_z * np.eye(len(measurements))
    Rinv = 1 / sigma_z * np.eye(len(measurements))
    Q = np.linalg.inv(Qinv.toarray())

    hessian = G.T @ Qinv @ G + H.T @ Rinv @ H
    hess_inv = np.linalg.inv(hessian)  # at 30 timepoints, condition:1.2e6

    err_mat = (hess_inv @ H.T @ Rinv @ H - np.eye(2 * T)) @ G_dagger
    var_err = err_mat @ Q @ err_mat.T
    max_cond = np.log(np.linalg.cond(err_mat) ** 2 * np.linalg.cond(Q)) / np.log(10)
    prematrix = root_pinv(var_err, 10 ** (max_cond-16))
    
    p_err = stats.kstest(prematrix @ err, stats.norm.cdf).pvalue
    assert p_err > 0.05


def test_solve_marginal(sample_data, sigma_z, sigma_x):
    measurements, x_true, x_dot_true, H, times = sample_data
    x_hat, x_dot_hat, G, Qinv, sigma_hat = kal_exp.solve_marginal(
        measurements, H, times, sigma_z
    )
    mse = np.sqrt(np.linalg.norm(x_hat - x_true) ** 2 / len(x_true))
    assert mse < 0.05


def test_solve_prior(sample_data, sigma_z, sigma_x):
    measurements, x_true, x_dot_true, H, times = sample_data
    x_hat, x_dot_hat, G, Qinv = kal_exp.solve_prior(
        measurements, H, times, sigma_z, sigma_x
    )
    mse = np.sqrt(np.linalg.norm(x_hat - x_true) ** 2 / len(x_true))
    assert mse < 0.05


def test_solve_variance(sample_data, sigma_z, sigma_x):
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
    x_hat, x_dot_hat, G, Qinv = kal_exp.solve(measurements, H, times, sigma_z, sigma_x)

    n = len(x_true)
    x_solve = kal_exp.restack(x_hat, x_dot_hat)
    # x_ttrue = kal_exp.restack(x_true, x_dot_true)
    G_dagger = np.linalg.pinv(G.toarray())
    R = sigma_z * np.eye(len(measurements))
    Rinv = 1 / sigma_z * np.eye(len(measurements))
    Q = np.linalg.inv(Qinv.toarray())
    var_x = G_dagger @ Q @ G_dagger.T
    prematrix = root_pinv(var_x)

    x_true_norm = prematrix @ kal_exp.restack(
        x_true, x_dot_true
    )
    p_x = stats.kstest(x_true_norm, stats.norm.cdf).pvalue
    assert p_x > 0.05

    var_z = R + H @ var_x @ H.T  # at 30 timepoints, condition: 5.6e0
    prematrix = root_pinv(var_z)

    meas_normalized = np.linalg.inv(np.linalg.cholesky(var_z)) @ measurements
    p_z = stats.kstest(meas_normalized, stats.norm.cdf).pvalue
    assert p_z > 0.05

    hessian = G.T @ Qinv @ G + H.T @ Rinv @ H
    hess_inv = np.linalg.inv(hessian)  # at 30 timepoints, condition:1.2e6
    var_x_hat = hess_inv @ H.T @ Rinv @ var_z @ Rinv @ H @ hess_inv
    max_cond = np.log(np.linalg.cond(hessian) ** 2 * np.linalg.cond(var_z)) / np.log(10)
    prematrix = root_pinv(var_x_hat, 10 ** (max_cond-16))
    # Need to form the root pseudoinverse of var_x_hat
    # L_perm, D, perm = scipy.linalg.ldl(var_x_hat)
    # L_p_inv = np.linalg.inv(L_perm)

    # D_diag = np.diagonal(D)
    # D_diag_log = np.log(np.abs(D_diag)) / np.log(10)
    # D_log_max = D_diag_log.max()
    # cond_cutoff_exponent = D_log_max - 16 + np.log(max_cond) / np.log(10)
    # valid_D_indices = D_diag_log > cond_cutoff_exponent
    # D_tilde_root_inv = np.diag(D_diag[valid_D_indices] ** -0.5)
    
    # est_normal = D_tilde_root_inv @ L_p_inv[valid_D_indices] @ x_solve
    est_normal = prematrix @ x_solve
    p_x_hat = stats.kstest(est_normal, stats.norm.cdf).pvalue
    assert p_x_hat > 0.05


def test_marginal_gradient(sample_data, sigma_z, sigma_x):
    z, x_true, x_dot_true, H, times = sample_data
    delta_times = times[1:] - times[:-1]
    G = kal_exp.gen_G(delta_times)
    Qinv = kal_exp.gen_Qinv(delta_times, sigma_x)
    T = len(times)
    Rinv = 1/ sigma_z * scipy.sparse.eye(T)
    Theta = H.T @ Rinv @ H
    Pi = G.T @ Qinv @ G
    alpha = kal_exp.alpha_proj(G, H, z, Qinv)
    obj = kal_exp.marg_obj(T, Theta, Pi, alpha)
    grad = kal_exp.marg_grad(T, Theta, Pi, alpha)
    x0 = 10
    lhs, rhs = kal_exp.gradient_test(obj, grad, x0)
    assert np.abs(lhs/rhs-1) < 1e-3

    lhs, rhs = kal_exp.complex_step_test(obj, grad, x0)
    assert np.abs(lhs/rhs-1) < 1e-3


def test_map_gradient(seed, sample_data, sigma_z, sigma_x):
    z, x_true, x_dot_true, H, times = sample_data
    rng = np.random.default_rng(seed)
    delta_times = times[1:] - times[:-1]
    G = kal_exp.gen_G(delta_times)
    Qinv = kal_exp.gen_Qinv(delta_times, sigma_x)
    T = len(times)
    Rinv = 1/ sigma_z * scipy.sparse.eye(T)
    Theta = H.T @ Rinv @ H
    Pi = G.T @ Qinv @ G
    subtract = H.T @ Rinv @ z
    eps = .1
    sigma_tilde = 2
    log_coef = T + 1 + eps
    log_add = (2 + 2 * eps) * sigma_tilde
    obj = kal_exp.prior_obj(Pi, subtract, log_coef, log_add)
    grad = kal_exp.prior_grad(Theta, Pi, subtract, log_coef, log_add)
    x0 = rng.normal(loc=kal_exp.restack(x_true, x_dot_true))

    lhs, rhs = kal_exp.gradient_test(obj, grad, x0)
    assert np.abs(lhs/rhs-1) < 1e-3

    lhs, rhs = kal_exp.complex_step_test(obj, grad, x0)
    assert np.abs(lhs/rhs-1) < 1e-3


def root_pinv(Q, threshold=1e-15):
    r"""Calculate the root pseudoinverse of matrix Q in R m x m using the SVD

    If :math:`x \sim \mathcal N(0, Q)`, then 
    :math:`\tilde U^T\Sigma^{-1/2}x\sim \mathcal N(0, I)` if 
    :math:`Q=\tilde U\tilde\sigma\tilde U^T.

    Q must be symmetric, though to make sense as a (singular) covariance, it must be
    positive (semi) definite)

    Arguments:
        Q: The matrix to calculate
        threshold: for determining rank of Q

    Returns:
        Qp_root, the k x m matrix where k is the rank of Q and the above equation holds.
    """
    U, s, _ = scipy.linalg.svd(Q)
    s_diag_nonzero = s/s.max() > threshold
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
        S_root_pinv = np.diag(s[s_diag_nonzero]**-.5)
    return S_root_pinv @ U.T[s_diag_nonzero]
