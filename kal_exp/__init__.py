from pathlib import Path

import numpy as np
from scipy import sparse
from scipy import optimize

tuners = {}


def register_tuner(name):
    def decorator(func):
        tuners[name] = func
        return func

    return decorator


def gen_sine(seed, *, stop=1, dt=None, nt=None, meas_var=0.1):
    """Generate (deterministic) sine trajectory and (random) measurements"""
    rng = np.random.default_rng(seed)
    if dt is None and nt is None:
        raise ValueError("Either dt or nt must be provided")
    elif nt is not None:
        times = np.linspace(0, stop, nt)
        dt = times[1] - times[0]
    else:
        times = np.arange(0, stop, dt)
    n = len(times)
    x_true = np.sin(times)
    x_dot_true = np.cos(times)
    measurements = rng.normal(x_true, meas_var)
    H = sparse.lil_matrix((n, 2 * n))
    H[:, 1::2] = sparse.eye(n)
    return measurements, x_true, x_dot_true, H, times


def gen_data(seed, *, stop=1, dt=None, nt=None, meas_var=0.1, process_var=1):
    """Generate trajectory and measurements for a Kalman process.

    Always starts at origin
    """
    rng = np.random.default_rng(seed)
    if dt is None and nt is None:
        raise ValueError("Either dt or nt must be provided")
    elif nt is not None:
        times = np.linspace(0, stop, nt)
        dt = times[1] - times[0]
    else:
        times = np.arange(0, stop, dt)
    n = len(times)
    Qs = [
        np.array([[dt, dt**2 / 2], [dt**2 / 2, dt**3 / 3]]) for _ in range(n - 1)
    ]
    Q = process_var * sparse.block_diag(Qs)
    x = rng.multivariate_normal(np.zeros(2 * n - 2), Q.toarray())
    H = sparse.lil_matrix((n, 2 * n))
    H[:, 1::2] = sparse.eye(n)
    dx_dot = H[:-1, 1:-1] @ x
    x_dot_true = np.concatenate((np.zeros(1), dx_dot.cumsum()))
    dx = H[:-1, :-2] @ x + dt * x_dot_true[:-1]
    x_true = np.concatenate((np.zeros(1), dx.cumsum()))
    meas_stdev = np.sqrt(meas_var)
    measurements = rng.normal(x_true, meas_stdev)

    return measurements, x_true, x_dot_true, H, times


def run(seed, sim_params={}, solver_params={}, trials_folder=Path(__file__)):
    meas_var = sim_params["meas_var"]
    measurements, x_true, x_dot_true, obs_operator, times = gen_data(seed, **sim_params)
    process_var = tuners[solver_params["tuner"]](
        measurements, obs_operator, times, meas_var
    )
    x_hat, x_dot_hat = solve(measurements, obs_operator, times, meas_var, process_var)
    metrics = {
        "posit_mse": ((x_hat - x_true) ** 2).mean(),
        "vel_mse": ((x_dot_hat - x_dot_true) ** 2).mean(),
    }
    return metrics


def initialize_values(measurements, times, sigma_z):
    delta_times = times[1:] - times[:-1]
    T = len(times)
    R = sigma_z
    if isinstance(R, float) or isinstance(R, int):
        R = R * sparse.eye(len(measurements))
        Rinv = 1 / sigma_z * sparse.eye(len(measurements))
    elif isinstance(R, np.ndarray):
        print(R)
        Rinv = np.linalg.inv(R)
    else:
        raise ValueError(
            "meaurement variance sigma_z must either be a number or array."
        )
    G = gen_G(delta_times)
    return delta_times, T, R, Rinv, G


def gen_G(delta_times):
    T = len(delta_times) + 1
    G_left = sparse.block_diag([-np.array([[1, 0], [dt, 1]]) for dt in delta_times])
    G_right = sparse.eye(2 * (T - 1))
    align_cols = sparse.csc_matrix((2 * (T - 1), 2))
    return sparse.hstack((G_left, align_cols)) + sparse.hstack((align_cols, G_right))


def gen_Qinv(delta_times, sigma_x):
    Qs = [
        np.array([[dt, dt**2 / 2], [dt**2 / 2, dt**3 / 3]]) for dt in delta_times
    ]
    Qinv = sigma_x * sparse.block_diag([np.linalg.inv(Q) for Q in Qs])
    return (Qinv + Qinv.T) / 2  # ensure symmetry


def solve(measurements, obs_operator, times, sigma_z, sigma_x):
    H, z, delta_times, T, R, Rinv, G = (
        obs_operator,
        measurements.reshape((-1, 1)),
        *initialize_values(measurements, times, sigma_z),
    )
    Qinv = gen_Qinv(delta_times, sigma_x)

    rhs = H.T @ Rinv @ z.reshape((-1, 1))
    lhs = H.T @ Rinv @ H + G.T @ Qinv @ G
    sol = np.linalg.solve(lhs.toarray(), rhs)
    x_hat = (H @ sol).flatten()
    x_dot_hat = (H[:, list(range(1, 2 * T)) + [0]] @ sol).flatten()
    return x_hat, x_dot_hat, G, Qinv


def solve_prior(
    measurements, obs_operator, times, sigma_z, sigma_tilde=1, x0=None
):
    H, z, delta_times, T, R, Rinv, G = (
        obs_operator,
        measurements.reshape((-1, 1)),
        *initialize_values(measurements, times, sigma_z),
    )
    eps = 0.1
    if x0 is None:
        x0 = np.zeros((2 * T, 1))
    Qinv = gen_Qinv(delta_times, 1)
    # Precomputations
    Theta = H.T @ Rinv @ H
    Pi = G.T @ Qinv @ G
    subtract = H.T @ Rinv @ z
    log_coef = T + 1 + eps
    log_add = (2 + 2 * eps) * sigma_tilde

    grad = prior_grad(Theta, Pi, subtract, log_coef, log_add)
    objective = prior_obj(Theta, Pi, subtract, log_coef, log_add)

    def sigma_of_x(x):
        return ((x.T @ Pi @ x + log_add) / 2 / log_coef)[0, 0]

    x_sol = optimize.minimize(
        lambda x: objective(x.reshape((-1, 1))),
        x0.flatten(),
        jac=lambda x: grad(x.reshape((-1, 1))).flatten(),
    ).x.reshape((-1,1))
    sigma_hat = sigma_of_x(x_sol)
    x_hat, x_dot_hat = unstack(x_sol)
    return x_hat, x_dot_hat, G, Qinv, sigma_hat


def prior_obj(Theta, Pi, subtract, log_coef, log_add):
    def objective(x):
        return (
            x.T @ Theta @ x
            - 2 * subtract.T @ x
            + log_coef * np.log(x.T @ Pi @ x + log_add)
        )
    return objective


def prior_grad(Theta, Pi, subtract, log_coef, log_add):
    def grad(x):
        return (
            # H.T @ Rinv @ (H @ x - z)
            2 * Theta @ x
            - 2 * subtract
            # + (T+1+eps)* np.log((x.T @ G.T @ Qinv @ G @ x) + (2+2*eps)*sigma_tilde)
            + log_coef * 2 * Pi @ x / ((x.T @ Pi @ x) + log_add)
        )
    return grad


def solve_marginal(
    measurements, obs_operator, times, sigma_z, sigma0=1, maxiter=100
):
    H, z, delta_times, T, R, Rinv, G = (
        obs_operator,
        measurements.reshape((-1, 1)),
        *initialize_values(measurements, times, sigma_z),
    )
    Qinv = gen_Qinv(delta_times, 1)
    # Precomputations
    Theta = H.T @ Rinv @ H
    Pi = G.T @ Qinv @ G
    rhs = H.T @ Rinv @ z

    alpha = alpha_proj(G, H, z, Qinv)

    grad = marg_grad(T, Theta, Pi, alpha)
    objective = marg_obj(T, Theta, Pi, alpha)
    grad_scaled = marg_grad_scaled(T, Theta, Pi, alpha)
    second_d = marg_2ndd_scaled(T, Theta, Pi)

    def x_of_sigma(sigma):
        lhs = (Theta + 1 / sigma * Pi).toarray()
        return np.linalg.solve(lhs, rhs)

    sigma_min = optimize.minimize_scalar(
        fun=objective,
        bracket=(1e-9, 1e9)
    ).x[0,0]
    x_min = x_of_sigma(sigma_min)

    sigma_root = optimize.root_scalar(
        grad_scaled,
        fprime=second_d,
        method="newton",
        x0=sigma0,
    ).root[0,0]
    x_root = x_of_sigma(sigma_root)

    g0 = grad(sigma0)
    if g0 < 0:
        left = sigma0
        while left < 1e10:
            right = left * 10
            if grad(right) > 0:
                break 
            left = right
    elif g0 > 0:
        right = sigma0
        while right > 1e-10:
            left = right / 10
            if grad(left) > 0:
                break
            right = left

    abs_tol = 1e-16
    for i in range(maxiter):
        midpoint = (left + right) / 2
        gmid = grad(midpoint)
        if gmid < 0:
            left = midpoint
        elif gmid > 0:
            right = midpoint
        objmid = objective(midpoint)[0,0]
        print(f"feval: {objmid:.5} \u03C3\u00B2 estimated as {midpoint:.2}")
        if np.abs(gmid) < abs_tol:
            print("Minimizer found in iter ", i)
            break

    x = x_of_sigma(midpoint)

    x_hat, x_dot_hat = unstack(x)
    return x_hat, x_dot_hat, G, Qinv, midpoint


def alpha_proj(G, H, z, Qinv):
    temp_vec = G @ np.linalg.inv((H.T @ H + G.T @ G).toarray()) @ H.T @ z
    return 1/2 * temp_vec.T @ Qinv @ temp_vec


def marg_grad(T, Theta, Pi, alpha):
    def grad(sigma):
        trace_inverse = np.trace(np.linalg.inv((Theta + Pi / sigma).toarray()) @ Pi)
        return (T-1)/sigma - alpha/sigma**2 - 1 / 2 / sigma**2 * trace_inverse
    return grad


def marg_grad_scaled(T, Theta, Pi, alpha):
    """Marginal likelihood gradient, multiplied by positive (sigma^2)^2"""
    def grad(sigma):
        # Technically, grad * sigma^2 to remove denominator
        if not isinstance(sigma, float):
            sigma = sigma[0,0]
        trace_inverse = np.trace(np.linalg.inv((Theta + Pi / sigma).toarray()) @ Pi)
        return (T-1) * sigma - alpha - 1/2 * trace_inverse
    return grad


def marg_2ndd_scaled(T, Theta, Pi):
    """Marginal likelihood Hessian, from gradient multiplied by positive (sigma^2)^2"""
    def second_deriv(sigma):
        if not isinstance(sigma, float):
            sigma = sigma[0,0]
        inv = np.linalg.inv((Theta + Pi / sigma).toarray())
        trace_inverse = np.trace(inv @ Pi @ inv @ Pi)
        return (T-1) + 1 / (2 * sigma ** 2) * trace_inverse
    return second_deriv


def marg_obj(T, Theta, Pi, alpha):
    def objective(sigma):
        if not isinstance(sigma, float):
            sigma = sigma[0,0]
        return (
            (T - 1) * np.log(sigma)
            + alpha / sigma
            + 1 / 2 * np.log(np.linalg.det((Theta + Pi/sigma).toarray()))
        )
    return objective


def restack(x, x_dot):
    """Interleave x and x_dot to get vector represented by Kalman eqns

    Assumes first axis is time.
    """
    output_shape = (x.shape[0] + x_dot.shape[0], *x.shape[1:])
    c = np.empty(output_shape)
    c[0::2] = x_dot
    c[1::2] = x
    return c


def unstack(x):
    """unstack x vector represented by Kalman eqns to get x_dot and x

    Assumes first axis is time.
    """
    return x[1::2].flatten(), x[::2].flatten()


def gradient_test(f, g, x0):
    """Verifies that analytic function f matches gradient function g"""
    if isinstance(x0, np.ndarray):
        h = np.ones_like(x0) / np.linalg.norm(x0) / 1e2
    else: # x0 is float or int
        h = x0/1e2
    return (f(x0 + h) - f(x0 - h))/2, np.dot(h, g(x0))


def complex_step_test(f, g, x0):
    """Verifies that analytic function f matches gradient function g"""
    if isinstance(x0, np.ndarray):
        h = np.ones_like(x0) / np.linalg.norm(x0) / 1e2
    else: # x0 is float or int
        h = x0/1e2
    return f(x0 + h*1j).imag, np.dot(h, g(x0))
