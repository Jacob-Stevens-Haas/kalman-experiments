from pathlib import Path

import numpy as np
from scipy import sparse

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
    measurements = rng.normal(x_true, meas_var)

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
        measurements,
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
    measurements, obs_operator, times, sigma_z, sigma_x=1, sigma_tilde=1, x0=None
):
    H, z, delta_times, T, R, Rinv, G = (
        obs_operator,
        measurements,
        *initialize_values(measurements, times, sigma_z),
    )
    eps = 0.1
    if x0 is None:
        x0 = np.zeros((2 * T, 1))
    Qinv = gen_Qinv(delta_times, sigma_x)
    # Precomputations
    Sigma = H.T @ Rinv @ H
    Pi = G.T @ Qinv @ G
    subtract = H.T @ Rinv @ z
    log_coef = T + 1 + eps
    log_add = (2 + 2 * eps) * sigma_tilde

    def grad(x):
        return (
            # H.T @ Rinv @ (H @ x - z)
            Sigma @ x
            - subtract
            # + (T+1+eps)* np.log((x.T @ G.T @ Qinv @ G @ x) + (2+2*eps)*sigma_tilde)
            + log_coef * Pi @ x / ((x.T @ Pi @ x) + log_add)
        )

    def objective(x):
        # ||Hx - z||_Rinv^2 + (T+1+eps)*log(||Gx||^2_Qinv + (2+2eps)*sigma_tilde)
        return (
            x.T @ Pi @ x
            - 2 * subtract.T @ x
            + log_coef * np.log(x.T @ Pi @ x + log_add)
        )

    for i in range(10):
        pass
    pass


def restack(x, x_dot):
    """Interleave x and x_dot to get vector represented by Kalman eqns

    Assumes first axis is time.
    """
    output_shape = (x.shape[0] + x_dot.shape[0], *x.shape[1:])
    c = np.empty(output_shape)
    c[0::2] = x_dot
    c[1::2] = x
    return c
