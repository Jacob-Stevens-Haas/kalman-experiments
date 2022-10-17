import numpy as np
from scipy import sparse
from pathlib import Path
import mitosis

tuners = {}


def register_tuner(name):
    def decorator(func):
        tuners[name] = func
        return func

    return decorator


def gen_data(seed, *, stop=1, dt=None, nt=None, meas_var = .1, process_var = 1):
    rng = np.random.default_rng(seed)
    if dt is None and nt is None:
        raise ValueError("Either dt or nt must be provided")
    elif nt is not None:
        times = np.linspace(0, stop, nt)
        dt = times[1]-times[0]
    else:
        times = np.arange(0, stop, dt)
    delta_times = times[1:] - times[:-1]
    n = len(times)
    Qs = [
        np.array([[dt, dt**2 / 2], [dt**2 / 2, dt**3 / 3]]) for _ in range(n-1)
    ]
    Q = process_var * sparse.block_diag(Qs)
    x = rng.multivariate_normal(np.zeros(2*n-2), Q.toarray())
    H = sparse.lil_matrix((n, 2 * n))
    H[:, 1::2] = sparse.eye(n)
    dx = H[:-1,:-2] @ x
    x_true = np.concatenate((np.zeros(1), dx.cumsum()))
    dx_dot = H[:-1,1:-1] @ x
    x_dot_true = np.concatenate((np.zeros(1), dx_dot.cumsum()))
    measurements = rng.normal(x_true, meas_var)

    return measurements, x_true, x_dot_true, H, times


def run(seed, sim_params={}, solver_params={}, trials_folder=Path(__file__)):
    meas_var = sim_params["meas_var"]
    measurements, x_true, x_dot_true, obs_operator, times = gen_data(seed, **sim_params)
    process_var = tuners[solver_params["tuner"]](measurements, obs_operator, times, meas_var)
    alpha = meas_var / process_var
    x_hat, x_dot_hat = solve(
        measurements, obs_operator, times, alpha
    )


def solve(measurements, obs_operator, times, alpha):
    H = obs_operator
    z = measurements
    
    delta_times = times[1:] - times[:-1]
    n = len(times)
    Qs = [
        np.array([[dt, dt**2 / 2], [dt**2 / 2, dt**3 / 3]]) for dt in delta_times
    ]
    Qinv = alpha * sparse.block_diag([np.linalg.inv(Q) for Q in Qs])
    Qinv = (Qinv + Qinv.T) / 2  # force to be symmetric

    G_left = sparse.block_diag([-np.array([[1, 0], [dt, 1]]) for dt in delta_times])
    G_right = sparse.eye(2 * (n - 1))
    align_cols = sparse.csc_matrix((2 * (n - 1), 2))
    G = sparse.hstack((G_left, align_cols)) + sparse.hstack((align_cols, G_right))

    H = sparse.lil_matrix((n, 2 * n))
    H[:, 1::2] = sparse.eye(n)

    rhs = H.T @ z.reshape((-1, 1))
    lhs = H.T @ H + G.T @ Qinv @ G
    sol = np.linalg.solve(lhs.toarray(), rhs)
    x_hat = (H @ sol).flatten()
    x_dot_hat = (H[:, list(range(1, 2 * n)) + [0]] @ sol).flatten()
    return x_hat, x_dot_hat
