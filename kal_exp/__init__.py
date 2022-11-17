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
    measurements, obs_operator, times, sigma_z, sigma_tilde=1, x0=None, maxiter=100
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

    def grad(x):
        return (
            # H.T @ Rinv @ (H @ x - z)
            Theta @ x
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
        )[0, 0]

    def sigma_of_x(x):
        return ((x.T @ Pi @ x + log_add) / 2 / log_coef)[0, 0]

    abs_tol = 1e-6
    rel_tol = 1e-3
    x = grad_descent(objective, x0, grad, rel_tol, abs_tol, maxiter)
    alt_x = optimize.minimize(
        lambda x: objective(x.reshape((-1, 1))),
        x0.flatten(),
        jac=lambda x: grad(x.reshape((-1, 1))).flatten(),
    )
    sigma_hat = sigma_of_x(x)
    x_hat, x_dot_hat = unstack(x)
    return x_hat, x_dot_hat, G, Qinv, sigma_hat


def solve_marginal(
    measurements, obs_operator, times, sigma_z, sigma0=None, maxiter=100
):
    H, z, delta_times, T, R, Rinv, G = (
        obs_operator,
        measurements.reshape((-1, 1)),
        *initialize_values(measurements, times, sigma_z),
    )
    if x0 is None:
        x0 = np.zeros((2 * T, 1))
    Qinv = gen_Qinv(delta_times)
    # Precomputations
    Theta = H.T @ Rinv @ H
    Pi = G.T @ Qinv @ G
    rhs = H.T @ Rinv @ z
    temp_vec = G @ np.linalg.inv(H.T @ H + G.T @ G) @ H.T @ z
    alpha = temp_vec.T @ Qinv @ temp_vec

    def grad(sigma):
        # Technically, grad * sigma^2 to remove denominator
        trace_inverse = np.trace(np.linalg.inv(Theta + Pi / sigma) @ Pi)
        return (T - 1) * sigma - alpha - trace_inverse

    def objective(sigma):
        # ||Hx - z||_Rinv^2 + (T+1+eps)*log(||Gx||^2_Qinv + (2+2eps)*sigma_tilde)
        return (
            (T - 1) * np.log(sigma)
            + alpha / sigma**2
            + 1 / 2 * np.log(np.linalg.det(Theta + 1 / sigma * Pi))
        )[0, 0]

    def x_of_sigma(sigma):
        lhs = Theta + 1 / sigma * Pi
        return np.linalg.solve(lhs, rhs)

    g0 = grad(sigma0)
    if g0 < 0:
        right = find_right_bound(sigma0)
        left = sigma0
    elif g0 > 0:
        left = find_left_bound(sigma0)
        right = sigma0
    elif g0 == 0:
        raise ValueError("You're cheating!!!")
    else:
        raise TypeError("NAAAAAAAN")

    abs_tol = 1e-6
    for i in range(maxiter):
        midpoint = (left + right) / 2
        gmid = grad(midpoint)
        if gmid < 0:
            left = midpoint
        elif gmid > 0:
            right = midpoint
        elif gmid == 0:
            raise ValueError("You're cheating!!!")
        else:
            raise TypeError("NAAAAAAAN")
        objmid = objective(midpoint)
        print(f"feval: {objmid:.5} \u03C3\u00B2 estimated as {midpoint:.2}")
        if np.abs(objmid) < abs_tol:
            print("Minimizer found in iter ", i)
            break

    x = x_of_sigma(midpoint)

    x_hat, x_dot_hat = unstack(x)
    return x_hat, x_dot_hat, G, Qinv, midpoint


def linesearch(x0, x1, objective):
    # not guaranteed for nonconvex problem, even for local minima
    gamma0 = 0.5
    gamma = gamma0
    o0 = objective(x0)
    o1 = objective(x1)
    maxiter = 10
    for i in range(maxiter):
        x_new = gamma * x1 + (1 - gamma) * x0
        o_new = objective(x_new)
        if o_new < gamma * o1 + (1 - gamma) * o0:
            print("linesearch found in iter ", i)
            break
        gamma *= gamma0
    else:
        print("linesearch not found")
        return False
    return x_new


def grad_descent(f, x0, grad, rel_tol, abs_tol, maxiter):
    x = x0
    LR = 1e-3
    obj_val = np.inf
    for i in range(maxiter):
        nabla_x = grad(x)
        x_plus = x - LR * nabla_x
        obj_val_plus = f(x)
        abs_err = np.linalg.norm(x - x_plus)
        rel_err = abs_err / min(np.linalg.norm(x), 1)
        print("feval: ", obj_val)  # , " \u03C3\u00B2 estimated ", sigma_of_x(x))
        x = x_plus
        if abs_err < abs_tol or rel_err < rel_tol:
            print("Minimizer found in iter ", i)
            break
        if obj_val_plus > obj_val:
            x_plus = linesearch(x, x_plus, f)
            if x_plus is None:
                x_plus = x
            LR = 0.1 * LR
            print("decreasing learning rate to ", LR)
        obj_val = obj_val_plus
    return x_plus


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
