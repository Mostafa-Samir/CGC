import os
import multiprocessing as mp
import warnings

warnings.filterwarnings("error")
mp.set_start_method('spawn', force=True)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from vorpy.symplectic_integration.nonseparable_hamiltonian import integrate, heuristic_estimate_for_omega
from scipy.integrate import odeint, solve_ivp, ODEintWarning
import matplotlib.pyplot as plt

from cgc.graph import ComputationalGraph, derivative
from cgc.optimizers import TwoStepsNGDptimizerForKF
from cgc.utils import KernelParameter as KP
from cgc.kernels.factory import KernelsFactory

from sklearn.preprocessing import PolynomialFeatures
import re

from jax import config
config.update("jax_enable_x64", True)

np.int = int


N = 400
T_MAX = 80
OBSERVATIONS_END = 200
KFLOW_LEARNING = False
USE_SYMPLICTIC_INT = False

H_KERNEL = "separable-polynomial"
H_KERNEL_PARAMS = {"constant": KP(1.0, learnable=False), "exponent": KP(3.0, learnable=False)}
H_KERNEL_NUGGET = 0.001

# The following is applied to all components of p in the system
P_KERNEL = "gaussian"
P_KERNEL_PARAMS = {"scale": KP(1.0, learnable=True)}
P_KERNEL_NUGGET = 1e-5

# The following is applied to all components of q in the system
Q_KERNEL = "gaussian"
Q_KERNEL_PARAMS = {"scale": KP(1.0, learnable=True)}
Q_KERNEL_NUGGET = 1e-5


def get_hamiltonian_from(graph: ComputationalGraph, Z):
    """Retunr a callable for the learned hamiltonina from CGC."""
    H_fn = graph._unknown_functions["H"]
    return lambda pq: H_fn._f(pq, H_fn.parameter(Z), H_fn.observation(Z))


def hamiltonian_symplectic_int(H_grad, initial_conditions, t, order=2, c=10):
    """Symplectically integerate a hamiltonina system."""
    dt = jnp.mean(jnp.diff(t))
    omega = heuristic_estimate_for_omega(delta=dt, order=order, c=c)
    dim = int(len(initial_conditions) / 2)

    dH_dp = lambda q, p: H_grad(jnp.hstack((p, q)))[:dim]
    dH_dq = lambda q, p: H_grad(jnp.hstack((p, q)))[dim:]

    init_conds = jnp.asarray(initial_conditions).reshape((2, -1))[::-1]

    pq_inverted = np.squeeze(integrate(
        initial_coordinates=init_conds,
        t_v=t,
        dH_dp=dH_dp,
        dH_dq=dH_dq,
        order=order,
        omega=omega
    ))

    if dim == 1:
        pq = pq_inverted[:, [1, 0]]
    else:
        pq_inverted_reshaped = np.reshape(pq_inverted, (-1, 4))
        pq = pq_inverted_reshaped[:, [2, 3, 0, 1]]

    return pq



def get_pq_from(graph: ComputationalGraph, Z):
    _, size = Z.shape
    dims = (size - 2) // 2

    p_fns = [graph._unknown_functions[f"p{index + 1 if dims > 1 else ''}"] for index in range(dims)]
    q_fns = [graph._unknown_functions[f"q{index + 1 if dims > 1 else ''}"] for index in range(dims)]   

    def pq_fn(t):
        ps = [p_fn._f(t, p_fn.parameter(Z), p_fn.observation(Z)) for p_fn in p_fns]
        qs = [q_fn._f(t, q_fn.parameter(Z), q_fn.observation(Z)) for q_fn in q_fns]

        return jnp.hstack(ps + qs)

    return pq_fn


def __reduce_mask(nd_mask):
        reduced_mask = nd_mask
        if nd_mask.ndim > 1:
            reduced_mask = np.multiply.reduce(nd_mask, axis=1).astype(bool)
        return reduced_mask

def _adjust_shapes(matrix, sims, y_train):
    _, matrix_trainling_dim_size = matrix.shape
    observation_leading_dim_size, *_ = y_train.shape
    *_, sims_trailing_dim_size = sims.shape

    if matrix_trainling_dim_size != observation_leading_dim_size:
        y_train = jnp.reshape(y_train, (matrix_trainling_dim_size, ), order='F')

    if matrix_trainling_dim_size != sims_trailing_dim_size:
        sims = jnp.reshape(sims, (matrix_trainling_dim_size, ), order='F')

    return sims, y_train

def two_steps_initialization_low(graph, X, M, observations_end, use_T_in_second_step=False):
    p_kernel = KernelsFactory.create(P_KERNEL, P_KERNEL_PARAMS, P_KERNEL_NUGGET)
    q_kernel = KernelsFactory.create(Q_KERNEL, Q_KERNEL_PARAMS, Q_KERNEL_NUGGET)
    H_kernel = KernelsFactory.create(H_KERNEL, H_KERNEL_PARAMS, H_KERNEL_NUGGET, linear_functional=jax.jacobian)

    _, pq_size = X.shape
    ndims = (pq_size - 2) // 2
    p_start, p_end = 1, ndims + 1
    q_start, q_end = ndims + 1, 2 * ndims + 1
    true_observations_mask = __reduce_mask(M[:observations_end, 1:pq_size - 1])

    T = X[:, 0]
    T_ghost = X[:observations_end, 0]
    T_train = X[:observations_end, 0][true_observations_mask]
    P_train = X[:observations_end, p_start:p_end][true_observations_mask]
    Q_train = X[:observations_end, q_start:q_end][true_observations_mask]

    def p(t):
        sims = p_kernel(t, T_train)
        matrix = p_kernel.matrix(T_train)

        sims, y_train = _adjust_shapes(matrix, sims, P_train)

        return sims @ jnp.linalg.solve(matrix, y_train)
    
    def q(t):
        sims = q_kernel(t, T_train)
        matrix = q_kernel.matrix(T_train)

        sims, y_train = _adjust_shapes(matrix, sims, Q_train)

        return sims @ jnp.linalg.solve(matrix, y_train)
    
    p_dot = jax.jacobian(p)
    q_dot = jax.jacobian(q)

    if use_T_in_second_step:
        PQ_train = jnp.concatenate((jax.jit(jax.vmap(p))(T_ghost), jax.jit(jax.vmap(q))(T_ghost)), axis=1)
        QP_DOT_train = jnp.concatenate((jax.jit(jax.vmap(q_dot))(T_ghost), -jax.jit(jax.vmap(p_dot))(T_ghost)), axis=1)
    else:
        PQ_train = jnp.concatenate((P_train, Q_train), axis=1)
        QP_DOT_train = jnp.concatenate((
                jax.jit(jax.vmap(q_dot))(T_train),
                -jax.jit(jax.vmap(p_dot))(T_train)
            )
        )

    def H(pq):
        sims = H_kernel(pq, PQ_train)
        matrix = H_kernel.matrix(PQ_train)

        sims, y_train = _adjust_shapes(matrix, sims, QP_DOT_train)

        return jnp.squeeze(sims @ jnp.linalg.solve(matrix, y_train))
    
    H_grad = jax.jit(jax.grad(H))

    init_point = X[observations_end - 1, 1:pq_size - 1]

    if not USE_SYMPLICTIC_INT:

        def ms_kernel_ode(pq, t):
            dims = (pq_size - 2) // 2
            grad_val = H_grad(pq)
            dpH = grad_val[0: dims]
            dqH = grad_val[dims: 2 * dims]

            return np.concatenate([-dqH, dpH])
        
        pq_2s = odeint(ms_kernel_ode, init_point, T[observations_end - 1:])
    else: 
        pq_2s = hamiltonian_symplectic_int(H_grad, init_point, T[observations_end - 1:])

    pq_interpolated = jnp.concatenate((
            jax.jit(jax.vmap(p))(X[:observations_end, 0]),
            jax.jit(jax.vmap(q))(X[:observations_end, 0])
        ),
        axis=1
    )

    X_initialized = X.copy()
    X_initialized[observations_end - 1:, 1:pq_size - 1] = pq_2s
    X_initialized[:observations_end, 1:pq_size - 1][~true_observations_mask] = pq_interpolated[~true_observations_mask, :]
    H_init = jax.jit(jax.vmap(H))(X_initialized[:, 1:pq_size - 1])
    X_initialized[:, pq_size - 1] = H_init

    _, separate_losses = graph._loss(X_initialized, X_initialized, M, return_separate=True)
    rkhs_order_magnitude = np.floor(np.log10(separate_losses.get("rkhs_norm")))
    unk_funcs_order_magnitude = np.floor(np.log10(separate_losses.get("unknown_funcs_loss")))
    constraints_order_magnitude = np.floor(np.log10(separate_losses.get("constraints_loss")))
    data_compliance_order_magnitude = np.floor(np.log10(separate_losses.get("data_compliance_loss")))

    unknown_functions_loss_multiplier = 10 ** (rkhs_order_magnitude - unk_funcs_order_magnitude)
    constraint_loss_multiplier = 10 ** (rkhs_order_magnitude - constraints_order_magnitude)
    data_compliance_loss_multiplier = 10 ** (rkhs_order_magnitude - data_compliance_order_magnitude)

    return X_initialized, unknown_functions_loss_multiplier, constraint_loss_multiplier, data_compliance_loss_multiplier


def two_steps_initialization(graph: ComputationalGraph, X, M, observations_end):

    _, pq_size = X.shape

    true_observations_mask = __reduce_mask(M[:observations_end, 1:pq_size - 1])

    H_est = get_hamiltonian_from(graph, X[:observations_end, :][true_observations_mask])
    pq_est = get_pq_from(graph, X[:observations_end, :][true_observations_mask])
    H_grad = jax.jit(jax.grad(H_est))

    init_point = X[observations_end - 1, 1:pq_size - 1]
    t = X[:, 0]

    def ms_kernel_ode(pq, t):
        dims = (pq_size - 2) // 2
        grad_val = H_grad(pq)
        dpH = grad_val[0: dims]
        dqH = grad_val[dims: 2 * dims]

        return np.concatenate([-dqH, dpH])
    
    # running odeint anyway to catch an ODEIntRuntimeError if occurs
    # adds more time to the process, but enusures it never freezes
    pq_2s = odeint(ms_kernel_ode, init_point, t[observations_end - 1:], full_output=1)

    if USE_SYMPLICTIC_INT: 
        pq_2s = hamiltonian_symplectic_int(H_grad, init_point, t[observations_end - 1:])
    
    pq_interpolated = jax.jit(jax.vmap(pq_est))(t[:observations_end])

    X_initialized = X.copy()
    X_initialized[observations_end - 1:, 1:pq_size - 1] = pq_2s
    X_initialized[:observations_end, 1:pq_size - 1][~true_observations_mask] = pq_interpolated[~true_observations_mask, :]
    H_init = jax.jit(jax.vmap(H_est))(X_initialized[:, 1:pq_size - 1])
    X_initialized[:, pq_size - 1] = H_init

    _, separate_losses = graph._loss(X_initialized, X_initialized, M, return_separate=True)
    rkhs_order_magnitude = np.floor(np.log10(separate_losses.get("rkhs_norm")))
    unk_funcs_order_magnitude = np.floor(np.log10(separate_losses.get("unknown_funcs_loss")))
    constraints_order_magnitude = np.floor(np.log10(separate_losses.get("constraints_loss")))
    data_compliance_order_magnitude = np.floor(np.log10(separate_losses.get("data_compliance_loss")))

    unknown_functions_loss_multiplier = 10 ** (rkhs_order_magnitude - unk_funcs_order_magnitude)
    constraint_loss_multiplier = 10 ** (rkhs_order_magnitude - constraints_order_magnitude)
    data_compliance_loss_multiplier = 10 ** (rkhs_order_magnitude - data_compliance_order_magnitude)

    return X_initialized, unknown_functions_loss_multiplier, constraint_loss_multiplier, data_compliance_loss_multiplier


def run_kflow_for(graph: ComputationalGraph, X, M, observations_end):

    _, pq_size = X.shape

    true_observations_mask = __reduce_mask(M[:observations_end, 1:pq_size - 1])

    graph.set_uknknow_fns_parameters_range()

    for _, fn in graph._unknown_functions.items():
        optimizer = TwoStepsNGDptimizerForKF(fn.kflow_loss)
        params, trainable_mask, weights_mask = graph._gather_parameters()
        
        fn_params_only_mask = np.zeros_like(trainable_mask)
        fn_params_start, fn_params_end = fn.parameters_range
        fn_params_only_mask[fn_params_start:fn_params_end] = 1.0
    
        new_params = optimizer.run(params, X[:observations_end, :][true_observations_mask],  M, original_params=params, trainable_mask=trainable_mask, sparse_mask=weights_mask, special_mask=fn_params_only_mask)
        graph._scatter_parameters(new_params)

        graph.report_kernel_params()


def mse(predictions: np.ndarray, truth: np.ndarray):
    if predictions.size == 0 and truth.size == 0:
        error = 0
    else:
        error = np.mean((predictions - truth) ** 2)

    return error


def relative_error(predictions: np.ndarray, truth: np.ndarray):
    return np.mean(np.abs((predictions - truth)) / (np.abs(truth) + np.abs(predictions)))

def relative_error_l2(predictions: np.ndarray, truth: np.ndarray):
    if predictions.size == 0 and truth.size == 0:
        error = 0
    else:
        error = np.linalg.norm(truth - predictions, ord=2) / np.linalg.norm(truth, ord=2)

    return error


def get_seprable_H_coef_error(Z, observations_end, true_coefs):

    H_kernel = KernelsFactory.create(H_KERNEL, H_KERNEL_PARAMS, H_KERNEL_NUGGET)

    _, pq_size = Z.shape
    ndims = (pq_size - 2) // 2
    p_start, p_end = 1, ndims + 1
    q_start, q_end = ndims + 1, 2 * ndims + 1

    H_estimated = Z[:observations_end, -1]
    P_train = Z[:observations_end, p_start:p_end]
    Q_train = Z[:observations_end, q_start:q_end]
    PQ_train = jnp.concatenate((P_train, Q_train), axis=1)

    H_k = H_kernel.matrix(PQ_train)

    poly_features = PolynomialFeatures(degree=3)
    mapped_features = np.concatenate((
        poly_features.fit_transform(P_train),
        poly_features.fit_transform(Q_train)
    ), axis=1)

    w_p = mapped_features.T @ jnp.linalg.solve(H_k, H_estimated)

    def _replace_func(new_var, text):
        def repl(match):
            num = int(match.group(1))
            return f"{new_var}{num+1}"
    
        return re.sub(r"x(\d+)", repl, text)

    terms = [_replace_func('p', f) for f in poly_features.get_feature_names_out()] + \
            [_replace_func('q', f) for f in poly_features.get_feature_names_out()]
    
    w_true = np.zeros_like(w_p)
    for i, term in enumerate(terms):
        if term in true_coefs:
            w_true[i] = true_coefs[term]
            
    return relative_error_l2(w_p, w_true)



def build_1d_graph():

    graph = ComputationalGraph(observables_order=["t", "p", "q", "H"])

    graph.add_observable("t")
    graph.add_unknown_fn("t", "q", alpha=P_KERNEL_NUGGET, kernel=P_KERNEL, kernel_parameters=P_KERNEL_PARAMS)
    graph.add_unknown_fn("t", "p", alpha=Q_KERNEL_NUGGET, kernel=Q_KERNEL, kernel_parameters=Q_KERNEL_PARAMS)

    graph.add_known_fn("p", "p_dot", derivative)
    graph.add_known_fn("q", "q_dot", derivative)
    graph.add_known_fn("p_dot", "-p_dot", lambda p_dot: -p_dot)

    graph.add_aggregator(["q_dot", "-p_dot"], "qp_dot")

    graph.add_aggregator(["p", "q"], "pq")
    graph.add_unknown_fn("pq", "H", linear_functional=jax.jacobian, observations="qp_dot", alpha=H_KERNEL_NUGGET, kernel=H_KERNEL, kernel_parameters=H_KERNEL_PARAMS)
    graph.add_known_fn("H", "grad_H", derivative)

    graph.add_aggregator(["q_dot", "grad_H"], "(q_dot, grad_H)")
    graph.add_aggregator(["p_dot", "grad_H"], "(p_dot, grad_H)")

    def p_dot_constraint(p_dot_grad_H):
        p_dot, grad_H = p_dot_grad_H[:, 0], p_dot_grad_H[:, 1:]
        return p_dot + grad_H[:, 1]

    def q_dot_constraint(q_dot_grad_H):
        q_dot, grad_H = q_dot_grad_H[:, 0], q_dot_grad_H[:, 1:]
        return q_dot - grad_H[:, 0]

    graph.add_constraint("(p_dot, grad_H)", "W1", p_dot_constraint)
    graph.add_constraint("(q_dot, grad_H)", "W2", q_dot_constraint)

    return graph


def build_2d_graph():

    graph = ComputationalGraph(observables_order=["t", "p1", "p2", "q1", "q2", "H"])

    graph.add_observable("t")

    graph.add_unknown_fn("t", "p1", alpha=P_KERNEL_NUGGET, kernel=P_KERNEL, kernel_parameters=P_KERNEL_PARAMS)
    graph.add_unknown_fn("t", "p2", alpha=P_KERNEL_NUGGET, kernel=P_KERNEL, kernel_parameters=P_KERNEL_PARAMS)
    graph.add_unknown_fn("t", "q1", alpha=Q_KERNEL_NUGGET, kernel=Q_KERNEL, kernel_parameters=Q_KERNEL_PARAMS)
    graph.add_unknown_fn("t", "q2", alpha=Q_KERNEL_NUGGET, kernel=Q_KERNEL, kernel_parameters=Q_KERNEL_PARAMS)


    graph.add_known_fn("p1", "p1_dot", derivative)
    graph.add_known_fn("p2", "p2_dot", derivative)
    graph.add_known_fn("q1", "q1_dot", derivative)
    graph.add_known_fn("q2", "q2_dot", derivative)

    graph.add_aggregator(["q1_dot", "q2_dot"], "q_dot")
    graph.add_aggregator(["p1_dot", "p2_dot"], "p_dot")
    graph.add_known_fn("p_dot", "-p_dot", lambda p_dot: -p_dot)

    graph.add_aggregator(["q_dot", "-p_dot"], "qp_dot")
    graph.add_aggregator(["p1", "p2", "q1", "q2"], "pq")

    graph.add_unknown_fn("pq", "H", linear_functional=jax.jacobian, observations="qp_dot", alpha=H_KERNEL_NUGGET, kernel=H_KERNEL, kernel_parameters=H_KERNEL_PARAMS)

    graph.add_known_fn("H", "grad_H", derivative)

    graph.add_aggregator(["p_dot", "grad_H"], "(p_dot, grad_H)")
    def p_dot_constraint(p_dot_grad_H):
        p_dot, grad_H = p_dot_grad_H[:, :2], p_dot_grad_H[:, 2:]
        return p_dot + grad_H[:, 2:]

    graph.add_aggregator(["q_dot", "grad_H"], "(q_dot, grad_H)")
    def q_dot_constraint(q_dot_grad_H):
        q_dot, grad_H = q_dot_grad_H[:, :2], q_dot_grad_H[:, 2:]
        return q_dot - grad_H[:, :2]

    graph.add_constraint("(p_dot, grad_H)", "W1", p_dot_constraint)
    graph.add_constraint("(q_dot, grad_H)", "W2", q_dot_constraint)

    return graph


def generate_ms_data():

    def ms_system_ode(pq, t):
        p, q = pq
        h_grad = [
            -q,
            p
        ]   

        return h_grad

    t = np.linspace(0, T_MAX, N)
    pq = odeint(ms_system_ode, [1, 0], t)

    p, q = pq.T
    H = 0.5 * (p ** 2 + q ** 2)

    X_true = np.concatenate((
        t[:, np.newaxis],
        pq,
        H[:, np.newaxis],
    ), axis=1)

    true_coefs = {"p1^2": 0.5, "q1^2": 0.5}

    return X_true, true_coefs


def generate_m2s3_data():

    def m2s3_system_ode(pq, t):
        p1, p2, q1, q2 = pq
        h_grad = [
            -q1 + (q2 - q1),
            -q2 - (q2 - q1),
            p1,
            p2
        ]

        return h_grad

    t = np.linspace(0, T_MAX, N)
    pq = odeint(m2s3_system_ode, [0.1, -0.1, 0.2, -0.1], t)

    p1, p2, q1, q2 = pq.T
    H = 0.5 * (q1 ** 2 + q2 ** 2 + (q2 - q1) ** 2 + p1 ** 2 + p2 ** 2)

    X_true = np.concatenate((
        t[:, np.newaxis],
        pq,
        H[:, np.newaxis],
    ), axis=1)

    true_coefs = {"p1^2": 0.5, "p2^2": 0.5, "q1^2": 1, "q2^2": 1, "q1 q2": -1}

    return X_true, true_coefs


def generate_hh_data():

    def hh_system_ode(pq, t):
        p1, p2, q1, q2 = pq
        h_grad = [
            -q1 - 2 * q1 * q2,
            -q2 - q1 ** 2 + q2 ** 2,
            p1,
            p2
        ]

        return h_grad

    t = np.linspace(0, T_MAX, N)
    pq = odeint(hh_system_ode, [0.1, -0.1, 0.2, -0.1], t)

    p1, p2, q1, q2 = pq.T
    H = 0.5 * (q1 ** 2 + q2 ** 2 + p1 ** 2 + p2 ** 2) + q2 * q1 ** 2 - (1/3) * q2 ** 3

    X_true = np.concatenate((
        t[:, np.newaxis],
        pq,
        H[:, np.newaxis],
    ), axis=1)

    true_coefs = {"p1^2": 0.5, "p2^2": 0.5, "q1^2": 0.5, "q2^2": 0.5, "q1^2 q2": 1, "q2^3": -(1/3)}

    return X_true, true_coefs


def generate_np_data():

    def H(p, q):
        return (0.5 * p ** 2) - jnp.cos(q)

    def system_ode(pq, t):
        h_grad = [0, 0]
        p, q = pq
        h_grad[0] = -jnp.sin(q)
        h_grad[1] = p
        return h_grad

    t = np.linspace(0, T_MAX, N)
    pq = odeint(system_ode, [0.0, 0.95*np.pi], t=t)

    p, q = pq.T
    H = H(p, q)

    X_true = np.concatenate((
        t[:, np.newaxis],
        pq,
        H[:, np.newaxis],
    ), axis=1)

    return X_true, {}


def generate_X_and_M(X_true, sparsity_factor, seed):

    M = np.ones_like(X_true).astype(bool)
    rng = np.random.default_rng(seed=seed)
    sparse_mask = rng.choice([False, True], p=[sparsity_factor, 1 - sparsity_factor], size=OBSERVATIONS_END)
    sparse_mask[-1] = True

    _, size = X_true.shape
    for i in range(1, size - 1):
        M[:OBSERVATIONS_END, i] = sparse_mask
    M[OBSERVATIONS_END:, 1:size -1] = False
    M[:, size - 1] = False

    X = np.zeros_like(X_true)
    X[M] = X_true[M]

    return X, M


def run_for(data_generator, graph_generator, sparsity_factor, seed, get_H_coef_error=False):

    X_true, true_coefs = data_generator()
    graph = graph_generator()
    X, M = generate_X_and_M(X_true, sparsity_factor, seed)
    
    _, size = X_true.shape
    ndims = (size - 2) // 2
    p_start, p_end = 1, ndims + 1
    q_start, q_end = ndims + 1, 2 * ndims + 1
    
    if KFLOW_LEARNING:
        run_kflow_for(graph, X, M, OBSERVATIONS_END)

    true_observations_mask = __reduce_mask(M[:OBSERVATIONS_END, 1:size - 1])
    
    X_init, unknown_functions_loss_multiplier, constraint_loss_multiplier, data_compliance_loss_multiplier = two_steps_initialization_low(graph, X, M, OBSERVATIONS_END, use_T_in_second_step=True)

    two_steps_mse_p_int = mse(X_true[:OBSERVATIONS_END, p_start:p_end][~true_observations_mask], X_init[:OBSERVATIONS_END, p_start:p_end][~true_observations_mask])
    two_steps_mse_q_int = mse(X_true[:OBSERVATIONS_END, q_start:q_end][~true_observations_mask], X_init[:OBSERVATIONS_END, q_start:q_end][~true_observations_mask])
    two_steps_h_mse_no_center_int = mse(X_true[:OBSERVATIONS_END, -1], X_init[:OBSERVATIONS_END, -1])
    two_steps_h_mse_int = mse((X_true[:OBSERVATIONS_END, -1] - np.median(X_true[:OBSERVATIONS_END, -1])), (X_init[:OBSERVATIONS_END, -1] - np.median(X_init[:OBSERVATIONS_END, -1])))
    two_steps_re_p_int = relative_error_l2(X_true[:OBSERVATIONS_END, p_start:p_end][~true_observations_mask], X_init[:OBSERVATIONS_END, p_start:p_end][~true_observations_mask]) * 100
    two_steps_re_q_int = relative_error_l2(X_true[:OBSERVATIONS_END, q_start:q_end][~true_observations_mask], X_init[:OBSERVATIONS_END, q_start:q_end][~true_observations_mask]) * 100
    two_steps_h_re_int = relative_error_l2(X_true[:OBSERVATIONS_END, -1], X_init[:OBSERVATIONS_END, -1]) * 100

    two_steps_mse_p_ext = mse(X_true[OBSERVATIONS_END:, p_start:p_end], X_init[OBSERVATIONS_END:, p_start:p_end])
    two_steps_mse_q_ext = mse(X_true[OBSERVATIONS_END:, q_start:q_end], X_init[OBSERVATIONS_END:, q_start:q_end])
    two_steps_h_mse_no_center_ext = mse(X_true[OBSERVATIONS_END:, -1], X_init[OBSERVATIONS_END:, -1])
    two_steps_h_mse_ext = mse((X_true[OBSERVATIONS_END:, -1] - np.median(X_true[OBSERVATIONS_END:, -1])), (X_init[OBSERVATIONS_END:, -1] - np.median(X_init[OBSERVATIONS_END:, -1])))
    two_steps_re_p_ext = relative_error_l2(X_true[OBSERVATIONS_END:, p_start:p_end], X_init[OBSERVATIONS_END:, p_start:p_end]) * 100
    two_steps_re_q_ext = relative_error_l2(X_true[OBSERVATIONS_END:, q_start:q_end], X_init[OBSERVATIONS_END:, q_start:q_end]) * 100
    two_steps_h_re_ext = relative_error_l2(X_true[OBSERVATIONS_END:, -1], X_init[OBSERVATIONS_END:, -1]) * 100

    if get_H_coef_error:
        two_steps_h_coef_error = get_seprable_H_coef_error(X_init, OBSERVATIONS_END, true_coefs)

    graph.set_loss_multipliers(unknown_functions_loss_multiplier, constraint_loss_multiplier, data_compliance_loss_multiplier)

    Z = graph.complete(X_init[:OBSERVATIONS_END, :], M[:OBSERVATIONS_END, :], OBSERVATIONS_END, optimizer='l-bfgs-b', learn_parameters=False, n_rounds=1)
    
    H_est = get_hamiltonian_from(graph, Z[:OBSERVATIONS_END, :])
    H_grad = jax.jit(jax.grad(H_est))
    _, pq_size = Z.shape
    init_point = Z[OBSERVATIONS_END - 1, 1:pq_size - 1]
    t = X_true[:, 0]
    
    if USE_SYMPLICTIC_INT:
        pq = hamiltonian_symplectic_int(H_grad, init_point, t[OBSERVATIONS_END - 1:])
    else:
        def ms_kernel_ode(pq, t):
            dims = (pq_size - 2) // 2
            grad_val = H_grad(pq)
            dpH = grad_val[0: dims]
            dqH = grad_val[dims: 2 * dims]

            return np.concatenate([-dqH, dpH])
        
        pq = odeint(ms_kernel_ode, init_point, t[OBSERVATIONS_END - 1:])

    Z_new = np.asarray(X_init).copy()
    Z_new[:OBSERVATIONS_END, 1:pq_size - 1] = Z[:, 1:pq_size - 1]
    Z_new[OBSERVATIONS_END - 1:, 1:pq_size - 1] = pq
    Z_new[:, -1] = jax.jit(jax.vmap(H_est))(Z_new[:, 1:pq_size - 1])   
    Z = Z_new 

    one_step_mse_p_int = mse(X_true[:OBSERVATIONS_END, p_start:p_end], Z[:OBSERVATIONS_END, p_start:p_end])
    one_step_mse_q_int = mse(X_true[:OBSERVATIONS_END, q_start:q_end], Z[:OBSERVATIONS_END, q_start:q_end])
    one_step_h_mse_int = mse((X_true[:OBSERVATIONS_END, -1] - np.median(X_true[:OBSERVATIONS_END, -1])), (Z[:OBSERVATIONS_END, -1] - np.median(Z[:OBSERVATIONS_END, -1])))
    one_step_h_mse_no_center_int = mse(X_true[:OBSERVATIONS_END, -1], Z[:OBSERVATIONS_END, -1])
    one_step_re_p_int = relative_error_l2(X_true[:OBSERVATIONS_END, p_start:p_end], Z[:OBSERVATIONS_END, p_start:p_end]) * 100
    one_step_re_q_int = relative_error_l2(X_true[:OBSERVATIONS_END, q_start:q_end], Z[:OBSERVATIONS_END, q_start:q_end]) * 100
    one_step_h_re_int = relative_error_l2(X_true[:OBSERVATIONS_END, -1], Z[:OBSERVATIONS_END, -1]) * 100
    
    one_step_mse_p_ext = mse(X_true[OBSERVATIONS_END:, p_start:p_end], Z[OBSERVATIONS_END:, p_start:p_end])
    one_step_mse_q_ext = mse(X_true[OBSERVATIONS_END:, q_start:q_end], Z[OBSERVATIONS_END:, q_start:q_end])
    one_step_h_mse_ext = mse((X_true[OBSERVATIONS_END:, -1] - np.median(X_true[OBSERVATIONS_END:, -1])), (Z[OBSERVATIONS_END:, -1] - np.median(Z[OBSERVATIONS_END:, -1])))
    one_step_h_mse_no_center_ext = mse(X_true[OBSERVATIONS_END:, -1], Z[OBSERVATIONS_END:, -1])
    one_step_re_p_ext = relative_error_l2(X_true[OBSERVATIONS_END:, p_start:p_end], Z[OBSERVATIONS_END:, p_start:p_end]) * 100
    one_step_re_q_ext = relative_error_l2(X_true[OBSERVATIONS_END:, q_start:q_end], Z[OBSERVATIONS_END:, q_start:q_end]) * 100
    one_step_h_re_ext = relative_error_l2(X_true[OBSERVATIONS_END:, -1], Z[OBSERVATIONS_END:, -1]) * 100

    if get_H_coef_error:
        one_step_h_coef_error = get_seprable_H_coef_error(Z, OBSERVATIONS_END, true_coefs)

    return_values = []

    return_values.extend([
        X_true, X_init, Z, M,
        two_steps_re_p_int, two_steps_re_p_ext,
        two_steps_re_q_int, two_steps_re_q_ext,
        one_step_re_p_int, one_step_re_p_ext,
        one_step_re_q_int, one_step_re_q_ext,
        two_steps_mse_p_int, two_steps_mse_p_ext,
        two_steps_mse_q_int, two_steps_mse_q_ext,
        one_step_mse_p_int, one_step_mse_p_ext,
        one_step_mse_q_int, one_step_mse_q_ext,
        two_steps_h_re_int, two_steps_h_re_ext,
        two_steps_h_mse_int, two_steps_h_mse_ext,
        one_step_h_re_int, one_step_h_re_ext,
        one_step_h_mse_int, one_step_h_mse_ext,
        two_steps_h_mse_no_center_int, two_steps_h_mse_no_center_ext,
        one_step_h_mse_no_center_int, one_step_h_mse_no_center_ext
    ])

    if get_H_coef_error:
        return_values.extend([two_steps_h_coef_error, one_step_h_coef_error])

    return return_values


if __name__ == "__main__":

    exp_dir = f"{OBSERVATIONS_END}-{H_KERNEL}-with-data-adjusted-and-re-l2-and-h-coef-err{'-and-symplectic' if USE_SYMPLICTIC_INT else ''}"

    columns = [
        "Two-Steps P RE (Interpolation)", "Two-Steps P RE (Extrapolation)",
        "Two-Steps Q RE (Interpolation)", "Two-Steps Q RE (Extrapolation)",
        "One-Step P RE (Interpolation)", "One-Step P RE (Extrapolation)",
        "One-Step Q RE (Interpolation)", "One-Step Q RE (Extrapolation)",
        "Two-Steps P MSE (Interpolation)", "Two-Steps P MSE (Extrapolation)",
        "Two-Steps Q MSE (Interpolation)", "Two-Steps Q MSE (Extrapolation)",
        "One-Step P MSE (Interpolation)", "One-Step P MSE (Extrapolation)",
        "One-Step Q MSE (Interpolation)", "One-Step Q MSE (Extrapolation)",
        "Two-Step H RE (Interpolation)", "Two-Step H RE (Extrapolation)",
        "Two-Step H MSE (Interpolation)", "Two-Step H MSE (Extrapolation)",
        "One-Step H RE (Interpolation)", "One-Step H RE (Extrapolation)",
        "One-Step H MSE (Interpolation)", "One-Step H MSE (Extrapolation)",
        "Two-Step H MSE (No Center) (Interpolation)", "Two-Step H MSE (No Center) (Extrapolation)",
        "One-Step H MSE (No Center) (Interpolation)", "One-Step H MSE (No Center) (Extrapolation)"
    ]

    assert len(columns) == len(set(columns))

    experiements = [
        #("ms", generate_ms_data, build_1d_graph),
        #("m2s3", generate_m2s3_data, build_2d_graph),
        ("hh", generate_hh_data, build_2d_graph),
        ("np", generate_np_data, build_1d_graph)
    ]

    for (sys_name, data_generator, graph_builder) in experiements:

        if ((sys_name == "np") and ("poly" in H_KERNEL)):
            H_KERNEL = "separable-polynomial-rbf"
            H_KERNEL_PARAMS = {"constant": KP(1.0, learnable=False), "exponent": KP(3.0, learnable=False), "rbf_scale": KP(1.0, learnable=False)}

        sys_dir = f"{exp_dir}/{sys_name}"
        os.makedirs(sys_dir, exist_ok=True)

        factors = [0.0, 0.5, 0.6, 0.7, 0.8] if ((sys_name == "m2s3") and ("poly" in H_KERNEL)) else [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]

        get_h_coef_errors = ("poly" in H_KERNEL) and ("np" not in sys_name)

        active_columns = (columns + ["Two-Steps H Coeffecients Error", "One-Step H Coeffecients Error"]) if get_h_coef_errors else columns

        for sparsity_factor in factors:

            os.makedirs(f"{sys_dir}/{sparsity_factor}-best-data", exist_ok=True)

            print(f"Running experiments for {sys_name} @ sparsity of {sparsity_factor}")
            results_fname = f"sparse-{sparsity_factor}.csv"

            results_dict = {c: [] for c in active_columns}
            num_rounds = 10 if sparsity_factor > 0.0 else 1

            X_true, best_X_2d, best_Z, best_M = None, None, None, None
            best_error = np.inf

            num_runs = 0
            seed = 0

            while (num_runs < num_rounds):
                try:
                    X_true, X_2d, Z, M, *errors = run_for(data_generator, graph_builder, sparsity_factor, seed=seed, get_H_coef_error=get_h_coef_errors)
                except (ODEintWarning, RuntimeWarning) as e:
                    seed += 1
                    continue

                for col, val in zip(active_columns, errors):
                    results_dict[col].append(val)

                _, _, _, _, one_step_re_p_int, one_step_re_p_ext, one_step_re_q_int, one_step_re_q_ext, *_ = errors
                avg_err = (one_step_re_p_int + one_step_re_p_ext + one_step_re_q_int + one_step_re_q_ext) / 4
                if avg_err < best_error:
                    best_X_2d = X_2d
                    best_Z = Z
                    best_M = M
                
                num_runs += 1
                seed += 1
                jax.clear_caches()

            #for i in range(num_rounds):
            #    X_true, X_2d, Z, M, *errors = run_for(data_generator, graph_builder, sparsity_factor, seed=i, return_errors_only=False)
            #    for col, val in zip(columns, errors):
            #        results_dict[col].append(val)
#
            #    _, _, one_step_re_p, one_step_re_q, *_ = errors
            #    avg_err = (one_step_re_p + one_step_re_q) / 2
            #    if avg_err < best_error:
            #        best_X_2d = X_2d
            #        best_Z = Z
            #        best_M = M
            #    
            #    jax.clear_caches()
            #    jax.clear_backends()


            for col in active_columns:
                mean = np.mean(results_dict[col])
                std = np.std(results_dict[col])

                results_dict[col].extend([mean, std])

            np.save(f"{sys_dir}/{sparsity_factor}-best-data/X_true.npy", X_true)
            np.save(f"{sys_dir}/{sparsity_factor}-best-data/best_X_2d.npy", best_X_2d)
            np.save(f"{sys_dir}/{sparsity_factor}-best-data/best_Z.npy", best_Z)
            np.save(f"{sys_dir}/{sparsity_factor}-best-data/best_M.npy", best_M)

            results_df = pd.DataFrame(results_dict, index=[f"{i}" for i in range(num_rounds)] + ["Mean", "STD"])
            results_df.to_csv(f"{sys_dir}/{results_fname}")




