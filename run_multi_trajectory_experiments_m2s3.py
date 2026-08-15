import re
import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.integrate import odeint, ODEintWarning
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures

from cgc.graph import ComputationalGraph, derivative
from cgc.utils import KernelParameter as KP
from cgc.kernels.factory import KernelsFactory

cpu_device = jax.devices("cpu")[0]


from jax import config
config.update("jax_enable_x64", True)
pd.set_option("display.precision", 17)

np.int = int

N_TRAJICTORIES_TEST = 50
MAX_N_TRAJICTORIES_TRAIN = 10

N_ROUNDS = 10

N = 400
T_MAX = 80
OBSERVATIONS_END = 200
KFLOW_LEARNING = False
USE_SYMPLECTIC_INTEGERATION = False

H_KERNEL = "separable-polynomial"
H_KERNEL_PARAMS = {"constant": KP(1, learnable=False), "exponent": KP(3.0, learnable=False)}
H_KERNEL_NUGGET = 0.001

# The following is applied to all components of p in the system
P_KERNEL = "gaussian"
P_KERNEL_PARAMS = {"scale": KP(1.0, learnable=True)}
P_KERNEL_NUGGET = 1e-5

# The following is applied to all components of q in the system
Q_KERNEL = "gaussian"
Q_KERNEL_PARAMS = {"scale": KP(1.0, learnable=True)}
Q_KERNEL_NUGGET = 1e-5

TRUE_COEFS = jnp.asarray([0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 1, 0, -(1/3)])

TRAIN_BANDS = [(0.2, 1.2), (2.0, 3.0)]
TEST_BANDS = [(1.2, 2.0), (3.0, 3.7)]


def __reduce_mask(nd_mask):
        reduced_mask = nd_mask
        if nd_mask.ndim > 1:
            reduced_mask = np.multiply.reduce(nd_mask, axis=1).astype(bool)
        return reduced_mask


def _adjust_y_train_shape(matrix, y_train):
    _, matrix_trainling_dim_size = matrix.shape
    observation_leading_dim_size, *_ = y_train.shape

    if matrix_trainling_dim_size != observation_leading_dim_size:
        y_train = jnp.reshape(y_train, (matrix_trainling_dim_size, ), order='F')

    return y_train

def _adjust_sims_shape(matrix, sims):
    _, matrix_trainling_dim_size = matrix.shape
    *_, sims_trailing_dim_size = sims.shape

    if matrix_trainling_dim_size != sims_trailing_dim_size:
        sims = jnp.reshape(sims, (matrix_trainling_dim_size, ), order='F')

    return sims

def one_step_hamiltonian(Z, observations_end, num_trajectories=1):
    p_kernel = KernelsFactory.create(P_KERNEL, P_KERNEL_PARAMS, P_KERNEL_NUGGET)
    q_kernel = KernelsFactory.create(Q_KERNEL, Q_KERNEL_PARAMS, Q_KERNEL_NUGGET)
    H_kernel = KernelsFactory.create(H_KERNEL, H_KERNEL_PARAMS, H_KERNEL_NUGGET, linear_functional=jax.jacobian)

    _, pq_size = Z.shape
    ndims = (pq_size - 6) // 2
    p_start, p_end = 5, ndims + 5
    q_start, q_end = ndims + 5, 2 * ndims + 5

    T_train = Z[:observations_end * num_trajectories, 0:5]
    P_train = Z[:observations_end * num_trajectories, p_start:p_end]
    Q_train = Z[:observations_end * num_trajectories, q_start:q_end]

    def p(t):
        sims = p_kernel(t, T_train)
        matrix = p_kernel.matrix(T_train)

        sims = _adjust_sims_shape(matrix, sims)
        y_train = _adjust_y_train_shape(matrix, P_train)

        return sims @ jnp.linalg.solve(matrix, y_train)
    
    def q(t):
        sims = q_kernel(t, T_train)
        matrix = q_kernel.matrix(T_train)

        sims = _adjust_sims_shape(matrix, sims)
        y_train = _adjust_y_train_shape(matrix, Q_train)

        return sims @ jnp.linalg.solve(matrix, y_train)
    
    p_dot = jax.jacobian(p)
    q_dot = jax.jacobian(q)

    PQ_train = jnp.concatenate((P_train, Q_train), axis=1)
    QP_DOT_train = jnp.concatenate((
            jnp.squeeze(jax.jit(jax.vmap(q_dot))(T_train))[:, :, 0],
            jnp.squeeze(-jax.jit(jax.vmap(p_dot))(T_train))[:, :, 0]
        ),
        axis=1
    )

    H_matrix = H_kernel.matrix(PQ_train)
    H_y_train = _adjust_y_train_shape(H_matrix, QP_DOT_train)
    inv = jnp.linalg.solve(H_matrix, H_y_train)

    def H(pq):
        sims = H_kernel(pq, PQ_train)

        sims = _adjust_sims_shape(H_matrix, sims)

        return jnp.squeeze(sims @ inv)
    
    return H

def two_steps_pqH(X, M, observations_end, use_T_in_second_step=False, num_trajectories=1):
    p_kernel = KernelsFactory.create(P_KERNEL, P_KERNEL_PARAMS, P_KERNEL_NUGGET)
    q_kernel = KernelsFactory.create(Q_KERNEL, Q_KERNEL_PARAMS, Q_KERNEL_NUGGET)
    H_kernel = KernelsFactory.create(H_KERNEL, H_KERNEL_PARAMS, H_KERNEL_NUGGET, linear_functional=jax.jacobian)


    _, pq_size = X.shape
    ndims = (pq_size - 6) // 2
    p_start, p_end = 5, ndims + 5
    q_start, q_end = ndims + 5, 2 * ndims + 5
    true_observations_mask = __reduce_mask(M[:observations_end * num_trajectories, 5:pq_size - 1])

    T_ghost = X[:observations_end * num_trajectories, 0:5]
    T_train = X[:observations_end * num_trajectories, 0:5][true_observations_mask]
    P_train = X[:observations_end * num_trajectories, p_start:p_end][true_observations_mask]
    Q_train = X[:observations_end * num_trajectories, q_start:q_end][true_observations_mask]

    def p(t):
        sims = p_kernel(t, T_train)
        matrix = p_kernel.matrix(T_train)

        sims = _adjust_sims_shape(matrix, sims)
        y_train = _adjust_y_train_shape(matrix, P_train)

        return sims @ jnp.linalg.solve(matrix, y_train)
    
    def q(t):
        sims = q_kernel(t, T_train)
        matrix = q_kernel.matrix(T_train)

        sims = _adjust_sims_shape(matrix, sims)
        y_train = _adjust_y_train_shape(matrix, Q_train)

        return sims @ jnp.linalg.solve(matrix, y_train)
    
    p_dot = jax.jacobian(p)
    q_dot = jax.jacobian(q)

    if use_T_in_second_step:
        PQ_train = jnp.concatenate((jax.jit(jax.vmap(p))(T_ghost), jax.jit(jax.vmap(q))(T_ghost)), axis=1)
        QP_DOT_train = jnp.concatenate((
                jnp.squeeze(jax.jit(jax.vmap(q_dot))(T_ghost))[:, :, 0], 
                jnp.squeeze(-jax.jit(jax.vmap(p_dot))(T_ghost))[:, :, 0])
            , axis=1)
    else:
        PQ_train = jnp.concatenate((P_train, Q_train), axis=1)
        QP_DOT_train = jnp.concatenate((
                jnp.squeeze(jax.jit(jax.vmap(q_dot))(T_train))[:, :, 0],
                jnp.squeeze(-jax.jit(jax.vmap(p_dot))(T_train))[:, :, 0]
            )
        )


    H_matrix = H_kernel.matrix(PQ_train)
    H_y_train = _adjust_y_train_shape(H_matrix, QP_DOT_train)
    inv = jnp.linalg.solve(H_matrix, H_y_train)

    def H(pq):
        sims = H_kernel(pq, PQ_train)

        sims = _adjust_sims_shape(H_matrix, sims)

        return jnp.squeeze(sims @ inv)
    
    return p, q, H


def integerate_H_est(H_est, init_point, time, dims=1):
    
    H_grad = jax.jit(jax.grad(H_est))
    
    def kernel_ode(pq, t):
        grad_val = H_grad(pq)
        dpH = grad_val[0: dims]
        dqH = grad_val[dims: 2 * dims]

        return np.concatenate([-dqH, dpH])
    
    pq = odeint(kernel_ode, init_point, time)

    return pq

def initialize_X_for_one_step(X, M, p_fn, q_fn, H_fn, obssrvations_end, num_trajectories, dims=1):
    
    _, pq_size = X.shape
    X_initialized = X.copy()
    extra_steps = N - obssrvations_end

    T = X[:, 0]
    true_observations_mask = __reduce_mask(M[:obssrvations_end * num_trajectories, 5:pq_size - 1])
    
    for i in range(num_trajectories):

        init_point = X[obssrvations_end * (i + 1)  - 1, 5:pq_size - 1]
        
        extrapolation_time_start = obssrvations_end * num_trajectories
        extrapolation_time = np.concatenate((
            np.reshape(T[obssrvations_end * (i + 1) - 1], (1,)), 
            T[extrapolation_time_start + extra_steps * i: extrapolation_time_start + extra_steps * (i + 1)]
        ), axis=0)

        pq_2s = integerate_H_est(H_fn, init_point, extrapolation_time, dims=dims)

        X_initialized[extrapolation_time_start + extra_steps * i: extrapolation_time_start + extra_steps * (i + 1), 5:pq_size - 1] = pq_2s[1:, :]

    pq_interpolated = jnp.concatenate((
            jax.jit(jax.vmap(p_fn))(X[:obssrvations_end * num_trajectories, 0:5]),
            jax.jit(jax.vmap(q_fn))(X[:obssrvations_end * num_trajectories, 0:5])
        ),
        axis=1
    )

    X_initialized[:obssrvations_end * num_trajectories, 5:pq_size - 1][~true_observations_mask] = pq_interpolated[~true_observations_mask, :]
    H_init = jax.jit(jax.vmap(H_fn))(X_initialized[:, 5:pq_size - 1])
    X_initialized[:, pq_size - 1] = H_init

    return X_initialized



def initalize_one_step_loss_multipliers(graph, X_init, M):
    _, separate_losses = graph._loss(X_init, X_init, M, return_separate=True)
    rkhs_order_magnitude = np.floor(np.log10(separate_losses.get("rkhs_norm")))
    unk_funcs_order_magnitude = np.floor(np.log10(separate_losses.get("unknown_funcs_loss")))
    constraints_order_magnitude = np.floor(np.log10(separate_losses.get("constraints_loss")))
    data_compliance_order_magnitude = np.floor(np.log10(separate_losses.get("data_compliance_loss")))

    unknown_functions_loss_multiplier = 10 ** (rkhs_order_magnitude - unk_funcs_order_magnitude)
    constraint_loss_multiplier = 10 ** (rkhs_order_magnitude - constraints_order_magnitude)
    data_compliance_loss_multiplier = 10 ** (rkhs_order_magnitude - data_compliance_order_magnitude)

    return unknown_functions_loss_multiplier, constraint_loss_multiplier, data_compliance_loss_multiplier


def get_seprable_H_coef(Z, observations_end, num_trajectories = 1):

    H_kernel = KernelsFactory.create(H_KERNEL, H_KERNEL_PARAMS, H_KERNEL_NUGGET)

    _, pq_size = Z.shape
    ndims = (pq_size - 6) // 2
    p_start, p_end = 5, ndims + 5
    q_start, q_end = ndims + 5, 2 * ndims + 5

    H_estimated = Z[:observations_end * num_trajectories, -1]
    P_train = Z[:observations_end * num_trajectories, p_start:p_end]
    Q_train = Z[:observations_end * num_trajectories, q_start:q_end]
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
            
    return w_p

def mse(truth: np.ndarray, predictions: np.ndarray, ):
    return np.mean((predictions - truth) ** 2)

def relative_error(truth: np.ndarray, predictions: np.ndarray):
    return np.linalg.norm((truth - predictions), ord=2) / np.linalg.norm(truth, ord=2)

def h_true(p1, p2, q1, q2):
    return 0.5 * (q1 ** 2 + q2 ** 2 + (q2 - q1) ** 2 + p1 ** 2 + p2 ** 2)

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

def generate_trajectory(p0q0):
    return odeint(m2s3_system_ode, p0q0, t)


def generate_data():

    rng = np.random.default_rng(0)

    test_trajectories = []
    training_trajectories = []
    test_initial_conds = []
    training_initial_conds = []

    generated_test_trajectories_count = 0
    generated_train_trajectories_count = 0

    attempts = 0

    while True:

        if (generated_test_trajectories_count == N_TRAJICTORIES_TEST) and (generated_train_trajectories_count == MAX_N_TRAJICTORIES_TRAIN):
            break

        p10 = rng.uniform(-1, 1)
        q10 = rng.uniform(-1, 1)
        p20 = rng.uniform(-1, 1)
        q20 = rng.uniform(-1, 1)

        energy = h_true(p10, p20, q10, q20)

        is_training_trajectory = any(low < energy < high for low, high in TRAIN_BANDS)
        is_test_trajectory = any(low < energy < high for low, high in TEST_BANDS)

        if is_test_trajectory or is_training_trajectory:
            assert is_test_trajectory != is_training_trajectory

        if is_training_trajectory and (generated_train_trajectories_count < MAX_N_TRAJICTORIES_TRAIN):
            training_initial_conds.append((p10, p20, q10, q20))
            training_trajectories.append(generate_trajectory((p10, p20, q10, q20)))
            generated_train_trajectories_count += 1

        if is_test_trajectory and (generated_test_trajectories_count < N_TRAJICTORIES_TEST):
            test_initial_conds.append((p10, p20, q10, q20))
            test_trajectories.append(generate_trajectory((p10, p20, q10, q20)))
            generated_test_trajectories_count += 1

        attempts += 1

    print(f"Generated {N_TRAJICTORIES_TEST} test trajectories and {MAX_N_TRAJICTORIES_TRAIN} training trajectories in {attempts} attempts.")

    return training_trajectories, training_initial_conds, test_trajectories, test_initial_conds


def generate_Xs_M(training_trajectories, training_initial_conds, seed, sparsity_factor, num_trajectories):

    all_trajectories_observed = []
    all_trajectories_hidden = []
    all_p0q0_observed = []
    all_p0q0_hidden = []
    all_H_observed = []
    all_H_hidden = []

    t = np.linspace(0, T_MAX, N)

    for p0q0, pq in zip(training_initial_conds, training_trajectories):
        p10, p20, q10, q20 = pq.T
        H = h_true(p10, p20, q10, q20)

        all_trajectories_observed.append(pq[:OBSERVATIONS_END, :])
        all_trajectories_hidden.append(pq[OBSERVATIONS_END:, :])
        all_p0q0_observed.append(np.tile(p0q0, (OBSERVATIONS_END, 1)))
        all_p0q0_hidden.append(np.tile(p0q0, (N - OBSERVATIONS_END, 1)))
        all_H_observed.append(H[:OBSERVATIONS_END, np.newaxis])
        all_H_hidden.append(H[OBSERVATIONS_END:, np.newaxis])


    X_true = np.concatenate([
        np.concatenate(
            [np.tile(t[:OBSERVATIONS_END], num_trajectories)[:, np.newaxis], 
            np.tile(t[OBSERVATIONS_END:], num_trajectories)[:, np.newaxis]],
            axis=0
        ),
        np.concatenate(
            [np.concatenate(all_p0q0_observed, axis=0), 
            np.concatenate(all_p0q0_hidden, axis=0)],
            axis=0
        ),
        np.concatenate(
            [np.concatenate(all_trajectories_observed, axis=0), 
            np.concatenate(all_trajectories_hidden, axis=0)],
            axis=0
        ),
        np.concatenate(
            [np.concatenate(all_H_observed, axis=0),
            np.concatenate(all_H_hidden, axis=0)],
            axis=0
        )],
        axis=1
    )

    M = np.ones_like(X_true).astype(bool)

    rng = np.random.default_rng(seed)
    for i in range(num_trajectories):
        sparse_mask = rng.choice([False, True], p=[sparsity_factor, 1 - sparsity_factor], size=OBSERVATIONS_END)
        sparse_mask[-1] = True

        M[OBSERVATIONS_END * i: OBSERVATIONS_END * (i + 1), 5] = sparse_mask
        M[OBSERVATIONS_END * i: OBSERVATIONS_END * (i + 1), 6] = sparse_mask
        M[OBSERVATIONS_END * i: OBSERVATIONS_END * (i + 1), 7] = sparse_mask
        M[OBSERVATIONS_END * i: OBSERVATIONS_END * (i + 1), 8] = sparse_mask

    M[OBSERVATIONS_END * num_trajectories:, 3:5] = False
    M[:, 9] = False

    X = np.full_like(X_true, fill_value=0)
    X[M] = X_true[M]

    return X_true, X, M


def generate_graph():

    hh_graph = ComputationalGraph(observables_order=["t", "p10", "p20", "q10", "q20", "p1", "p2", "q1", "q2", "H"])

    hh_graph.add_observable("t")
    hh_graph.add_observable("p10")
    hh_graph.add_observable("p20")
    hh_graph.add_observable("q10")
    hh_graph.add_observable("q20")

    hh_graph.add_aggregator(["t", "p10", "p20", "q10", "q20"], "tp0q0")

    hh_graph.add_unknown_fn("tp0q0", "q1", alpha=P_KERNEL_NUGGET, kernel=P_KERNEL, kernel_parameters=P_KERNEL_PARAMS)
    hh_graph.add_unknown_fn("tp0q0", "p1", alpha=Q_KERNEL_NUGGET, kernel=Q_KERNEL, kernel_parameters=Q_KERNEL_PARAMS)
    hh_graph.add_unknown_fn("tp0q0", "q2", alpha=P_KERNEL_NUGGET, kernel=P_KERNEL, kernel_parameters=P_KERNEL_PARAMS)
    hh_graph.add_unknown_fn("tp0q0", "p2", alpha=Q_KERNEL_NUGGET, kernel=Q_KERNEL, kernel_parameters=Q_KERNEL_PARAMS)

    hh_graph.add_known_fn("p1", "p1_grad", derivative)
    hh_graph.add_known_fn("p2", "p2_grad", derivative)
    hh_graph.add_known_fn("q1", "q1_grad", derivative)
    hh_graph.add_known_fn("q2", "q2_grad", derivative)

    hh_graph.add_known_fn("p1_grad", "p1_dot", lambda p1_grad: p1_grad[:, 0])
    hh_graph.add_known_fn("q1_grad", "q1_dot", lambda q1_grad: q1_grad[:, 0])
    hh_graph.add_known_fn("p2_grad", "p2_dot", lambda p2_grad: p2_grad[:, 0])
    hh_graph.add_known_fn("q2_grad", "q2_dot", lambda q2_grad: q2_grad[:, 0])

    hh_graph.add_aggregator(["q1_dot", "q2_dot"], "q_dot")
    hh_graph.add_aggregator(["p1_dot", "p2_dot"], "p_dot")
    hh_graph.add_known_fn("p_dot", "-p_dot", lambda p_dot: -p_dot)

    hh_graph.add_aggregator(["q_dot", "-p_dot"], "qp_dot")
    hh_graph.add_aggregator(["p1", "p2", "q1", "q2"], "pq")

    hh_graph.add_unknown_fn("pq", "H", linear_functional=jax.jacobian, observations="qp_dot", alpha=H_KERNEL_NUGGET, kernel=H_KERNEL, kernel_parameters=H_KERNEL_PARAMS)
    hh_graph.add_known_fn("H", "grad_H", derivative)

    hh_graph.add_aggregator(["p_dot", "grad_H"], "(p_dot, grad_H)")
    def p_dot_constraint(p_dot_grad_H):
        p_dot, grad_H = p_dot_grad_H[:, :2], p_dot_grad_H[:, 2:]
        return p_dot + grad_H[:, 2:]

    hh_graph.add_aggregator(["q_dot", "grad_H"], "(q_dot, grad_H)")
    def q_dot_constraint(q_dot_grad_H):
        q_dot, grad_H = q_dot_grad_H[:, :2], q_dot_grad_H[:, 2:]
        return q_dot - grad_H[:, :2]

    hh_graph.add_constraint("(p_dot, grad_H)", "W1", p_dot_constraint)
    hh_graph.add_constraint("(q_dot, grad_H)", "W2", q_dot_constraint)

    return hh_graph


def get_test_errors(H_est, test_initial_conds, test_trajectories):

    p_mse_errors = []
    p_re_errors = []
    q_mse_errors = []
    q_re_errors = []

    t = np.linspace(0, T_MAX, N)

    for init_cond, pq_true in tqdm(zip(test_initial_conds, test_trajectories)):

        pq_pred = integerate_H_est(H_est, init_cond, t, dims=2)
        p_mse_errors.append(mse(pq_true[:, 0], pq_pred[:, 0]))
        p_re_errors.append(relative_error(pq_true[:, 0], pq_pred[:, 0]) * 100)
        q_mse_errors.append(mse(pq_true[:, 1], pq_pred[:, 1]))
        q_re_errors.append(relative_error(pq_true[:, 1], pq_pred[:, 1]) * 100)

    return p_mse_errors, p_re_errors, q_mse_errors, q_re_errors


def run_for(n_trajectories, sparsity_factor, seed, train_set, test_set):
    training_initial_conds, training_trajectories = train_set 
    test_initial_conds, test_trajectories = test_set

    X_true, X, M = generate_Xs_M(training_trajectories[:n_trajectories], training_initial_conds[:n_trajectories], seed, sparsity_factor, n_trajectories)
    graph = generate_graph()

    p_2s, q_2s, H_2s = two_steps_pqH(X, M, OBSERVATIONS_END, num_trajectories=n_trajectories, use_T_in_second_step=True)
    X_init = initialize_X_for_one_step(X, M, p_2s, q_2s, H_2s, OBSERVATIONS_END, n_trajectories, dims=2)
    unknown_functions_loss_multiplier, constraint_loss_multiplier, data_compliance_loss_multiplier = initalize_one_step_loss_multipliers(graph, X_init, M)

    estimated_coefs_2s = get_seprable_H_coef(X_init, OBSERVATIONS_END, n_trajectories)

    coefs_error_2s = relative_error(TRUE_COEFS, estimated_coefs_2s)
    with jax.default_device(cpu_device):
        p_mse_errors_2s, p_re_errors_2s, q_mse_errors_2s, q_re_errors_2s = get_test_errors(H_2s, test_initial_conds, test_trajectories)

    graph.set_loss_multipliers(unknown_functions_loss_multiplier, constraint_loss_multiplier, data_compliance_loss_multiplier)

    Z = graph.complete(
        X_init[:OBSERVATIONS_END * (n_trajectories), :], 
        M[:OBSERVATIONS_END * (n_trajectories), :], 
        OBSERVATIONS_END * (n_trajectories), optimizer="l-bfgs-b", learn_parameters=False, n_rounds=1)

    H_1s = one_step_hamiltonian(Z, OBSERVATIONS_END, num_trajectories=n_trajectories)

    estimated_coefs_1s = get_seprable_H_coef(Z, OBSERVATIONS_END, n_trajectories)

    coefs_error_1s = relative_error(TRUE_COEFS, estimated_coefs_1s)
    with jax.default_device(cpu_device):
        p_mse_errors_1s, p_re_errors_1s, q_mse_errors_1s, q_re_errors_1s = get_test_errors(H_1s, test_initial_conds, test_trajectories)

    return (
        coefs_error_2s, p_mse_errors_2s, p_re_errors_2s, q_mse_errors_2s, q_re_errors_2s,
        coefs_error_1s, p_mse_errors_1s, p_re_errors_1s, q_mse_errors_1s, q_re_errors_1s
    )

if __name__ == "__main__":

    experiment_dir = f"m2s3-up-to-{MAX_N_TRAJICTORIES_TRAIN}-with-{N_ROUNDS}-rounds"
    os.makedirs(experiment_dir, exist_ok=True)

    training_trajectories, training_initial_conds, test_trajectories, test_initial_conds = generate_data()
    sparsity_factors = [0, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for factor in sparsity_factors:

        results_dict = {
            "2S Coefs Error Mean": [],
            "2S Coefs Error STD": [],
            "2S P RE Mean": [],
            "2S P RE STD": [],
            "2S Q RE Mean": [],
            "2S Q RE STD": [],
            "2S P MSE Mean": [],
            "2S P MSE STD": [],
            "2S Q MSE Mean": [],
            "2S Q MSE STD": [],
            "1S Coefs Error Mean": [],
            "1S Coefs Error STD": [],
            "1S P RE Mean": [],
            "1S P RE STD": [],
            "1S Q RE Mean": [],
            "1S Q RE STD": [],
            "1S P MSE Mean": [],
            "1S P MSE STD": [],
            "1S Q MSE Mean": [],
            "1S Q MSE STD": [],
        }

        for n_training in range(MAX_N_TRAJICTORIES_TRAIN):

            seed = 0
            num_runs = 0

            p_mse_1s = []
            p_mse_2s = []
            q_mse_1s = []
            q_mse_2s = []
            p_re_1s = []
            p_re_2s = []
            q_re_1s = []
            q_re_2s = []
            coefs_1s = []
            coefs_2s = []

            while (num_runs < (N_ROUNDS if factor > 0 else 1)):
                try:
                    errors = run_for(
                        n_training + 1,
                        factor,
                        seed,
                        (training_initial_conds, training_trajectories),
                        (test_initial_conds, test_trajectories)
                    )
                except (ODEintWarning, RuntimeWarning) as e:
                    seed += 1
                    continue
                
                print(f"Finished round {num_runs + 1} of sparsity factor {factor} with {n_training + 1} trajectories.")

                coefs_error_2s, p_mse_errors_2s, p_re_errors_2s, q_mse_errors_2s, q_re_errors_2s, \
                coefs_error_1s, p_mse_errors_1s, p_re_errors_1s, q_mse_errors_1s, q_re_errors_1s = errors

                p_mse_1s.append(np.mean(p_mse_errors_1s))
                p_mse_2s.append(np.mean(p_mse_errors_2s))
                q_mse_1s.append(np.mean(q_mse_errors_1s))
                q_mse_2s.append(np.mean(q_mse_errors_2s))
                p_re_1s.append(np.mean(p_re_errors_1s))
                p_re_2s.append(np.mean(p_re_errors_2s))
                q_re_1s.append(np.mean(q_re_errors_1s))
                q_re_2s.append(np.mean(q_re_errors_2s))
                coefs_1s.append(coefs_error_1s)
                coefs_2s.append(coefs_error_2s)

                num_runs += 1
                seed += 1


            results_dict["2S Coefs Error Mean"].append(np.mean(coefs_2s))
            results_dict["2S Coefs Error STD"].append(np.std(coefs_2s))
            results_dict["2S P RE Mean"].append(np.mean(p_re_2s))
            results_dict["2S P RE STD"].append(np.std(p_re_2s))
            results_dict["2S P MSE Mean"].append(np.mean(p_mse_2s))
            results_dict["2S P MSE STD"].append(np.std(p_mse_2s))
            results_dict["2S Q RE Mean"].append(np.mean(q_re_2s))
            results_dict["2S Q RE STD"].append(np.std(q_re_2s))
            results_dict["2S Q MSE Mean"].append(np.mean(q_mse_2s))
            results_dict["2S Q MSE STD"].append(np.std(q_mse_2s))

            results_dict["1S Coefs Error Mean"].append(np.mean(coefs_1s))
            results_dict["1S Coefs Error STD"].append(np.std(coefs_1s))
            results_dict["1S P RE Mean"].append(np.mean(p_re_1s))
            results_dict["1S P RE STD"].append(np.std(p_re_1s))
            results_dict["1S P MSE Mean"].append(np.mean(p_mse_1s))
            results_dict["1S P MSE STD"].append(np.std(p_mse_1s))
            results_dict["1S Q RE Mean"].append(np.mean(q_re_1s))
            results_dict["1S Q RE STD"].append(np.std(q_re_1s))
            results_dict["1S Q MSE Mean"].append(np.mean(q_mse_1s))
            results_dict["1S Q MSE STD"].append(np.std(q_mse_1s))

        df = pd.DataFrame(results_dict)
        df.to_csv(f"{experiment_dir}/{factor}-sparse.csv")

