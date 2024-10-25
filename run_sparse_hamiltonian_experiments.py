import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from vorpy.symplectic_integration.nonseparable_hamiltonian import integrate, heuristic_estimate_for_omega
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from cgc.graph import ComputationalGraph, derivative
from cgc.optimizers import TwoStepsNGDptimizerForKF
from cgc.utils import KernelParameter as KP

from jax import config
config.update("jax_enable_x64", True)


N = 400
T_MAX = 80
OBSERVATIONS_END = 200
KFLOW_LEARNING = False

H_KERNEL = "gaussian"
H_KERNEL_PARAMS = {"scale": KP(1.0)}
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

def two_steps_initialization(graph: ComputationalGraph, X, M, observations_end):

    _, pq_size = X.shape

    true_observations_mask = __reduce_mask(M[:observations_end, 1:pq_size - 1])

    H_est = get_hamiltonian_from(graph, X[:observations_end, :][true_observations_mask])
    pq_est = get_pq_from(graph, X[:observations_end, :][true_observations_mask])
    H_grad = jax.jit(jax.grad(H_est))

    def ms_kernel_ode(pq, t):
        dims = (pq_size - 2) // 2
        grad_val = H_grad(pq)
        dpH = grad_val[0: dims]
        dqH = grad_val[dims: 2 * dims]

        return np.concatenate([-dqH, dpH])

    init_point = X[observations_end - 1, 1:pq_size - 1]

    t = X[:, 0]
    pq_2s = odeint(ms_kernel_ode, init_point, t[observations_end - 1:])
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
    return np.mean((predictions - truth) ** 2)


def relative_error(predictions: np.ndarray, truth: np.ndarray):
    return np.mean(np.abs((predictions - truth)) / (np.abs(truth) + np.abs(predictions)))



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

    return X_true


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

    return X_true


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

    return X_true


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

    return X_true


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


def run_for(data_generator, graph_generator, sparsity_factor, seed):

    X_true = data_generator()
    graph = graph_generator()
    X, M = generate_X_and_M(X_true, sparsity_factor, seed)
    
    _, size = X_true.shape
    ndims = (size - 2) // 2
    p_start, p_end = 1, ndims + 1
    q_start, q_end = ndims + 1, 2 * ndims + 1
    
    if KFLOW_LEARNING:
        run_kflow_for(graph, X, M, OBSERVATIONS_END)
    
    X_init, unknown_functions_loss_multiplier, constraint_loss_multiplier, data_compliance_loss_multiplier = two_steps_initialization(graph, X, M, OBSERVATIONS_END)

    two_steps_mse_p = mse(X_true[OBSERVATIONS_END:, p_start:p_end], X_init[OBSERVATIONS_END:, p_start:p_end])
    two_steps_mse_q = mse(X_true[OBSERVATIONS_END:, q_start:q_end], X_init[OBSERVATIONS_END:, q_start:q_end])
    two_steps_re_p = relative_error(X_true[OBSERVATIONS_END:, p_start:p_end], X_init[OBSERVATIONS_END:, p_start:p_end]) * 100
    two_steps_re_q = relative_error(X_true[OBSERVATIONS_END:, q_start:q_end], X_init[OBSERVATIONS_END:, q_start:q_end]) * 100

    graph.set_loss_multipliers(unknown_functions_loss_multiplier, constraint_loss_multiplier, data_compliance_loss_multiplier=0)

    Z = graph.complete(X_init, M, OBSERVATIONS_END, optimizer='l-bfgs-b', learn_parameters=False, n_rounds=1)

    one_step_mse_p = mse(X_true[OBSERVATIONS_END:, p_start:p_end], Z[OBSERVATIONS_END:, p_start:p_end])
    one_step_mse_q = mse(X_true[OBSERVATIONS_END:, q_start:q_end], Z[OBSERVATIONS_END:, q_start:q_end])
    one_step_re_p = relative_error(X_true[OBSERVATIONS_END:, p_start:p_end], Z[OBSERVATIONS_END:, p_start:p_end]) * 100
    one_step_re_q = relative_error(X_true[OBSERVATIONS_END:, q_start:q_end], Z[OBSERVATIONS_END:, q_start:q_end]) * 100

    return (
        two_steps_re_p, two_steps_re_q,
        one_step_re_p, one_step_re_q,
        two_steps_mse_p, two_steps_mse_q,
        one_step_mse_p, one_step_mse_q
    )


if __name__ == "__main__":

    exp_dir = f"{OBSERVATIONS_END}-{H_KERNEL}"

    columns = [
        "Two-Steps P RE",
        "Two-Steps Q RE",
        "One-Step P RE",
        "One-Step Q RE",
        "Two-Steps P MSE",
        "Two-Steps Q MSE",
        "One-Step P MSE",
        "One-Step Q MSE"
    ]

    experiements = [
        ("ms", generate_ms_data, build_1d_graph),
        ("m2s3", generate_m2s3_data, build_2d_graph),
        ("hh", generate_hh_data, build_2d_graph),
        ("np", generate_np_data, build_1d_graph)
    ]

    for (sys_name, data_generator, graph_builder) in experiements:

        sys_dir = f"{exp_dir}/{sys_name}"
        os.makedirs(sys_dir, exist_ok=True)

        for sparsity_factor in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print(f"Running experiments for {sys_name} @ sparsity of {sparsity_factor}")
            results_fname = f"sparse-{sparsity_factor}.csv"

            results_dict = {c: [] for c in columns}
            num_rounds = 10 if sparsity_factor > 0.0 else 1

            for i in range(num_rounds):
                errors = run_for(data_generator, graph_builder, sparsity_factor, seed=i)
                for col, val in zip(columns, errors):
                    results_dict[col].append(val)

            for col in columns:
                mean = np.mean(results_dict[col])
                std = np.std(results_dict[col])

                results_dict[col].extend([mean, std])

            results_df = pd.DataFrame(results_dict, index=[f"{i}" for i in range(num_rounds)] + ["Mean", "STD"])
            results_df.to_csv(f"{sys_dir}/{results_fname}")




