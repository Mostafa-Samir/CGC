import os
from collections import namedtuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PhaseSpacePlot = namedtuple("PhaseSpacePlot", ["p", "q", "M"])
TrajectoryPlot = namedtuple("QPPlot", ["t", "truth", "pred_1", "pred_2", "M"])
Plot = namedtuple("Plot", ["plot_data", "title", "x_label", "y_label", "fname"])

def plot_phase_space(plot_data: PhaseSpacePlot):
    plt.plot(plot_data.q, plot_data.p, color="black")
    plt.scatter(plot_data.q[plot_data.M], plot_data.p[plot_data.M], marker='x', color='red', label='Observations')

def plot_trajectory(plot_data: TrajectoryPlot):
    plt.plot(plot_data.t, plot_data.truth, label="Truth")
    if not np.all(~plot_data.M):
        plt.scatter(plot_data.t[plot_data.M], plot_data.truth[plot_data.M], marker='x', color='red', label='Observations')
    plt.plot(plot_data.t, plot_data.pred_2, label="2-Steps Predictions", linestyle='--')
    plt.plot(plot_data.t, plot_data.pred_1, label="1-Step Predictions", linestyle=':')

def get_subtype_plotter(plot_data):
    if isinstance(plot_data, PhaseSpacePlot):
        return plot_phase_space
    if isinstance(plot_data, TrajectoryPlot):
        return plot_trajectory

def plot(plot_def: Plot):
    plt.figure()
    plotter = get_subtype_plotter(plot_def.plot_data)
    plotter(plot_def.plot_data)
    plt.title(plot_def.title)
    plt.xlabel(plot_def.x_label)
    plt.ylabel(plot_def.y_label)
    plt.legend()


experiments_dir = "200-separable-polynomial-with-data-adjusted-and-re-l2-and-h-coef-err"
plots_dir = f"{experiments_dir}-plots"
os.makedirs(plots_dir, exist_ok=True)

for system in ["ms", "m2s3", "hh", "np"]:
    system_dir = f"{plots_dir}/{system}"
    os.makedirs(system_dir, exist_ok=True)

    error_plots_dir = f"{system_dir}/error-plots"
    os.makedirs(error_plots_dir, exist_ok=True)

    trajectory_files = ["X_true.npy", "best_M.npy", "best_X_2d.npy", "best_Z.npy"]
    if (system == "m2s3") and ('polynomial' in experiments_dir):
        values = [0.0, 0.5, 0.6, 0.7, 0.8]
    else:
        values = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    avg_2_re_p_int = []
    std_2_re_p_int = []
    avg_1_re_p_int = []
    std_1_re_p_int = []
    avg_2_re_q_int = []
    std_2_re_q_int = []
    avg_1_re_q_int = []
    std_1_re_q_int = []
    avg_2_re_h_int = []
    std_2_re_h_int = []
    avg_1_re_h_int = []
    std_1_re_h_int = []
    avg_2_re_p_ext = []
    std_2_re_p_ext = []
    avg_1_re_p_ext = []
    std_1_re_p_ext = []
    avg_2_re_q_ext = []
    std_2_re_q_ext = []
    avg_1_re_q_ext = []
    std_1_re_q_ext = []
    avg_2_re_h_ext = []
    std_2_re_h_ext = []
    avg_1_re_h_ext = []
    std_1_re_h_ext = []
    avg_1_coef_h = []
    std_1_coef_h = []
    avg_2_coef_h = []
    std_2_coef_h = []
    
    for val in values:
        path = f"{experiments_dir}/{system}/sparse-{val}.csv"
        data = pd.read_csv(path, index_col=0)
        avg_2_re_p_int.append(data.loc["Mean", "Two-Steps P RE (Interpolation)"])
        avg_1_re_p_int.append(data.loc["Mean", "One-Step P RE (Interpolation)"])
        avg_2_re_q_int.append(data.loc["Mean", "Two-Steps Q RE (Interpolation)"])
        avg_1_re_q_int.append(data.loc["Mean", "One-Step Q RE (Interpolation)"])
        avg_2_re_h_int.append(data.loc["Mean", "Two-Step H RE (Interpolation)"])
        avg_1_re_h_int.append(data.loc["Mean", "One-Step H RE (Interpolation)"])
        std_2_re_p_int.append(data.loc["STD", "Two-Steps P RE (Interpolation)"])
        std_1_re_p_int.append(data.loc["STD", "One-Step P RE (Interpolation)"])
        std_2_re_q_int.append(data.loc["STD", "Two-Steps Q RE (Interpolation)"])
        std_1_re_q_int.append(data.loc["STD", "One-Step Q RE (Interpolation)"])
        std_2_re_h_int.append(data.loc["STD", "Two-Step H RE (Interpolation)"])
        std_1_re_h_int.append(data.loc["STD", "One-Step H RE (Interpolation)"])
        avg_2_re_p_ext.append(data.loc["Mean", "Two-Steps P RE (Extrapolation)"])
        avg_1_re_p_ext.append(data.loc["Mean", "One-Step P RE (Extrapolation)"])
        avg_2_re_q_ext.append(data.loc["Mean", "Two-Steps Q RE (Extrapolation)"])
        avg_1_re_q_ext.append(data.loc["Mean", "One-Step Q RE (Extrapolation)"])
        avg_2_re_h_ext.append(data.loc["Mean", "Two-Step H RE (Extrapolation)"])
        avg_1_re_h_ext.append(data.loc["Mean", "One-Step H RE (Extrapolation)"])
        std_2_re_p_ext.append(data.loc["STD", "Two-Steps P RE (Extrapolation)"])
        std_1_re_p_ext.append(data.loc["STD", "One-Step P RE (Extrapolation)"])
        std_2_re_q_ext.append(data.loc["STD", "Two-Steps Q RE (Extrapolation)"])
        std_1_re_q_ext.append(data.loc["STD", "One-Step Q RE (Extrapolation)"])
        std_2_re_h_ext.append(data.loc["STD", "Two-Step H RE (Extrapolation)"])
        std_1_re_h_ext.append(data.loc["STD", "One-Step H RE (Extrapolation)"])
        if ("poly" in experiments_dir) and (system != "np"):
            avg_1_coef_h.append(data.loc["Mean", "One-Step H Coeffecients Error"])
            std_1_coef_h.append(data.loc["STD", "One-Step H Coeffecients Error"])
            avg_2_coef_h.append(data.loc["Mean", "Two-Steps H Coeffecients Error"])
            std_2_coef_h.append(data.loc["STD", "Two-Steps H Coeffecients Error"])

    avg_2_re_p_int = np.array(avg_2_re_p_int)
    std_2_re_p_int = np.array(std_2_re_p_int)
    avg_1_re_p_int = np.array(avg_1_re_p_int)
    std_1_re_p_int = np.array(std_1_re_p_int)
    avg_2_re_q_int = np.array(avg_2_re_q_int)
    std_2_re_q_int = np.array(std_2_re_q_int)
    avg_1_re_q_int = np.array(avg_1_re_q_int)
    std_1_re_q_int = np.array(std_1_re_q_int)
    avg_2_re_h_int = np.array(avg_2_re_h_int)
    std_2_re_h_int = np.array(std_2_re_h_int)
    avg_1_re_h_int = np.array(avg_1_re_h_int)
    std_1_re_h_int = np.array(std_1_re_h_int)
    avg_2_re_p_int = np.array(avg_2_re_p_int)
    std_2_re_p_ext = np.array(std_2_re_p_ext)
    avg_1_re_p_ext = np.array(avg_1_re_p_ext)
    std_1_re_p_ext = np.array(std_1_re_p_ext)
    avg_2_re_q_ext = np.array(avg_2_re_q_ext)
    std_2_re_q_ext = np.array(std_2_re_q_ext)
    avg_1_re_q_ext = np.array(avg_1_re_q_ext)
    std_1_re_q_ext = np.array(std_1_re_q_ext)
    avg_2_re_h_ext = np.array(avg_2_re_h_ext)
    std_2_re_h_ext = np.array(std_2_re_h_ext)
    avg_1_re_h_ext = np.array(avg_1_re_h_ext)
    std_1_re_h_ext = np.array(std_1_re_h_ext)
    if ("poly" in experiments_dir) and (system != "np"):
        avg_1_coef_h = np.array(avg_1_coef_h)
        std_1_coef_h = np.array(std_1_coef_h)
        avg_2_coef_h = np.array(avg_2_coef_h)
        std_2_coef_h = np.array(std_2_coef_h)

    plt.figure()
    plt.plot(values, avg_2_re_p_int, label="Avg. Two-Steps RE", color='blue')
    plt.scatter(values, avg_2_re_p_int, color='blue')
    plt.fill_between(values, np.maximum(avg_2_re_p_int - std_2_re_p_int, 0), avg_2_re_p_int + std_2_re_p_int, alpha=0.5, color='blue')
    plt.plot(values, avg_1_re_p_int, label="Avg. One-Step RE", color='orange')
    plt.scatter(values, avg_1_re_p_int, color='orange')
    plt.fill_between(values, np.maximum(avg_1_re_p_int - std_1_re_p_int, 0), avg_1_re_p_int + std_2_re_p_int, alpha=0.5, color='orange')
    plt.legend()
    plt.xlabel("Sparsity Factor")
    plt.ylabel("Relative Error")
    plt.title(f"2-Steps vs 1-Step Interpolation Error ({system}-P)")
    plt.savefig(f"{error_plots_dir}/p-int-re.png")

    plt.figure()
    plt.plot(values, avg_2_re_p_ext, label="Avg. Two-Steps RE", color='blue')
    plt.scatter(values, avg_2_re_p_ext, color='blue')
    plt.fill_between(values, np.maximum(avg_2_re_p_ext - std_2_re_p_ext, 0), avg_2_re_p_ext + std_2_re_p_ext, alpha=0.5, color='blue')
    plt.plot(values, avg_1_re_p_ext, label="Avg. One-Step RE", color='orange')
    plt.scatter(values, avg_1_re_p_ext, color='orange')
    plt.fill_between(values, np.maximum(avg_1_re_p_ext - std_1_re_p_ext, 0), avg_1_re_p_ext + std_2_re_p_ext, alpha=0.5, color='orange')
    plt.legend()
    plt.xlabel("Sparsity Factor")
    plt.ylabel("Relative Error")
    plt.title(f"2-Steps vs 1-Step Forecasting Error ({system}-P)")
    plt.savefig(f"{error_plots_dir}/p-ext-re.png")
    
    plt.figure()
    plt.plot(values, avg_2_re_q_int, label="Avg. Two-Steps RE", color='blue')
    plt.scatter(values, avg_2_re_q_int, color='blue')
    plt.fill_between(values, np.maximum(avg_2_re_q_int - std_2_re_q_int, 0), avg_2_re_q_int + std_2_re_q_int, alpha=0.5, color='blue')
    plt.plot(values, avg_1_re_q_int, label="Avg. One-Step RE", color='orange')
    plt.scatter(values, avg_1_re_q_int, color='orange')
    plt.fill_between(values, np.maximum(avg_1_re_q_int - std_1_re_q_int, 0), avg_1_re_q_int + std_1_re_q_int, alpha=0.5, color='orange')
    plt.legend()
    plt.xlabel("Sparsity Factor")
    plt.ylabel("Relative Error")
    plt.title(f"2-Steps vs 1-Step Interpolation Error ({system}-Q)")
    plt.savefig(f"{error_plots_dir}/q-int-re.png")
    plt.close()

    plt.figure()
    plt.plot(values, avg_2_re_q_ext, label="Avg. Two-Steps RE", color='blue')
    plt.scatter(values, avg_2_re_q_ext, color='blue')
    plt.fill_between(values, np.maximum(avg_2_re_q_ext - std_2_re_q_ext, 0), avg_2_re_q_ext + std_2_re_q_ext, alpha=0.5, color='blue')
    plt.plot(values, avg_1_re_q_ext, label="Avg. One-Step RE", color='orange')
    plt.scatter(values, avg_1_re_q_ext, color='orange')
    plt.fill_between(values, np.maximum(avg_1_re_q_ext - std_1_re_q_ext, 0), avg_1_re_q_ext + std_1_re_q_ext, alpha=0.5, color='orange')
    plt.legend()
    plt.xlabel("Sparsity Factor")
    plt.ylabel("Relative Error")
    plt.title(f"2-Steps vs 1-Step Forecasting Error ({system}-Q)")
    plt.savefig(f"{error_plots_dir}/q-ext-re.png")
    plt.close()

    plt.figure()
    plt.plot(values, avg_2_re_h_int, label="Avg. Two-Steps RE", color='blue')
    plt.scatter(values, avg_2_re_h_int, color='blue')
    plt.fill_between(values, np.maximum(avg_2_re_h_int - std_2_re_h_int, 0), avg_2_re_h_int + std_2_re_h_int, alpha=0.5, color='blue')
    plt.plot(values, avg_1_re_h_int, label="Avg. One-Step RE", color='orange')
    plt.scatter(values, avg_1_re_h_int, color='orange')
    plt.fill_between(values, np.maximum(avg_1_re_h_int - std_1_re_h_int, 0), avg_1_re_h_int + std_1_re_h_int, alpha=0.5, color='orange')
    plt.legend()
    plt.xlabel("Sparsity Factor")
    plt.ylabel("Relative Error")
    plt.title(f"2-Steps vs 1-Step Interpolation Error ({system}-H)")
    plt.savefig(f"{error_plots_dir}/h-int-re.png")
    plt.close()

    plt.figure()
    plt.plot(values, avg_2_re_h_ext, label="Avg. Two-Steps RE", color='blue')
    plt.scatter(values, avg_2_re_h_ext, color='blue')
    plt.fill_between(values, np.maximum(avg_2_re_h_ext - std_2_re_h_ext, 0), avg_2_re_h_ext + std_2_re_h_ext, alpha=0.5, color='blue')
    plt.plot(values, avg_1_re_h_ext, label="Avg. One-Step RE", color='orange')
    plt.scatter(values, avg_1_re_h_ext, color='orange')
    plt.fill_between(values, np.maximum(avg_1_re_h_ext - std_1_re_h_ext, 0), avg_1_re_h_ext + std_1_re_h_ext, alpha=0.5, color='orange')
    plt.legend()
    plt.xlabel("Sparsity Factor")
    plt.ylabel("Relative Error")
    plt.title(f"2-Steps vs 1-Step Forecasting Error ({system}-H)")
    plt.savefig(f"{error_plots_dir}/h-ext-re.png")
    plt.close()

    if ("poly" in experiments_dir) and (system != "np"):
        plt.figure()
        plt.plot(values, avg_2_coef_h, label="Avg. Two-Steps RE", color='blue')
        plt.scatter(values, avg_2_coef_h, color='blue')
        plt.fill_between(values, np.maximum(avg_2_coef_h - std_2_coef_h, 0), avg_2_coef_h + std_2_coef_h, alpha=0.5, color='blue')
        plt.plot(values, avg_1_coef_h, label="Avg. One-Step RE", color='orange')
        plt.scatter(values, avg_1_coef_h, color='orange')
        plt.fill_between(values, np.maximum(avg_1_coef_h - std_1_coef_h, 0), avg_1_coef_h + std_1_coef_h, alpha=0.5, color='orange')
        plt.legend()
        plt.xlabel("Sparsity Factor")
        plt.ylabel("Relative Error")
        plt.title(f"2-Steps vs 1-Step Polynomial Coeffcients Error for H ({system})")
        plt.savefig(f"{error_plots_dir}/h-coef-re.png")
        plt.close()

    phase_trajectory_dir = f"{system_dir}/phase-and-trajectory-plots"
    os.makedirs(phase_trajectory_dir, exist_ok=True)

    for sparsity_factor in values:
        value_dir = f"{phase_trajectory_dir}/{sparsity_factor}-sparsity"
        os.makedirs(value_dir, exist_ok=True)

        data_folder = f"{experiments_dir}/{system}/{sparsity_factor}-best-data"

        data = {fname.replace(".npy", ""): np.load(f"{data_folder}/{fname}") for fname in trajectory_files}
        X_true = data.get("X_true")
        X_pred_2 = data.get("best_X_2d")
        X_pred_1 = data.get("best_Z")
        M = data.get("best_M")[:, 1]
        M_H = data.get("best_M")[:, -1]

        _, ncols = X_true.shape
        is_2d = ncols > 4

        if is_2d:
            t, p1_true, p2_true, q1_true, q2_true, H_true = X_true.T
            _, p1_pred_2, p2_pred_2, q1_pred_2, q2_pred_2, H_pred_2 = X_pred_2.T
            _, p1_pred_1, p2_pred_1, q1_pred_1, q2_pred_1, H_pred_1 = X_pred_1.T

            plots_defs = [
                Plot(PhaseSpacePlot(p1_true, q1_true, M), f"Phase Space (Dimension 1) ({system} @ {sparsity_factor * 100}% Sparsity)", "$q_1$", "$p_1$", "phase-1"),
                Plot(PhaseSpacePlot(p2_true, q2_true, M), f"Phase Space (Dimension 2) ({system} @ {sparsity_factor * 100}% Sparsity)", "$q_2$", "$p_2$", "phase-2"),
                Plot(TrajectoryPlot(t, p1_true, p1_pred_1, p1_pred_2, M), f"({system} @ {sparsity_factor * 100} % Sparsity) $p_1$ Recovery and Extrapolation", "t", "$p_1$", "p1-trajectory"),
                Plot(TrajectoryPlot(t, p2_true, p2_pred_1, p2_pred_2, M), f"({system} @ {sparsity_factor * 100} % Sparsity) $p_2$ Recovery and Extrapolation", "t", "$p_2$", "p2-trajectory"),
                Plot(TrajectoryPlot(t, q1_true, q1_pred_1, q1_pred_2, M), f"({system} @ {sparsity_factor * 100} % Sparsity) $q_1$ Recovery and Extrapolation", "t", "$q_1$", "q1-trajectory"),
                Plot(TrajectoryPlot(t, q2_true, q2_pred_1, q2_pred_2, M), f"({system} @ {sparsity_factor * 100} % Sparsity) $q_2$ Recovery and Extrapolation", "t", "$q_2$", "q2-trajectory"),
                Plot(TrajectoryPlot(t, H_true - np.median(H_true), H_pred_1 - np.median(H_pred_1), H_pred_2 - np.median(H_pred_2), M_H), f"({system} @ {sparsity_factor * 100} % Sparsity) $H$ Recovery and Extrapolation", "t", "$H$", "H")
            ]
        else:
            t, p_true, q_true, H_true = X_true.T
            _, p_pred_2, q_pred_2, H_pred_2 = X_pred_2.T
            _, p_pred_1, q_pred_1, H_pred_1 = X_pred_1.T

            plots_defs = [
                Plot(PhaseSpacePlot(p_true, q_true, M), f"Phase Space ({system} @ {sparsity_factor * 100}% Sparsity)", "$q$", "$p$", "phase"),
                Plot(TrajectoryPlot(t, p_true, p_pred_1, p_pred_2, M), f"({system} @ {sparsity_factor * 100} % Sparsity) $p$ Recovery and Extrapolation", "t", "$p$", "p-trajectory"),
                Plot(TrajectoryPlot(t, q_true, q_pred_1, q_pred_2, M), f"({system} @ {sparsity_factor * 100} % Sparsity) $q$ Recovery and Extrapolation", "t", "$q$", "q-trajectory"),
                Plot(TrajectoryPlot(t, H_true - np.median(H_true), H_pred_1 - np.median(H_pred_1), H_pred_2 - np.median(H_pred_2), M_H), f"({system} @ {sparsity_factor * 100} % Sparsity) $H$ Recovery and Extrapolation", "t", "$H$", "H")
            ]

        for plot_def in plots_defs:
            plot(plot_def)
            plt.savefig(f"{value_dir}/{plot_def.fname}.png")
            plt.close()


    

   