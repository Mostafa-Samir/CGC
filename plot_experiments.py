import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


for sys in ["ms", "m2s3", "hh", "np"]:
    values = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    avg_2_re_p = []
    std_2_re_p = []
    avg_1_re_p = []
    std_1_re_p = []
    avg_2_re_q = []
    std_2_re_q = []
    avg_1_re_q = []
    std_1_re_q = []
    for val in values:
        path = f"200-gaussian/{sys}/sparse-{val}.csv"
        data = pd.read_csv(path, index_col=0)
        print(data)
        avg_2_re_p.append(data.loc["Mean", "Two-Steps P RE"])
        avg_1_re_p.append(data.loc["Mean", "One-Step P RE"])
        avg_2_re_q.append(data.loc["Mean", "Two-Steps Q RE"])
        avg_1_re_q.append(data.loc["Mean", "One-Step Q RE"])
        std_2_re_p.append(data.loc["STD", "Two-Steps P RE"])
        std_1_re_p.append(data.loc["STD", "One-Step P RE"])
        std_2_re_q.append(data.loc["STD", "Two-Steps Q RE"])
        std_1_re_q.append(data.loc["STD", "One-Step Q RE"])

    avg_2_re_p = np.array(avg_2_re_p)
    std_2_re_p = np.array(std_2_re_p)
    avg_1_re_p = np.array(avg_1_re_p)
    std_1_re_p = np.array(std_1_re_p)
    avg_2_re_q = np.array(avg_2_re_q)
    std_2_re_q = np.array(std_2_re_q)
    avg_1_re_q = np.array(avg_1_re_q)
    std_1_re_q = np.array(std_1_re_q)

    plt.figure()
    plt.plot(values, avg_2_re_p, label="Avg. Two-Steps RE", color='blue')
    plt.scatter(values, avg_2_re_p, color='blue')
    plt.fill_between(values, avg_2_re_p - std_2_re_p, avg_2_re_p + std_2_re_p, alpha=0.5, color='blue')
    plt.plot(values, avg_1_re_p, label="Avg. One-Step RE", color='orange')
    plt.scatter(values, avg_1_re_p, color='orange')
    plt.fill_between(values, avg_1_re_p - std_1_re_p, avg_1_re_p + std_2_re_p, alpha=0.5, color='orange')
    plt.legend()
    plt.xlabel("Sparsity Factor")
    plt.ylabel("Relative Error")
    plt.title(f"2-Steps vs 1-Step Error ({sys}-P)")
    plt.savefig(f"200-gaussian-{sys}-p.png")
    
    plt.figure()
    plt.plot(values, avg_2_re_q, label="Avg. Two-Steps RE", color='blue')
    plt.scatter(values, avg_2_re_q, color='blue')
    plt.fill_between(values, avg_2_re_q - std_2_re_q, avg_2_re_q + std_2_re_q, alpha=0.5, color='blue')
    plt.plot(values, avg_1_re_q, label="Avg. One-Step RE", color='orange')
    plt.scatter(values, avg_1_re_q, color='orange')
    plt.fill_between(values, avg_1_re_q - std_1_re_q, avg_1_re_q + std_1_re_q, alpha=0.5, color='orange')
    plt.legend()
    plt.xlabel("Sparsity Factor")
    plt.ylabel("Relative Error")
    plt.title(f"2-Steps vs 1-Step Error ({sys}-Q)")
    plt.savefig(f"200-gaussian-{sys}-q.png")
    

   