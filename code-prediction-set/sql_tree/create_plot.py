import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys

CURR_DIR = "/home/akhakhar/code/picard/code-prediction-set/sql_tree/"
sys.path.append(CURR_DIR)
from generate_probability_tree_from_sexpr import ExprWithProb
from pac import compute_k


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def store_data(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def create_plot_from_fn(
    x,
    y,
    save_path=f"/home/akhakhar/code/picard/code-prediction-set/results_v2/baseline_no_temp_scaling",
    save_title="fn",
    xlabel=None,
    ylabel=None,
    title=None,
):
    _, ax = plt.subplots()
    plt.plot(x, y)
    ax.spines["right"].set_color((0.8, 0.8, 0.8))
    ax.spines["top"].set_color((0.8, 0.8, 0.8))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_style("italic")
    xlab.set_size(10)
    ylab.set_style("italic")
    ylab.set_size(10)

    # tweak the title
    ttl = ax.title
    ttl.set_weight("bold")

    plt.tight_layout()
    plt.grid(linestyle="--", alpha=0.25)
    plt.savefig(f"{save_path}/{save_title}", bbox_inches="tight", transparent=False)


data = load_data("/home/akhakhar/code/picard/code-prediction-set/sql_tree/evaluation_result.bin")
# print(len(data), data[0])

map_p_to_target_in_pruned_pred = {}
map_p_to_frac_rm = {}
for sample in data:
    if sample["p"] not in map_p_to_target_in_pruned_pred:
        map_p_to_target_in_pruned_pred[sample["p"]] = (0, 0)
        map_p_to_frac_rm[sample["p"]] = []
    prev_cnt_sum = map_p_to_target_in_pruned_pred[sample["p"]]
    # print("sample["target_in_pruned_pred"][0]", sample["target_in_pruned_pred"][0])
    map_p_to_target_in_pruned_pred[sample["p"]] = (
        prev_cnt_sum[0] + int(sample["target_in_pruned_pred"][0]),
        prev_cnt_sum[1] + 1,
    )
    map_p_to_frac_rm[sample["p"]].append(sample["frac_included_nodes"])
for key in map_p_to_target_in_pruned_pred:
    val = map_p_to_target_in_pruned_pred[key]
    print(
        "p:",
        round(key, 3),
        "|error",
        -np.log(key),
        "|val:",
        val,
        "%:",
        round(val[0] / val[1] * 100, 3),
        " | % removed: ",
        round(100 - np.mean(map_p_to_frac_rm[key]) * 100, 1),
    )


# compute e-> tau map
e = [x / 100 for x in range(36, 40, 1)]
n = 146
d = 0.1
k = [compute_k(n, e_i, d) for e_i in e]


def find_tau_for_k(map_p_to_target_in_pruned_pred, k):
    taus = [tau for tau in map_p_to_target_in_pruned_pred]
    taus.reverse()
    prev_tau = None
    for tau in taus:
        val = map_p_to_target_in_pruned_pred[tau]
        if val[1] - val[0] > k:
            break
        prev_tau = tau
    return prev_tau


taus = [find_tau_for_k(map_p_to_target_in_pruned_pred, k_i) for k_i in k]

for i in range(len(e)):
    print(f"e: {e[i]} | k: {k[i]} | tau: {taus[i]}")


# # e-> optimal tau
create_plot_from_fn(
    e,
    taus,
    save_title="e_optimal_tau",
    xlabel="e",
    ylabel="Tau",
    title="Optimal Tau",
)

# # e-> Percent nodes removed
create_plot_from_fn(
    e,
    [100 - np.mean(map_p_to_frac_rm[taus[i]]) * 100 for i in range(len(e))],
    save_title="e_node_removal",
    xlabel="e",
    ylabel="Percentage of Nodes Removed (%)",
    title="Percent Nodes Removed Over e Space",
)

# e-> Percent code coverage
create_plot_from_fn(
    e,
    [
        map_p_to_target_in_pruned_pred[taus[i]][0] / map_p_to_target_in_pruned_pred[taus[i]][1] * 100
        for i in range(len(e))
    ],
    save_title="e_coverage",
    xlabel="e",
    ylabel="Percentage of Target in Set (%)",
    title="Percent Code Coverage Over e Space",
)
