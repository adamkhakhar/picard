import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from typing import List

CURR_DIR = "/home/akhakhar/code/picard/code-prediction-set/sql_tree/"
sys.path.append(CURR_DIR)
from generate_probability_tree_from_sexpr import ExprWithProb
from pac import compute_k
from evaluate_optimization import find_smallest_tau_with_less_than_k_errors


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
    save_path=f"/home/akhakhar/code/picard/code-prediction-set/results_v2/baseline_greedy_leaf",
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


def plot_multiple_series(
    x: List,
    y: List,
    series_label: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    save_title,
    save_path=f"/home/akhakhar/code/picard/code-prediction-set/results_v2/combined_plots",
):
    assert len(x) == len(y)
    assert len(x) == len(series_label)

    _, ax = plt.subplots()
    for i in range(len(x)):
        plt.plot(x[i], y[i], label=series_label[i])
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

    plt.legend()
    plt.tight_layout()
    plt.grid(linestyle="--", alpha=0.25)
    plt.savefig(f"{save_path}/{save_title}", bbox_inches="tight", transparent=False)


if __name__ == "__main__":
    computed = []
    label_eval = [
        # ("Greedy Cost Leaf Removal", "GREEDY_LEAF", 1e-5),
        # # ("Optimize Tau", "PAC", 0),
        # # ("Optimize Tau and Node Removal", "PAC_MIN_RM", 0),
        # # ("Optimize Tau w/o Temperature Scaling", "PAC_NO_TS", 0.00001),
        # ("Greedy Proportion of Leaf Removal", "PROP", 1e-5),
        ("Greedy Cost Leaf Removal w/o Temperature Scaling", "GREEDY_LEAF_NO_TS", 0),
    ]
    for label, eval, precision in label_eval:
        print(label, eval)
        e = [x / 100 for x in range(1, 50, 1)]
        n = 146
        d = 0.1
        k = [compute_k(n, e_i, d) for e_i in e]
        taus = []
        frac_rm = []
        prev_p = 1 - precision
        e_used = []
        for i in range(len(k)):
            k_i = k[i]
            print("k_i", k_i, "/", k[-1], flush=True)
            if k_i == None:
                continue
            tau, total, frac_included = find_smallest_tau_with_less_than_k_errors(
                eval, k_i, max_p=prev_p, precision=precision
            )
            if tau != None:
                assert total == n
                taus.append(tau)
                frac_rm.append(100 * (1 - frac_included))
                prev_p = tau
                e_used.append(e[i])
            exit()

        # only keep relevant data
        e_used = [e_used[i] for i in range(len(e_used)) if taus[i] is not None]
        frac_rm = [frac_rm[i] for i in range(len(frac_rm)) if taus[i] is not None]
        taus = [t for t in taus if t is not None]

        print(e_used)
        print(frac_rm)
        print(taus)
        assert len(taus) == len(e_used)
        assert len(taus) == len(frac_rm)

        computed.append({"e": e_used, "taus": taus, "percent_nodes_removed": frac_rm, "label": label})

    print("storing data", flush=True)
    store_data("computed_data.pkl", computed)

    # computed = load_data("/home/akhakhar/code/picard/code-prediction-set/sql_tree/computed_data.pkl")
    # computed = computed[:-1]
    # computed_orig = load_data("/home/akhakhar/code/picard/code-prediction-set/sql_tree/computed_create_plot.pkl")
    # computed_orig[1]['label'] = "Integer Program, Optimize Tau"
    # computed_orig[2]['label'] = "Integer Program, Optimize Tau and Nodes Removed"
    # computed.append(computed_orig[1])
    # computed.append(computed_orig[2])

    for key, y_label, title in [
        ("taus", "Tau", "Optimal Tau"),
        ("percent_nodes_removed", "Percentage of Nodes Removed (%)", "Percent Nodes Removed Over e Space"),
    ]:
        print("Creating plots", key, y_label, title, flush=True)
        plot_multiple_series(
            [computed[i]["e"] for i in range(len(computed))],
            [computed[i][key] for i in range(len(computed))],
            [computed[i]["label"] for i in range(len(computed))],
            title,
            "e",
            y_label,
            key,
        )
