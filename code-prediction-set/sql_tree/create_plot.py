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
    # FROM DATA METHOD
    d = 0.1
    data = load_data("/home/akhakhar/code/picard/code-prediction-set/sql_tree/evaluation_result_PAC.bin")

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

    taus_lst = []
    coverage_lst = []
    frac_rm_lst = []
    n = -1
    for key in map_p_to_target_in_pruned_pred:
        val = map_p_to_target_in_pruned_pred[key]
        taus_lst.append(key)
        coverage_lst.append(val[0] / val[1] * 100)
        frac_rm_lst.append(100 - np.mean(map_p_to_frac_rm[key]) * 100)
        n = val[1]

    assert len(taus_lst) == len(coverage_lst)
    assert len(taus_lst) == len(coverage_lst)
    assert len(taus_lst) == len(frac_rm_lst)

    # map data to e space
    e = np.linspace(0.02, 0.5, num=50).tolist()
    target_coverage = [100 - compute_k(n, e[i], d) / n * 100 for i in range(len(e))]

    e_to_taus_lst = []
    e_to_coverage_lst = []
    e_to_frac_rm_lst = []
    for i in range(len(e)):
        for j in range(len(taus_lst)):
            if coverage_lst[j] >= target_coverage[i]:
                e_to_taus_lst.append(taus_lst[j])
                e_to_coverage_lst.append(coverage_lst[j])
                e_to_frac_rm_lst.append(frac_rm_lst[j])
                break
    print("e", e)
    print("target_coverage", target_coverage)
    print("e_to_taus_lst", e_to_taus_lst)
    print("e_to_coverage_lst", e_to_coverage_lst)
    print("e_to_frac_rm_lst", e_to_frac_rm_lst)
    computed = []
    computed.append(
        {
            "e": e,
            "taus": e_to_taus_lst,
            "percent_nodes_removed": e_to_frac_rm_lst,
            "target_in_set": e_to_coverage_lst,
            "label": None,
        }
    )

    for key, y_label, title in [
        ("taus", r"$\tau$", r"Satisfying $\tau$ For Given $\epsilon$"),
        ("percent_nodes_removed", "Percentage of Nodes Removed (%)", r"Percent Nodes Removed Over $\epsilon$ Space"),
        ("target_in_set", "Percentage of Satisfying Code Sets (%)", r"Percent Code Set Coverage Over $\epsilon$ Space"),
    ]:
        print("Creating plots", key, y_label, title, flush=True)
        plot_multiple_series(
            [computed[i]["e"] for i in range(len(computed))],
            [computed[i][key] for i in range(len(computed))],
            [computed[i]["label"] for i in range(len(computed))],
            title,
            r"$\epsilon$",
            y_label,
            key,
        )

    # BINARY SEARCH METHOD:
    # computed = []
    # max_num_tries=1_000
    # label_eval = [
    #     ("Greedy Cost Leaf Removal", "GREEDY_LEAF", 1e-5),
    #     # # ("Optimize Tau", "PAC", 0),
    #     # # ("Optimize Tau and Node Removal", "PAC_MIN_RM", 0),
    #     # # ("Optimize Tau w/o Temperature Scaling", "PAC_NO_TS", 0.00001),
    #     ("Greedy Proportion of Leaf Removal", "PROP", 1e-5),
    #     ("Greedy Cost Leaf Removal w/o Temperature Scaling", "GREEDY_LEAF_NO_TS", 0),
    # ]
    # for label, eval, precision in label_eval:
    #     print(label, eval)
    #     e = [x / 100 for x in range(1, 50, 1)]
    #     n = 146
    #     d = 0.1
    #     k = [compute_k(n, e_i, d) for e_i in e]
    #     taus = []
    #     frac_rm = []
    #     prev_p = 1 - precision
    #     e_used = []
    #     set_coverage = []
    #     for i in range(len(k)):
    #         k_i = k[i]
    #         print("k_i", k_i, "/", k[-1], flush=True)
    #         if k_i == None:
    #             continue
    #         tau, total, frac_included, num_incorrect = find_smallest_tau_with_less_than_k_errors(
    #             eval, k_i, max_p=prev_p, precision=precision, max_num_tries=max_num_tries
    #         )
    #         if tau != None:
    #             assert total == n
    #             taus.append(tau)
    #             frac_rm.append(100 * (1 - frac_included))
    #             prev_p = tau
    #             e_used.append(e[i])
    #             set_coverage.append(100-100*(num_incorrect/total))

    #     # only keep relevant data
    #     e_used = [e_used[i] for i in range(len(e_used)) if taus[i] is not None]
    #     frac_rm = [frac_rm[i] for i in range(len(frac_rm)) if taus[i] is not None]
    #     set_coverage = [set_coverage[i] for i in range(len(set_coverage)) if taus[i] is not None]
    #     taus = [t for t in taus if t is not None]

    #     print(e_used)
    #     print(frac_rm)
    #     print(taus)
    #     print(set_coverage)
    #     assert len(taus) == len(e_used)
    #     assert len(taus) == len(frac_rm)

    #     computed.append({"e": e_used, "taus": taus, "percent_nodes_removed": frac_rm, "label": label, "target_in_set":set_coverage})

    # print("storing data", flush=True)
    # store_data("computed_data.pkl", computed)

    # computed[-1]["percent_nodes_removed"] = [x + 5 for x in computed[-1]["percent_nodes_removed"]]
    # # computed_orig = load_data("/home/akhakhar/code/picard/code-prediction-set/sql_tree/computed_create_plot.pkl")
    # # computed_orig[1]['label'] = "Integer Program, Optimize Tau"
    # # computed_orig[2]['label'] = "Integer Program, Optimize Tau and Nodes Removed"
    # # computed.append(computed_orig[1])
    # # computed.append(computed_orig[2])

    # for key, y_label, title in [
    #     ("taus", "Tau", "Optimal Tau"),
    #     ("percent_nodes_removed", "Percentage of Nodes Removed (%)", "Percent Nodes Removed Over e Space"),
    #     ("target_in_set", "Percentage of Satisfying Code Sets (%)", "Percent Code Set Coverage Over e Space")
    # ]:
    #     print("Creating plots", key, y_label, title, flush=True)
    #     plot_multiple_series(
    #         [computed[i]["e"] for i in range(len(computed))],
    #         [computed[i][key] for i in range(len(computed))],
    #         [computed[i]["label"] for i in range(len(computed))],
    #         title,
    #         "e",
    #         y_label,
    #         key,
    #     )
