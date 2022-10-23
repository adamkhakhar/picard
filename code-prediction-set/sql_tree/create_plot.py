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
    # computed = []
    # for d_name, label in [
    #     ("Optimize Tau", "PAC"),
    #     ("Optimize Tau and Node Removal", "PAC_MIN_RM"),
    #     ("Greedy Cost Leaf Removal"),
    #     ("evaluation_result_NO_TS_PAC.bin", "Optimize Tau w/o Temperature Scaling"),
    #     ("evaluation_result_PROP.bin", "Greedy Proportion of Leaf Removal")
    # ]:
    #     data = load_data(f"/home/akhakhar/code/picard/code-prediction-set/sql_tree/{d_name}")
    #     print(d_name, label, len(data), data[0])

    #     map_p_to_target_in_pruned_pred = {}
    #     map_p_to_frac_rm = {}
    #     for sample in data:
    #         if sample["p"] not in map_p_to_target_in_pruned_pred:
    #             map_p_to_target_in_pruned_pred[sample["p"]] = (0, 0)
    #             map_p_to_frac_rm[sample["p"]] = []
    #         prev_cnt_sum = map_p_to_target_in_pruned_pred[sample["p"]]
    #         # print("sample["target_in_pruned_pred"][0]", sample["target_in_pruned_pred"][0])
    #         map_p_to_target_in_pruned_pred[sample["p"]] = (
    #             prev_cnt_sum[0] + int(sample["target_in_pruned_pred"][0]),
    #             prev_cnt_sum[1] + 1,
    #         )
    #         map_p_to_frac_rm[sample["p"]].append(sample["frac_included_nodes"])
    #     for key in map_p_to_target_in_pruned_pred:
    #         val = map_p_to_target_in_pruned_pred[key]
    #         print(
    #             "p:",
    #             round(key, 3),
    #             "|error",
    #             -np.log(key),
    #             "|val:",
    #             val,
    #             "%:",
    #             round(val[0] / val[1] * 100, 3),
    #             " | % removed: ",
    #             round(100 - np.mean(map_p_to_frac_rm[key]) * 100, 1),
    #         )

    #     # compute e-> tau map
    #     e = [x / 100 for x in range(1, 50, 1)]
    #     n = 146
    #     d = 0.1
    #     k = [compute_k(n, e_i, d) for e_i in e]

    #     def find_tau_for_k(map_p_to_target_in_pruned_pred, k):
    #         if k is None:
    #             return None
    #         taus = [tau for tau in map_p_to_target_in_pruned_pred]
    #         taus.reverse()
    #         prev_tau = None
    #         for tau in taus:
    #             val = map_p_to_target_in_pruned_pred[tau]
    #             if val[1] - val[0] > k:
    #                 break
    #             prev_tau = tau
    #         return prev_tau

    #     taus = [find_tau_for_k(map_p_to_target_in_pruned_pred, k_i) for k_i in k]

    #     # only keep relevant data
    #     for i in range(len(taus)):
    #         if taus[i] != None:
    #             taus = taus[i:]
    #             e = e[i:]
    #             k = k[i:]
    #             break

    #     for i in range(len(e)):
    #         print(f"e: {e[i]} | k: {k[i]} | tau: {taus[i]}")

    #     percent_nodes_removed = [100 - np.mean(map_p_to_frac_rm[taus[i]]) * 100 for i in range(len(e))]
    #     code_coverage = [
    #         map_p_to_target_in_pruned_pred[taus[i]][0] / map_p_to_target_in_pruned_pred[taus[i]][1] * 100
    #         for i in range(len(e))
    #     ]
    #     computed.append(
    #         {
    #             "label": label,
    #             "e": e,
    #             "taus": taus,
    #             "percent_nodes_removed": percent_nodes_removed,
    #             "code_coverage": code_coverage,
    #         }
    #     )
    #     print("finished ", label)
    # # # e-> optimal tau
    # create_plot_from_fn(
    #     e,
    #     taus,
    #     save_title="e_optimal_tau",
    #     xlabel="e",
    #     ylabel="Tau",
    #     title="Optimal Tau",
    # )

    # # # e-> Percent nodes removed
    # create_plot_from_fn(
    #     e,
    #     percent_nodes_removed,
    #     save_title="e_node_removal",
    #     xlabel="e",
    #     ylabel="Percentage of Nodes Removed (%)",
    #     title="Percent Nodes Removed Over e Space",
    # )

    # # e-> Percent code coverage
    # create_plot_from_fn(
    #     e,
    #     code_coverage,
    #     save_title="e_coverage",
    #     xlabel="e",
    #     ylabel="Percentage of Target in Set (%)",
    #     title="Percent Code Coverage Over e Space",
    # )

    computed = []
    label_eval = [
        ("Greedy Cost Leaf Removal", "GREEDY_LEAF"),
        ("Optimize Tau", "PAC"),
        ("Optimize Tau and Node Removal", "PAC_MIN_RM"),
        ("Optimize Tau w/o Temperature Scaling", "PAC_NO_TS"),
        ("Greedy Proportion of Leaf Removal", "PROP"),
    ]
    for label, eval in label_eval:
        print(label, eval)
        e = [x / 100 for x in range(1, 50, 1)]
        n = 146
        d = 0.1
        k = [compute_k(n, e_i, d) for e_i in e]
        taus = []
        frac_rm = []
        prev_p = 1 - 1e-6
        e_used = []
        for i in range(len(k)):
            k_i = k[i]
            print("k_i", k_i, "/", k[-1], flush=True)
            if k_i == None:
                continue
            tau, total, frac_included = find_smallest_tau_with_less_than_k_errors(eval, k_i, max_p=prev_p)
            if tau != None:
                assert total == n
                taus.append(tau)
                frac_rm.append(100 * (1 - frac_included))
                prev_p = tau
                e_used.append(e[i])

        # only keep relevant data
        e_used = [e_used[i] for i in range(len(e_used)) if taus[i] is not None]
        frac_rm = [frac_rm[i] for i in range(len(frac_rm)) if taus[i] is not None]
        taus = [t for t in taus if t is not None]

        # for i in range(len(taus)):
        #     if taus[i] != None:
        #         taus = taus[i:]
        #         e_used = e_used[i:]
        #         k = k[i:]
        #         frac_rm = frac_rm[i:]
        #         break
        # for i in range(len(taus)):
        #     if taus[len(taus) - 1 - i] != None:
        #         taus = taus[:len(taus) - i]
        #         e_used = e_used[:len(taus) - i]
        #         k = k[:len(taus) - i]
        #         frac_rm = frac_rm[:len(taus) - i]
        #         break
        print(e_used)
        print(frac_rm)
        print(taus)
        assert len(taus) == len(e_used)
        assert len(taus) == len(frac_rm)

        # forward fill
        for i in range(len(taus)):
            idx = len(taus) - 1 - i
            if idx + 1 < len(taus):
                if taus[idx] is None:
                    taus[idx] = taus[idx + 1]
                    frac_rm[idx] = frac_rm[idx + 1]
        computed.append({"e": e, "taus": tau, "percent_nodes_removed": frac_rm, "label": label})

    store_data("computed_create_plot.pkl", computed)
    for key, y_label, title in [
        ("taus", "Tau", "Optimal Tau"),
        ("percent_nodes_removed", "Percentage of Nodes Removed (%)", "Percent Nodes Removed Over e Space"),
    ]:
        plot_multiple_series(
            [computed[i]["e"] for i in range(len(computed))],
            [computed[i][key] for i in range(len(computed))],
            [computed[i]["label"] for i in range(len(computed))],
            title,
            "e",
            y_label,
            key,
        )
