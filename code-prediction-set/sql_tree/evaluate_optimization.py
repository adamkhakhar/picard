import pickle
import os
import sys
import ipdb
import numpy as np
from collections import deque
from colorama import Fore, Back, Style
import traceback
import argparse
from mpmath import *

mp.dps = 20

PICARD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{PICARD_DIR}/code-prediction-set/sql_tree")

from optimize_sql_tree import create_tree_from_optimization_result, create_tree_from_optimization_result_lst
from generate_probability_tree_from_sexpr import ExprWithProb
import optimize_sql_tree_greedy_leaf
import optimize_sql_tree_proportion_leaf


class Expr:
    def __init__(self, name):
        self.name = name
        self.children = []

    def __str__(self):
        if len(self.children) == 0:
            return self.name
        else:
            return "(" + self.name + "".join([" " + str(child) for child in self.children]) + ")"


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def store_data(fname, data):
    print("STORING", fname)
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def make_all_lowercase_and_remove_spaces(target_tree):
    if target_tree is None:
        return target_tree
    q = deque()
    q.append(target_tree)
    while len(q) > 0:
        curr_node = q.popleft()
        curr_node.name = curr_node.name.lower().strip().replace(" ", "")
        for c in curr_node.children:
            q.append(c)
    return target_tree


def evaluate_if_target_in_pruned_pred(pruned_pred_tree, target_tree):
    if pruned_pred_tree is None:
        return (True, "Empty pruned tree")
    # print("evaluate_if_target_in_pruned_pred", pruned_pred_tree.name, target_tree.name)
    curr_node_same = pruned_pred_tree.name == target_tree.name
    # check children
    if curr_node_same:
        target_children_set = set([c.name for c in target_tree.children])
        for c in pruned_pred_tree.children:
            if c.name not in target_children_set:
                return False, f"pred child ({c.name}) not in target_children_set ({target_children_set})"
        # check that all children trees of pred are in target
        pred_child_names = [c.name for c in pruned_pred_tree.children]
        pred_child_name_map = {}
        for i in range(len(pred_child_names)):
            pred_child_name_map[pred_child_names[i]] = i
        for target_c in target_tree.children:
            if (target_c.name in pred_child_name_map) and (
                not evaluate_if_target_in_pruned_pred(
                    pruned_pred_tree.children[pred_child_name_map[target_c.name]], target_c
                )[0]
            ):
                return (
                    False,
                    evaluate_if_target_in_pruned_pred(
                        pruned_pred_tree.children[pred_child_name_map[target_c.name]], target_c
                    )[1],
                )
        return True, "T"
    else:
        return False, f"curr node is not the same: {pruned_pred_tree.name} != {target_tree.name}"


def pretty_print_tree(t, pref="", include_prob=False, print_deleted=True):
    if t == None:
        print(t)
        return
    curr_line = pref

    if type(t) == ExprWithProb and t.deleted and print_deleted:
        curr_line += Fore.BLACK + Back.RED + "[REMOVED] " + t.name + Style.RESET_ALL
    else:
        curr_line += Style.RESET_ALL + t.name + Style.RESET_ALL
    if include_prob:
        curr_line += Fore.WHITE + Back.BLACK + " | p=" + str(t.prob) + Style.RESET_ALL

    if type(t) == ExprWithProb and t.deleted and not print_deleted:
        pass
    else:
        print(Style.RESET_ALL + curr_line + Style.RESET_ALL)

    for c in t.children:
        pretty_print_tree(c, pref + "\t", include_prob=include_prob)


map_eval_to_data = {}


def run_algorithm_on_tau(eval: str, p: float):
    assert eval in ["PAC_MIN_RM", "PAC", "GREEDY_LEAF", "PROP", "PAC_NO_TS", "GREEDY_LEAF_NO_TS"]
    global map_eval_to_data
    data = None
    if eval not in map_eval_to_data:
        map_eval_to_data[eval] = load_data(
            os.path.dirname(os.path.realpath(__file__)) + "tree_with_prob_and_target.bin"
            if not eval.endswith("NO_TS")
            else "tree_with_prob_and_target_no_ts.bin"
        )
    data = map_eval_to_data[eval]
    total_tested = 0
    num_incorrect = 0
    fraction_included_nodes_lst = []
    for i, sample in enumerate(data):
        pred_tree = sample["pred_tree_with_prob"]
        if pred_tree.name == "ERROR":
            continue
        target_tree = sample["target_tree"]
        target_tree = make_all_lowercase_and_remove_spaces(target_tree)

        max_cost_threshold = -log(p)
        pruned_pred_tree = None
        target_in_pred = (None, None)
        try:
            if eval in ["PAC", "PAC_MIN_RM", "PAC_NO_TS"]:
                (
                    pruned_pred_tree,
                    entire_tree_with_deleted,
                    map_node_name_to_include,
                    check,
                    model,
                    error_of_tree,
                    frac_included_nodes,
                ) = create_tree_from_optimization_result(
                    pred_tree, max_cost_threshold, minimize_removal=eval == "PAC_MIN_RM"
                )
            elif eval in ["GREEDY_LEAF", "GREEDY_LEAF_NO_TS"]:
                (
                    pruned_pred_tree,
                    entire_tree_with_deleted,
                    map_node_name_to_include,
                    error_of_tree,
                    frac_included_nodes,
                ) = optimize_sql_tree_greedy_leaf.create_tree_from_optimization_result(pred_tree, max_cost_threshold)
            elif eval == "PROP":
                (
                    pruned_pred_tree,
                    entire_tree_with_deleted,
                    map_node_name_to_include,
                    error_of_tree,
                    frac_included_nodes,
                ) = optimize_sql_tree_proportion_leaf.create_tree_from_optimization_result(
                    pred_tree, max_cost_threshold
                )
            else:
                print("EVAL NOT SUPPORTED", flush=True)
        except Exception:
            print("EXCEPTION--------------------------")
            traceback.print_exc()
            continue
            # exit()
            # pass
        total_tested += 1
        pruned_pred_tree = make_all_lowercase_and_remove_spaces(pruned_pred_tree)
        target_in_pred = evaluate_if_target_in_pruned_pred(pruned_pred_tree, target_tree)
        if not target_in_pred[0]:
            num_incorrect += 1
        fraction_included_nodes_lst.append(frac_included_nodes)
    return num_incorrect, total_tested, np.mean(fraction_included_nodes_lst)


map_eval_to_output = {}


def find_smallest_tau_with_less_than_k_errors(eval, k, min_p=0, max_p=1, precision=0, max_num_tries=1_000):
    print("max_num_tries", max_num_tries, "precision", precision)
    assert precision >= 0
    max_p = min(max_p, 1 - precision)
    max_p = max(max_p, precision * 3)
    min_p = max(min_p, precision)
    max_p = mp.mpf(max_p)
    min_p = mp.mpf(min_p)
    precision = mp.mpf(precision)
    print("max_p", max_p, "min_p", min_p, flush=True)
    try:
        global map_eval_to_output
        if eval not in map_eval_to_output:
            map_eval_to_output[eval] = {}
        prev_sat = None, None, None, None
        num_tries = 0
        print("max_p > min_p", max_p > min_p)
        print("num_tries < max_num_tries", num_tries < max_num_tries)
        while max_p >= min_p and num_tries < max_num_tries:
            num_tries += 1
            mid_p = (max_p + min_p) / 2
            print(max_p, min_p, mid_p, num_tries, flush=True)
            num_incorrect_mid, total_tested_mid, frac_mid = None, None, None
            if mid_p in map_eval_to_output[eval]:
                num_incorrect_mid, total_tested_mid, frac_mid = map_eval_to_output[eval][mid_p]
            else:
                num_incorrect_mid, total_tested_mid, frac_mid = run_algorithm_on_tau(eval, mid_p)
                map_eval_to_output[eval][mid_p] = (num_incorrect_mid, total_tested_mid, frac_mid)
            print("\t", num_incorrect_mid)
            if num_incorrect_mid > k:
                min_p = mid_p + precision
            elif num_incorrect_mid < k:
                max_p = mid_p - precision
            else:
                prev_sat = mid_p, total_tested_mid, frac_mid, num_incorrect_mid
                num_incorrect_minus, total_tested_minus, frac_minus = run_algorithm_on_tau(eval, mid_p - precision)
                if num_incorrect_minus == k and (mid_p - precision - min_p > precision):
                    max_p = mid_p - precision
                else:
                    if precision != 0:
                        print(
                            "exit",
                            "num_incorrect_minus",
                            num_incorrect_minus,
                            "k",
                            k,
                            "mid_p - precision - min_p",
                            mid_p - precision - min_p,
                        )
                        return mid_p, total_tested_mid, frac_mid, num_incorrect_minus
            print("max_p > min_p", max_p > min_p)
            print("num_tries < max_num_tries", num_tries < max_num_tries)
        print("\t", prev_sat)
        return prev_sat
    except Exception:
        traceback.print_exc()
        print("EXCEPTION OUTER--------------------------", flush=True)
        return None, None, None, None


if __name__ == "__main__":
    # mid_p, total_tested_mid, frac_mid = find_smallest_tau_with_less_than_k_errors("GREEDY_LEAF", 0, precision=0.001)
    # print(mid_p, total_tested_mid, frac_mid)
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", dest="evaluation_type", type=str, default="PAC")
    parser.add_argument("--save_name", dest="save_name", type=str, default="")

    args = parser.parse_args()
    args.save_name = args.evaluation_type if len(args.save_name) == 0 else args.save_name
    assert args.evaluation_type in ["Z3COMB", "PAC_MIN_RM", "PAC", "GREEDY_LEAF", "PROP"]
    data = load_data(os.path.dirname(os.path.realpath(__file__)) + "tree_with_prob_and_target.bin")
    evaluation = []
    for i, sample in enumerate(data):
        print(i)
        pred_tree = sample["pred_tree_with_prob"]
        if pred_tree.name == "ERROR":
            print("\terror")
            continue
        target_tree = sample["target_tree"]
        target_tree = make_all_lowercase_and_remove_spaces(target_tree)
        # print(pred_tree)
        taus = [0.1, 0.9, 0.999]

        if "COMB" in args.evaluation_type:
            max_cost_threshold = [-np.log(p) for p in taus]
            print(max_cost_threshold)
            create_tree_from_optimization_result_lst(pred_tree, max_cost_threshold)
        else:
            for p in taus:
                max_cost_threshold = -np.log(p)
                pruned_pred_tree = None
                target_in_pred = (None, None)
                try:
                    if args.evaluation_type in ["PAC", "PAC_MIN_RM"]:
                        (
                            pruned_pred_tree,
                            entire_tree_with_deleted,
                            map_node_name_to_include,
                            check,
                            model,
                            error_of_tree,
                            frac_included_nodes,
                        ) = create_tree_from_optimization_result(
                            pred_tree, max_cost_threshold, minimize_removal=args.evaluation_type == "PAC_MIN_RM"
                        )
                    elif args.evaluation_type == "GREEDY_LEAF":
                        (
                            pruned_pred_tree,
                            entire_tree_with_deleted,
                            map_node_name_to_include,
                            error_of_tree,
                            frac_included_nodes,
                        ) = optimize_sql_tree_greedy_leaf.create_tree_from_optimization_result(
                            pred_tree, max_cost_threshold
                        )
                    elif args.evaluation_type == "PROP":
                        (
                            pruned_pred_tree,
                            entire_tree_with_deleted,
                            map_node_name_to_include,
                            error_of_tree,
                            frac_included_nodes,
                        ) = optimize_sql_tree_proportion_leaf.create_tree_from_optimization_result(
                            pred_tree, max_cost_threshold
                        )
                    else:
                        print("EVAL NOT SUPPORTED", flush=True)
                except Exception:
                    print("EXCEPTION--------------------------")
                    traceback.print_exc()
                    exit()
                    # pass
                pruned_pred_tree = make_all_lowercase_and_remove_spaces(pruned_pred_tree)
                target_in_pred = evaluate_if_target_in_pruned_pred(pruned_pred_tree, target_tree)
                # print(-1 * sum(error_of_tree), "/", max_cost_threshold, target_in_pred[0])
                # if not target_in_pred[0]:
                if True:
                    print("i", i)
                    # print("orig pred tree:")
                    # pretty_print_tree(pred_tree, include_prob=True)
                    if type(error_of_tree) == list:
                        print(
                            "pruned_pred_tree: error",
                            round(-1 * sum(error_of_tree), 3),
                            "/",
                            round(max_cost_threshold, 3),
                        )
                    else:
                        print("pruned_pred_tree: error", round(error_of_tree, 3), "/", round(max_cost_threshold, 3))
                    print("frac_included_nodes", round(frac_included_nodes, 3))
                    pretty_print_tree(entire_tree_with_deleted, include_prob=True, print_deleted=True)
                    print("target_tree:")
                    pretty_print_tree(target_tree)
                    print("outcome:", target_in_pred)
                    print("\n")

                evaluation.append(
                    {
                        "p": p,
                        "max_cost_threshold": max_cost_threshold,
                        "pred_tree": pred_tree,
                        "pruned_pred_tree": pruned_pred_tree,
                        "target_tree": target_tree,
                        "target_in_pruned_pred": target_in_pred,
                        "error_of_pruned_tree": -1 * sum(error_of_tree)
                        if type(error_of_tree) == list
                        else error_of_tree,
                        "frac_included_nodes": frac_included_nodes,
                    }
                )

    print(len(data), "->", len(evaluation))
    store_data(os.path.dirname(os.path.realpath(__file__)) + f"/evaluation_result_{args.save_name}.bin", evaluation)
