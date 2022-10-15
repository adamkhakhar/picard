import pickle
import os
import sys
import ipdb
import numpy as np
from collections import deque
from colorama import Fore, Back, Style
import traceback
import argparse

PICARD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{PICARD_DIR}/code-prediction-set/sql_tree")

from optimize_sql_tree import create_tree_from_optimization_result
from generate_probability_tree_from_sexpr import ExprWithProb


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e", dest="evaluation_type", type=str, default="PAC")

    args = parser.parse_args()
    assert args.e in ["PAC"]
    data = load_data(os.path.dirname(os.path.realpath(__file__)) + "tree_with_prob_and_target.bin")
    evaluation = []
    i = 0
    for sample in data:
        print(i)
        i += 1
        pred_tree = sample["pred_tree_with_prob"]
        if pred_tree.name == "ERROR":
            continue
        target_tree = sample["target_tree"]
        target_tree = make_all_lowercase_and_remove_spaces(target_tree)
        # print(pred_tree)
        # for p in np.arange(.99, 1.01, .001):
        for p in [x / 100 for x in range(1, 100, 1)]:
            # for p in [.9]:
            # print("p:", p)
            max_cost_threshold = -np.log(p)
            pruned_pred_tree = None
            target_in_pred = (None, None)
            try:
                # if args.
                (
                    pruned_pred_tree,
                    entire_tree_with_deleted,
                    map_node_name_to_include,
                    check,
                    model,
                    error_of_tree,
                    frac_included_nodes,
                ) = create_tree_from_optimization_result(pred_tree, max_cost_threshold)
            except Exception:
                traceback.print_exc()
                pass
            pruned_pred_tree = make_all_lowercase_and_remove_spaces(pruned_pred_tree)
            target_in_pred = evaluate_if_target_in_pruned_pred(pruned_pred_tree, target_tree)
            # print(-1 * sum(error_of_tree), "/", max_cost_threshold, target_in_pred[0])
            # if not target_in_pred[0]:
            if True:
                print("i", i)
                # print("orig pred tree:")
                # pretty_print_tree(pred_tree, include_prob=True)
                print("pruned_pred_tree: error", round(-1 * sum(error_of_tree), 3), "/", round(max_cost_threshold, 3))
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
                    "error_of_pruned_tree": -1 * sum(error_of_tree),
                    "frac_included_nodes": frac_included_nodes,
                }
            )

    print(len(data), "->", len(evaluation))
    store_data(os.path.dirname(os.path.realpath(__file__)) + "/evaluation_result.bin", evaluation)
