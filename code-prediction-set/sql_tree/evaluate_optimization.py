import pickle
import os
import sys
import ipdb
import numpy as np
from collections import deque
import traceback

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
            return '(' + self.name + ''.join([' ' + str(child) for child in self.children]) + ')'


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def store_data(fname, data):
    print("STORING", fname)
    with open(fname, "wb") as f:
        pickle.dump(data, f)

def make_all_lowercase_and_remove_spaces(target_tree):
    q = deque()
    q.append(target_tree)
    while len(q) > 0:
        curr_node = q.popleft()
        curr_node.name = curr_node.name.lower().strip().replace(" ", "")
        for c in curr_node.children:
            q.append(c)
    return target_tree


def evaluate_if_target_in_pruned_pred(pruned_pred_tree, target_tree):
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
            if (target_c.name in pred_child_name_map) and (not evaluate_if_target_in_pruned_pred(pruned_pred_tree.children[pred_child_name_map[target_c.name]], target_c)[0]):
                return False, evaluate_if_target_in_pruned_pred(pruned_pred_tree.children[pred_child_name_map[target_c.name]], target_c)[1]
        return True, "T"
    else:
        return False, f"curr node is not the same: {pruned_pred_tree.name} != {target_tree.name}"


def pretty_print_tree(t, pref="", include_prob=False):
    if include_prob:
        print(pref + t.name + "|p="+ str(t.prob))
    else:
        print(pref + t.name)
    for c in t.children:
        pretty_print_tree(c, pref+"\t", include_prob=include_prob)

if __name__ == "__main__":
    data = load_data(os.path.dirname(os.path.realpath(__file__)) + "tree_with_prob_and_target.bin")
    evaluation = []
    i = 0
    for sample in data:
        # print(i)
        i += 1
        pred_tree = sample["pred_tree_with_prob"]
        if pred_tree.name == "ERROR":
            continue
        target_tree = sample["target_tree"]
        target_tree = make_all_lowercase_and_remove_spaces(target_tree)
        # print(pred_tree)
        # for p in np.arange(.99, 1.01, .001):
        for p in [.999999]:
            # print("p:", p)
            max_cost_threshold = -np.log(p)
            pruned_pred_tree = None
            target_in_pred = (None, None)
            try:
                pruned_pred_tree, check, model, error_of_tree = create_tree_from_optimization_result(pred_tree, max_cost_threshold)
            except Exception:
                pass
            if pruned_pred_tree is not None:
                pruned_pred_tree = make_all_lowercase_and_remove_spaces(pruned_pred_tree)
                target_in_pred = evaluate_if_target_in_pruned_pred(pruned_pred_tree, target_tree)
            else:
                target_in_pred = (True, "Empty pruned tree")
            print(-1 * sum(error_of_tree), "/", max_cost_threshold, target_in_pred[0])
            if not target_in_pred[0]:
                print("i", i)
                print("orig pred tree:")
                pretty_print_tree(pred_tree, include_prob=True)
                print("pruned_pred_tree: error", round(-1 * sum(error_of_tree), 3), "/", round(max_cost_threshold, 3))
                pretty_print_tree(pruned_pred_tree)
                print("target_tree:")
                pretty_print_tree(target_tree)
                print("outcome:", target_in_pred)
                print("\n")
            
            evaluation.append({
                "p": p,
                "max_cost_threshold": max_cost_threshold,
                "pred_tree": pred_tree,
                "pruned_pred_tree": pruned_pred_tree,
                "target_tree": target_tree,
                "target_in_pruned_pred": target_in_pred,
                "error_of_pruned_tree": -1 * sum(error_of_tree)
            })
            # ipdb.set_trace()
            
    print(len(data), "->", len(evaluation))
    store_data(os.path.dirname(os.path.realpath(__file__))+"/evaluation_result.bin", evaluation)