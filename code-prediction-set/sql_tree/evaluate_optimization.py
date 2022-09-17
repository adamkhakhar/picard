import pickle
import os
import sys
import ipdb
import numpy as np

PICARD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{PICARD_DIR}/code-prediction-set/sql_tree")

from optimize_sql_tree import solve_optimization
from generate_probability_tree_from_sexpr import ExprWithProb


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def store_data(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    data = load_data(os.path.dirname(os.path.realpath(__file__)) + "tree_with_prob_and_target.bin")
    for sample in data:
        pred_tree = sample["pred_tree_with_prob"]
        target_tree = sample["target_tree"]
        max_cost_threshold = -np.log(0.8)
        check, model = solve_optimization(pred_tree, max_cost_threshold)
        print("pred_tree", str(pred_tree))
        print(check, model)
        ipdb.set_trace()
