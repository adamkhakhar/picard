import importlib
import argparse
import pickle
import os
import sys
import ipdb
import numpy as np

PICARD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{PICARD_DIR}/code-prediction-set/sql_tree")

lisp = importlib.import_module("synth-sql.python.lisp")

from tokenize_spider import tokenize as custom_tokenize


class ExprWithProb:
    def __init__(self, name, prob, children=[]):
        self.name = name
        self.children = children
        self.prob = prob

    def __str__(self):
        if len(self.children) == 0:
            return self.name + ":p=" + str(round(self.prob, 3))
        else:
            return (
                "("
                + self.name
                + ":p="
                + str(round(self.prob, 3))
                + "".join([" " + str(child) for child in self.children])
                + ")"
            )


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def store_data(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def combine_token_prob_map(probs, tokens):
    combined_toks, combined_probs = custom_tokenize(tokens, probs, prob_combine_fun=lambda l: sum(np.log(l)))
    m = {}
    for i in range(len(combined_toks)):
        m[combined_toks[i]] = combined_probs[i]
        if combined_toks[i] == "*":
            m["all"] = combined_probs[i]
    return m


def get_prob_from_name(name, token_prob_map):
    most_likely_prob = -1
    n_char = -1
    tok_used = None
    for tok in token_prob_map:
        if "." in name:
            name = name.split(".")[-1]
        if tok in name:
            if len(tok) > n_char:
                n_char = len(tok)
                most_likely_prob = token_prob_map[tok]
    return most_likely_prob


def create_prob_tree(expr, token_prob_map):
    # print("create_prob_tree", expr.name)
    curr_node_prob = get_prob_from_name(expr.name, token_prob_map)
    return ExprWithProb(
        expr.name, curr_node_prob, children=[create_prob_tree(c, token_prob_map) for c in expr.children]
    )


def prob_tree_runner(expr, probs, tokens):
    map_combined_token_to_combined_prob = combine_token_prob_map(probs, tokens)
    # print("map_combined_token_to_combined_prob", map_combined_token_to_combined_prob)
    return create_prob_tree(expr, map_combined_token_to_combined_prob)


# Generates tree with probabilities in class ExprWithProb
if __name__ == "__main__":
    data = load_data(f"{PICARD_DIR}/code-prediction-set/sql_tree/create_sql_tree_result.bin")

    trees = []
    for sample in data:
        # ipdb.set_trace()
        expr = lisp.parse(str(sample["preds"][0]["pred_sexpr"]))
        probs = sample["preds"][0]["lst_probs"]
        tokens = sample["preds"][0]["lst_tokens"]
        # print("tokens", tokens)
        # print("probs", probs)
        tree_with_probs = prob_tree_runner(expr, probs, tokens)
        print("PREDICTION:")
        print(str(sample["preds"][0]["prediction"]))
        print("TREE WITH PROB:")
        print(str(tree_with_probs))

        # ipdb.set_trace()
        target_tree = lisp.parse(str(sample["preds"][0]["target_sexpr"]))
        print("target_tree\n", str(target_tree))
        print("\n")
        trees.append({"pred_tree_with_prob": tree_with_probs, "target_tree": target_tree})

    store_data(os.path.dirname(os.path.realpath(__file__)) + "tree_with_prob_and_target.bin", trees)
