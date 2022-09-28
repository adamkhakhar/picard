from collections import deque
from z3 import *
import numpy as np
import ipdb

PICARD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{PICARD_DIR}/code-prediction-set/sql_tree")

from generate_probability_tree_from_sexpr import ExprWithProb

DEFAULT_COST_OF_NO_PROB_NODE = -1e-14

def add_tree_constraints(o, tree):
    ordered_probabilities = []
    indicator_variables = []
    map_node_to_indicator = {}
    # traverse tree and create ordered names of nodes and probabilities corresponding to that node
    # create boolean variables for inclusion in tree
    node_number = 0
    q = deque()
    q.append(tree)
    while len(q) > 0:
        curr_node = q.popleft()
        curr_indicator = Bool(curr_node.name + "::" + str(node_number))

        # ignore nodes in tree without prob
        if curr_node.prob != -1 and not (np.isnan(curr_node.prob)):
            indicator_variables.append(curr_indicator)
            map_node_to_indicator[curr_node] = curr_indicator
            ordered_probabilities.append(max(curr_node.prob, -10))
        for c in curr_node.children:
            q.append(c)
        node_number += 1

    # add constraints (linear time)
    q = deque()
    q.append((tree, None))
    while len(q) > 0:
        curr_node, parent_indicator = q.popleft()
        curr_indicator_variable = map_node_to_indicator[curr_node] if curr_node in map_node_to_indicator else parent_indicator
        for c in curr_node.children:
            child_indicator_variable = map_node_to_indicator[c] if c in map_node_to_indicator else curr_indicator_variable
            o.add(Implies(child_indicator_variable, curr_indicator_variable))
            q.append((c, curr_indicator_variable))

    assert len(ordered_probabilities) == len(indicator_variables)
    # make sure all float
    ordered_probabilities = [p if type(p) == float else p - 1e-7 for p in ordered_probabilities]
    return ordered_probabilities, indicator_variables


def solve_optimization(tree, max_cost_threshold):
    """
    solves optimization problem defined in https://www.overleaf.com/project/6304f33ff542595b403d373e
    """
    o = Optimize()
    # add node removal constraints
    probabilities, indicator_variables = add_tree_constraints(o, tree)

    # print([(indicator_variables[i], ":", probabilities[i]) for i in range(len(probabilities))])
    # print("sum probs", sum(probabilities), "max possible cost", -1 * sum(probabilities))

    # add threshold constraint
    o.add(
        sum([-1 * float(probabilities[i]) * indicator_variables[i] for i in range(len(indicator_variables))])
        <= max_cost_threshold
    )
    # add optimization
    o.maximize(sum([-1 * probabilities[i] * indicator_variables[i] for i in range(len(indicator_variables))]))
    return o.check(), o.model()

def create_tree_from_optimization_result(tree, max_cost_threshold):
    check, model = solve_optimization(tree, max_cost_threshold)
    if str(check) == "unsat":
        return None
    else:
        map_node_name_to_include = {}
        # print(str(model))
        tuples = [t.split(" = ") for t in str(model)[1:-1].split(",\n ")]
        for tup in tuples:
            map_node_name_to_include[tup[0]] = tup[1] == "True"

        # print(map_node_name_to_include)
        
        error_of_tree = []
        root = ExprWithProb("root", -1)
        node_number = 0
        q = deque()
        q.append({
            "parent": root,
            "child": tree
        })
        while len(q) > 0:
            curr = q.popleft()
            curr_child = curr["child"]
            curr_parent = curr["parent"]
            curr_node = ExprWithProb(curr_child.name, curr_child.prob)
            # if adding to tree, create new node and add to children of parent
            name_of_curr = curr_child.name+"::"+str(node_number)
            # node in map if no prob associated with token
            if name_of_curr not in map_node_name_to_include or map_node_name_to_include[name_of_curr]:
                curr_parent.children.append(curr_node)
                error_of_tree.append(curr_child.prob if curr_child.prob != -1 else DEFAULT_COST_OF_NO_PROB_NODE)
            # need to explore all children even if not including in tree to maintain node number count
            for c in curr_child.children:
                q.append({
                    "parent": curr_node,
                    "child": c
                })
            node_number += 1
        return root.children[0] if len(root.children) > 0 else None, check, model, error_of_tree

