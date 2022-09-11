from z3 import *
import numpy as np


def add_tree_constraints(o, tree):
    ordered_probabilities = []
    indicator_variables = []
    map_node_to_indicator = {}
    # traverse tree and create ordered names of nodes and probabilities corresponding to that node
    # create boolean variables for inclusion in tree
    node_number = 0
    q = [tree]
    while len(q) > 0:
        curr_node = q.popleft()
        curr_indicator = Bool(curr_node.name + ":" + str(node_number))
        indicator_variables.append(curr_indicator)
        map_node_to_indicator[curr_node] = curr_indicator
        ordered_probabilities.append(curr_node.prob)
        for c in curr_node.children:
            q.append(c)
        node_number += 1

    # add constraints (linear time)
    q = [tree]
    while len(q) > 0:
        curr_node = q.popleft()
        curr_indicator_variable = map_node_to_indicator[curr_node]
        for c in curr_node.children:
            child_indicator_variable = map_node_to_indicator[c]
            o.add(Implies(child_indicator_variable, curr_indicator_variable))
            q.append(c)

    assert len(ordered_probabilities) == len(indicator_variables)
    return ordered_probabilities, indicator_variables


def solve_optimization(tree, max_cost_threshold):
    """
    solves optimization problem defined in https://www.overleaf.com/project/6304f33ff542595b403d373e
    """
    o = Optimize()
    # add node removal constraints
    probabilities, indicator_variables = add_tree_constraints(s, tree)
    # add threshold constraint
    o.add(
        sum([-1 * probabilities[i] * indicator_variables[i] for i in range(len(indicator_variables))])
        <= max_cost_threshold
    )
    # add optimization
    o.maximize(sum([-1 * probabilities[i] * indicator_variables[i] for i in range(len(indicator_variables))]))
    return o.check(), o.model()
