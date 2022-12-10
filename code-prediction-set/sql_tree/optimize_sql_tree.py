from collections import deque
from z3 import *
import numpy as np
import ipdb
import sys
import os
from typing import List

PICARD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{PICARD_DIR}/code-prediction-set/sql_tree")

from generate_probability_tree_from_sexpr import ExprWithProb


def add_tree_constraints(o, tree, cost_id="", m=-1):
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
        curr_indicator = Bool(
            curr_node.name + "::" + str(node_number)
            if cost_id == ""
            else str(cost_id) + "::" + curr_node.name + "::" + str(node_number)
        )

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
        curr_indicator_variable = (
            map_node_to_indicator[curr_node] if curr_node in map_node_to_indicator else parent_indicator
        )
        for c in curr_node.children:
            child_indicator_variable = (
                map_node_to_indicator[c] if c in map_node_to_indicator else curr_indicator_variable
            )
            o.add(Implies(child_indicator_variable, curr_indicator_variable))
            q.append((c, curr_indicator_variable))

    assert len(ordered_probabilities) == len(indicator_variables)
    # make sure all float
    ordered_probabilities = [p if type(p) == float else p - 1e-7 for p in ordered_probabilities]

    # if m != -1:
    #     HOLE_FLOAT = 1-1e-5
    #     # iterate through nodes and add constraint for number of holes
    #     sum_holes = 0
    #     for node in map_node_to_indicator:
    #         for child_node in node.children:
    #             sum_holes += HOLE_FLOAT * Not(map_node_to_indicator[node]) * map_node_to_indicator[child_node] if node in map_node_to_indicator and child_node in map_node_to_indicator else 0
    #     o.add(sum_holes <= m)
    return ordered_probabilities, indicator_variables


def solve_optimization_lst(tree, m, max_cost_threshold: List, minimize_removal=False):
    """
    assumes max_cost_threshold sorted in descending order
    solves optimization problem defined in first part of  https://www.overleaf.com/project/6304f33ff542595b403d373e
    """
    o = Optimize()
    assert all([max_cost_threshold[i] >= max_cost_threshold[i + 1] for i in range(len(max_cost_threshold) - 1)])
    all_probabilities = []
    all_indicator_variables = []
    # add node removal constraints for each threshold
    for curr_max_cost_threshold in max_cost_threshold:
        probabilities, indicator_variables = add_tree_constraints(o, tree, cost_id=curr_max_cost_threshold, m=m)
        all_probabilities.append(probabilities)
        all_indicator_variables.append(indicator_variables)
        # add single tau level constraint
        o.add(
            sum([-1 * float(probabilities[i]) * indicator_variables[i] for i in range(len(indicator_variables))])
            <= curr_max_cost_threshold
        )
    # add between tau level constraints
    for i in range(len(max_cost_threshold) - 1):
        larger_threshold_indicator_variables = all_indicator_variables[i]
        smaller_threshold_indicator_variables = all_indicator_variables[i + 1]
        assert len(smaller_threshold_indicator_variables) == len(larger_threshold_indicator_variables)
        for j in range(len(larger_threshold_indicator_variables)):
            o.add(Implies(Not(larger_threshold_indicator_variables[j]), Not(smaller_threshold_indicator_variables[j])))
    # add optimization
    if not minimize_removal:
        o.maximize(sum([-1 * probabilities[i] * indicator_variables[i] for i in range(len(indicator_variables))]))
        o.maximize(
            sum(
                [
                    sum(
                        [
                            -1
                            * all_probabilities[threshold_ind][indicator_ind]
                            * all_indicator_variables[threshold_ind][indicator_ind]
                            for indicator_ind in range(len(all_indicator_variables[threshold_ind]))
                        ]
                    )
                    for threshold_ind in range(len(max_cost_threshold))
                ]
            )
        )
    else:
        print("min removal not supported yet")
        raise Exception("not supported yet")
    return o.check(), o.model(), len(all_indicator_variables[0])


def create_tree_from_optimization_result_lst(tree, m, max_cost_threshold: List, minimize_removal=False):
    check, model, var_per_tree = solve_optimization_lst(tree, m, max_cost_threshold, minimize_removal=False)
    pruned_tree_data = []
    tuples = []
    for i in range(model.__len__()):
        key = model.__getitem__(i)
        value = model.get_interp(key)
        tuples.append([str(key), str(value)])
    for i in range(0, len(model), var_per_tree):
        pruned_tree_data.append(create_tree(tree, check, tuples[i : i + var_per_tree]))
    return pruned_tree_data


def solve_optimization(tree, max_cost_threshold, minimize_removal=False):
    """
    solves optimization problem defined in second part of  https://www.overleaf.com/project/6304f33ff542595b403d373e
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
    if not minimize_removal:
        o.maximize(sum([-1 * probabilities[i] * indicator_variables[i] for i in range(len(indicator_variables))]))
    else:
        o.maximize(
            sum([-1 * probabilities[i] * indicator_variables[i] for i in range(len(indicator_variables))])
            + sum([1.01 * indicator_variables[i] for i in range(len(indicator_variables))])
        )
    return o.check(), o.model()


def create_tree(tree, check, model):
    if str(check) == "unsat":
        return None
    else:
        map_node_name_to_include = {}
        tuples = None
        print(model)
        if type(model) == list:
            tuples = model
        else:
            tuples = [t.split(" = ") for t in str(model)[1:-1].split(",\n ")]
        for tup in tuples:
            if tup[0].count("::") >= 2:
                tup[0] = tup[0][tup[0].index("::") + 2 :]
            if len(tup) == 2:
                map_node_name_to_include[tup[0]] = tup[1] == "True"
        print("Nodes Included", sum([map_node_name_to_include[key] for key in map_node_name_to_include]))

        included_nodes = 0
        total_nodes = 0
        error_of_tree = []
        pruned_root = ExprWithProb("root", -1)
        entire_tree_with_deleted = ExprWithProb("root", -1)
        node_number = 0
        q = deque()
        q.append({"pruned_tree_parent": pruned_root, "entire_tree_parent": entire_tree_with_deleted, "child": tree})
        while len(q) > 0:
            curr = q.popleft()
            curr_child = curr["child"]
            pruned_tree_curr_parent = curr["pruned_tree_parent"]
            entire_tree_curr_parent = curr["entire_tree_parent"]
            pruned_tree_curr_node = ExprWithProb(curr_child.name, curr_child.prob)
            entire_tree_curr_node = ExprWithProb(curr_child.name, curr_child.prob)

            total_nodes += 1
            name_of_curr = curr_child.name + "::" + str(node_number)
            pruned_tree_curr_node.colon_name = name_of_curr
            entire_tree_curr_node.colon_name = name_of_curr
            entire_tree_curr_parent.children.append(entire_tree_curr_node)
            # if adding to tree, create new node and add to children of parent
            # node in map if no prob associated with token
            if name_of_curr not in map_node_name_to_include or map_node_name_to_include[name_of_curr]:
                pruned_tree_curr_node.deleted = False
                entire_tree_curr_node.deleted = False
                pruned_tree_curr_parent.children.append(pruned_tree_curr_node)
                error_of_tree.append(curr_child.prob if curr_child.prob not in [-1, np.nan] else 0)
                included_nodes += 1
            else:
                pruned_tree_curr_node.deleted = True
                entire_tree_curr_node.deleted = True

            # need to explore all children even if not including in tree to maintain node number count
            for c in curr_child.children:
                q.append(
                    {
                        "pruned_tree_parent": pruned_tree_curr_node,
                        "entire_tree_parent": entire_tree_curr_node,
                        "child": c,
                    }
                )
            node_number += 1
        return (
            pruned_root.children[0] if len(pruned_root.children) > 0 else None,
            entire_tree_with_deleted.children[0] if len(entire_tree_with_deleted.children) > 0 else None,
            map_node_name_to_include,
            check,
            model,
            error_of_tree,
            included_nodes / total_nodes,
        )


def create_tree_from_optimization_result(tree, max_cost_threshold, minimize_removal=False):
    # minimize_removal = True
    check, model = solve_optimization(tree, max_cost_threshold, minimize_removal=minimize_removal)
    return create_tree(tree, check, model)
