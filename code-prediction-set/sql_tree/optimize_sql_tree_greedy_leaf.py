from collections import deque
import numpy as np
import heapq as hq
import sys
import os

PICARD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{PICARD_DIR}/code-prediction-set/sql_tree")

from generate_probability_tree_from_sexpr import ExprWithProb


def solve_optimization(tree, max_cost_threshold):
    # map node to parent, compute current cost
    # list of leaves
    # current cost of tree
    map_node_name_to_include = {}
    map_node_to_parent = {}
    curr_tree_cost = 0
    leaves_cost = []
    q = deque()
    q.append((tree, None))
    indx = 0
    while len(q) > 0:
        curr_node, curr_node_parent = q.popleft()
        if curr_node.prob != -1 and not (np.isnan(curr_node.prob)):
            assert curr_node.prob <= 0
            curr_tree_cost += -1 * max(curr_node.prob, -10)
            # print("curr_tree_cost", curr_tree_cost, "curr_node.prob", curr_node.prob)
        map_node_to_parent[curr_node] = curr_node_parent
        curr_node.colon_name = curr_node.name + "::" + str(indx)
        curr_node.deleted = False
        map_node_name_to_include[curr_node.colon_name] = True
        for c in curr_node.children:
            q.append((c, curr_node))
        if len(curr_node.children) == 0 and curr_node.prob != -1 and not (np.isnan(curr_node.prob)):
            leaves_cost.append((max(curr_node.prob, -10), indx, curr_node))
        indx += 1

    hq.heapify(leaves_cost)


    # greedily rm largest cost leave until threshold is met
    while curr_tree_cost > max_cost_threshold and len(leaves_cost) > 0:
        curr_node_prob, indx, curr_node = hq.heappop(leaves_cost)
        map_node_name_to_include[curr_node.colon_name] = False
        assert curr_node_prob <= 0 or np.isnan(curr_node_prob)
        if curr_node.prob != -1 and not np.isnan(curr_node_prob):
            curr_tree_cost += curr_node_prob
        curr_node.deleted = True
        if map_node_to_parent[curr_node] is not None and all(
            [
                x is not None and (x.deleted or np.isnan(x.prob) or x.prob == -1)
                for x in map_node_to_parent[curr_node].children
            ]
        ):
            parent = map_node_to_parent[curr_node]
            # if np.isnan(parent.prob) or parent.prob == -1:
            #     continue
            hq.heappush(leaves_cost, (0 if np.isnan(parent.prob) or parent.prob == -1 else max(parent.prob, -10), indx, parent))

    # if not satisfiable
    # if curr_tree_cost > max_cost_threshold:
    #     for key in map_node_name_to_include:
    #         map_node_name_to_include[key] = False
    #     curr_tree_cost = 0
    return map_node_name_to_include, curr_tree_cost


def create_tree_from_optimization_result(tree, max_cost_threshold: float):
    map_node_name_to_include, pruned_tree_error = solve_optimization(tree, max_cost_threshold)
    pruned_root = ExprWithProb("root", -1)
    entire_tree_with_deleted = ExprWithProb("root", -1)
    included_nodes = 0
    total_nodes = 0
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
    assert (
        np.abs(
            (
                sum([1 if map_node_name_to_include[key] else 0 for key in map_node_name_to_include])
                / len(map_node_name_to_include)
            )
            - included_nodes / total_nodes
        )
        < 0.01
    )
    return (
        pruned_root.children[0] if len(pruned_root.children) > 0 else None,
        entire_tree_with_deleted.children[0] if len(entire_tree_with_deleted.children) > 0 else None,
        map_node_name_to_include,
        pruned_tree_error,
        sum([1 if map_node_name_to_include[key] else 0 for key in map_node_name_to_include])
        / len(map_node_name_to_include),
    )
