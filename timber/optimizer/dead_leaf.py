"""Pass 1: Dead Leaf Elimination.

Prune leaves whose contribution to the final prediction falls below a
configurable threshold relative to the largest leaf value in the ensemble.
"""

from __future__ import annotations

from typing import Any

from timber.ir.model import TimberIR, TreeEnsembleStage, TreeNode


def dead_leaf_elimination(
    ir: TimberIR,
    threshold: float = 0.001,
) -> tuple[bool, TimberIR, dict[str, Any]]:
    """Remove leaves with negligible contribution.

    Returns (changed, new_ir, details).
    """
    ensemble = ir.get_tree_ensemble()
    if ensemble is None:
        return False, ir, {"skipped": "no tree ensemble found"}

    # Find the global max absolute leaf value across all trees
    max_leaf = 0.0
    for tree in ensemble.trees:
        for node in tree.nodes:
            if node.is_leaf:
                max_leaf = max(max_leaf, abs(node.leaf_value))

    if max_leaf == 0.0:
        return False, ir, {"skipped": "all leaf values are zero"}

    cutoff = threshold * max_leaf
    total_pruned = 0
    trees_modified = 0

    for tree in ensemble.trees:
        pruned = _prune_tree(tree.nodes, cutoff)
        if pruned > 0:
            total_pruned += pruned
            trees_modified += 1
            tree.recount()

    changed = total_pruned > 0
    details: dict[str, Any] = {
        "threshold": threshold,
        "max_leaf_value": max_leaf,
        "cutoff": cutoff,
        "leaves_pruned": total_pruned,
        "trees_modified": trees_modified,
    }
    return changed, ir, details


def _prune_tree(nodes: list[TreeNode], cutoff: float) -> int:
    """Prune dead leaves in-place. Returns the count of pruned leaves.

    When both children of an internal node are leaves below the cutoff,
    collapse the internal node into a leaf with the value of the larger child.
    When only one child is below cutoff, replace the parent with the
    surviving child's subtree.
    """
    if not nodes:
        return 0

    pruned = 0
    changed = True

    while changed:
        changed = False
        for node in nodes:
            if node.is_leaf:
                continue

            left_idx = node.left_child
            right_idx = node.right_child

            if left_idx < 0 or right_idx < 0:
                continue
            if left_idx >= len(nodes) or right_idx >= len(nodes):
                continue

            left = nodes[left_idx]
            right = nodes[right_idx]

            if not left.is_leaf or not right.is_leaf:
                continue

            left_dead = abs(left.leaf_value) < cutoff
            right_dead = abs(right.leaf_value) < cutoff

            if left_dead and right_dead:
                # Both dead — collapse to the larger one
                survivor_val = left.leaf_value if abs(left.leaf_value) >= abs(right.leaf_value) else right.leaf_value
                node.is_leaf = True
                node.leaf_value = survivor_val
                node.feature_index = -1
                node.threshold = 0.0
                node.left_child = -1
                node.right_child = -1
                pruned += 2
                changed = True
            elif left_dead:
                # Left is negligible — collapse node to right value
                node.is_leaf = True
                node.leaf_value = right.leaf_value
                node.feature_index = -1
                node.threshold = 0.0
                node.left_child = -1
                node.right_child = -1
                pruned += 1
                changed = True
            elif right_dead:
                # Right is negligible — collapse node to left value
                node.is_leaf = True
                node.leaf_value = left.leaf_value
                node.feature_index = -1
                node.threshold = 0.0
                node.left_child = -1
                node.right_child = -1
                pruned += 1
                changed = True

    return pruned
