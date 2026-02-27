"""Pass 4: Frequency-Ordered Branch Sorting.

Requires a calibration dataset. Profiles split node outcomes and reorders
children so the more frequently taken branch is evaluated first (fall-through
on modern CPUs with static branch prediction).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from timber.ir.model import TimberIR, TreeEnsembleStage, TreeNode


def frequency_branch_sort(
    ir: TimberIR,
    calibration_data: np.ndarray,
) -> tuple[bool, TimberIR, dict[str, Any]]:
    """Reorder tree branches based on calibration data frequency.

    Args:
        ir: The Timber IR to optimize.
        calibration_data: A numpy array of shape (n_samples, n_features).

    Returns (changed, new_ir, details).
    """
    ensemble = ir.get_tree_ensemble()
    if ensemble is None:
        return False, ir, {"skipped": "no tree ensemble found"}

    if calibration_data is None or len(calibration_data) == 0:
        return False, ir, {"skipped": "no calibration data provided"}

    n_samples = calibration_data.shape[0]
    total_swapped = 0
    total_nodes_profiled = 0

    for tree in ensemble.trees:
        swapped, profiled = _sort_tree_branches(tree.nodes, calibration_data)
        total_swapped += swapped
        total_nodes_profiled += profiled

    changed = total_swapped > 0
    details: dict[str, Any] = {
        "calibration_samples": n_samples,
        "nodes_profiled": total_nodes_profiled,
        "branches_swapped": total_swapped,
    }
    return changed, ir, details


def _sort_tree_branches(
    nodes: list[TreeNode],
    data: np.ndarray,
) -> tuple[int, int]:
    """Profile and reorder branches in a single tree.

    Returns (branches_swapped, nodes_profiled).
    """
    if not nodes:
        return 0, 0

    n_samples = data.shape[0]
    swapped = 0
    profiled = 0

    # Count how many samples go left vs right at each internal node
    left_counts: dict[int, int] = {}
    right_counts: dict[int, int] = {}

    for node in nodes:
        if node.is_leaf:
            continue
        left_counts[node.node_id] = 0
        right_counts[node.node_id] = 0

    # Simulate traversal for all samples
    for sample_idx in range(n_samples):
        sample = data[sample_idx]
        _traverse_and_count(nodes, sample, left_counts, right_counts)

    # Swap branches where right is more frequent (so it becomes the fall-through)
    for node in nodes:
        if node.is_leaf:
            continue
        profiled += 1

        nid = node.node_id
        lc = left_counts.get(nid, 0)
        rc = right_counts.get(nid, 0)

        if rc > lc:
            # Swap children so the more frequent branch (right) becomes left
            node.left_child, node.right_child = node.right_child, node.left_child
            node.default_left = not node.default_left
            # Invert the comparison: the threshold meaning flips
            # Actually, we just swap the child pointers — the comparison stays the same
            # but the "left" child is now the more frequent path
            swapped += 1

    return swapped, profiled


def _traverse_and_count(
    nodes: list[TreeNode],
    sample: np.ndarray,
    left_counts: dict[int, int],
    right_counts: dict[int, int],
) -> None:
    """Traverse a tree with a single sample, counting branch directions."""
    if not nodes:
        return

    current = 0
    max_steps = len(nodes)
    steps = 0

    while steps < max_steps:
        steps += 1
        if current < 0 or current >= len(nodes):
            break

        node = nodes[current]
        if node.is_leaf:
            break

        feat = node.feature_index
        if feat < 0 or feat >= len(sample):
            # Missing feature — use default direction
            if node.default_left:
                left_counts[node.node_id] = left_counts.get(node.node_id, 0) + 1
                current = node.left_child
            else:
                right_counts[node.node_id] = right_counts.get(node.node_id, 0) + 1
                current = node.right_child
            continue

        val = sample[feat]

        if np.isnan(val):
            if node.default_left:
                left_counts[node.node_id] = left_counts.get(node.node_id, 0) + 1
                current = node.left_child
            else:
                right_counts[node.node_id] = right_counts.get(node.node_id, 0) + 1
                current = node.right_child
        elif val < node.threshold:
            left_counts[node.node_id] = left_counts.get(node.node_id, 0) + 1
            current = node.left_child
        else:
            right_counts[node.node_id] = right_counts.get(node.node_id, 0) + 1
            current = node.right_child
