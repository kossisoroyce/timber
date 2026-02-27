"""Pass 2: Constant Feature Detection.

Identify features with zero variance (splits that always go the same way)
and fold the corresponding split nodes to their dominant branch.
"""

from __future__ import annotations

from typing import Any

from timber.ir.model import TimberIR, TreeEnsembleStage, TreeNode


def constant_feature_detection(
    ir: TimberIR,
) -> tuple[bool, TimberIR, dict[str, Any]]:
    """Detect constant features and fold trivial splits.

    A feature is considered constant if all split nodes using that feature
    have the same threshold direction (i.e., a model-level heuristic).
    More accurately, this pass identifies internal nodes where both children
    are leaves with identical values — the split is meaningless.

    Returns (changed, new_ir, details).
    """
    ensemble = ir.get_tree_ensemble()
    if ensemble is None:
        return False, ir, {"skipped": "no tree ensemble found"}

    total_folded = 0
    features_folded: set[int] = set()

    for tree in ensemble.trees:
        folded, feats = _fold_trivial_splits(tree.nodes)
        total_folded += folded
        features_folded.update(feats)
        if folded > 0:
            tree.recount()

    changed = total_folded > 0
    details: dict[str, Any] = {
        "nodes_folded": total_folded,
        "features_with_folds": sorted(features_folded),
    }
    return changed, ir, details


def _fold_trivial_splits(nodes: list[TreeNode]) -> tuple[int, set[int]]:
    """Fold internal nodes where both children are leaves with equal value.

    Returns (count_folded, set_of_feature_indices_folded).
    """
    if not nodes:
        return 0, set()

    folded = 0
    features: set[int] = set()
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

            # Fold if both children are leaves with the same value
            if left.is_leaf and right.is_leaf and left.leaf_value == right.leaf_value:
                features.add(node.feature_index)
                node.is_leaf = True
                node.leaf_value = left.leaf_value
                node.feature_index = -1
                node.threshold = 0.0
                node.left_child = -1
                node.right_child = -1
                folded += 1
                changed = True

    return folded, features
