"""Constant-time tree traversal pass for Timber IR models.

Transforms decision trees into branchless, constant-time evaluation to
prevent timing side-channels. All paths through a tree execute the same
number of operations regardless of input values.

Key transformation: instead of conditional branching
    if (val < threshold) go_left else go_right
compute both children and use arithmetic select:
    result = left_val * (val < threshold) + right_val * (val >= threshold)

Trees are flattened into level-order arrays for indexed constant-time access.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Any

from timber.ir.model import (
    TimberIR,
    Tree,
    TreeEnsembleStage,
    TreeNode,
)

DISCLAIMER = (
    "ADVISORY ONLY — Constant-time transformation is a best-effort software "
    "technique. It does NOT provide hardware-verified timing guarantees. "
    "For side-channel critical applications, validate with hardware timing "
    "analysis (e.g., dudect, ctgrind) on the target platform."
)

# Annotation keys used to mark constant-time transformed trees
CT_ANNOTATION = "constant_time"
CT_LEVEL_ORDER = "ct_level_order_nodes"
CT_MAX_DEPTH = "ct_padded_depth"


def constant_time_pass(ir: TimberIR, **kwargs: Any) -> tuple[bool, TimberIR, dict[str, Any]]:
    """Transform trees for branchless constant-time traversal.

    Follows the Timber pass interface.

    Args:
        ir: Input Timber IR model.
        **kwargs: Optional settings:
            max_depth_limit (int): Maximum tree depth to pad to. Trees deeper
                than this are left unchanged. Default 20.
            pad_leaves (bool): Pad incomplete trees to perfect binary trees
                for uniform traversal. Default True.

    Returns:
        Tuple of (changed, new_ir, diagnostics).
    """
    warnings.warn(DISCLAIMER, stacklevel=2)

    max_depth_limit = kwargs.get("max_depth_limit", 20)
    pad_leaves = kwargs.get("pad_leaves", True)

    new_ir = ir.deep_copy()
    changed = False
    diagnostics: dict[str, Any] = {
        "trees_transformed": 0,
        "trees_skipped_depth": 0,
        "total_nodes_before": 0,
        "total_nodes_after": 0,
        "total_padding_nodes_added": 0,
    }

    for stage in new_ir.pipeline:
        if not isinstance(stage, TreeEnsembleStage):
            continue

        for i, tree in enumerate(stage.trees):
            diagnostics["total_nodes_before"] += len(tree.nodes)

            if tree.max_depth > max_depth_limit:
                warnings.warn(
                    f"Tree {tree.tree_id} has depth {tree.max_depth} > {max_depth_limit}, "
                    "skipping constant-time padding \u2014 timing side-channel risk"
                )
                diagnostics["trees_skipped_depth"] += 1
                diagnostics["total_nodes_after"] += len(tree.nodes)
                continue

            nodes_before = len(tree.nodes)

            # Step 1: Build level-order flat array (perfect binary tree)
            if pad_leaves:
                _pad_to_perfect_tree(tree)

            # Step 2: Flatten to level-order array for indexed access
            level_order = _flatten_level_order(tree)
            if level_order is None:
                diagnostics["total_nodes_after"] += len(tree.nodes)
                continue

            # Step 3: Replace tree nodes with level-order array
            tree.nodes = level_order
            tree.recount()

            nodes_after = len(tree.nodes)
            diagnostics["total_nodes_after"] += nodes_after
            diagnostics["total_padding_nodes_added"] += max(0, nodes_after - nodes_before)
            diagnostics["trees_transformed"] += 1
            changed = True

        # Mark the stage as constant-time transformed
        if diagnostics["trees_transformed"] > 0:
            stage.annotations[CT_ANNOTATION] = True
            stage.annotations[CT_MAX_DEPTH] = max(
                (t.max_depth for t in stage.trees), default=0
            )

    diagnostics["disclaimer"] = DISCLAIMER

    return changed, new_ir, diagnostics


def _pad_to_perfect_tree(tree: Tree) -> None:
    """Pad a tree to a perfect binary tree by adding dummy leaf nodes.

    Every internal node gets both children. Missing children are filled
    with leaf nodes whose value copies the nearest ancestor leaf, ensuring
    constant-time traversal always reaches depth == max_depth.
    """
    if not tree.nodes:
        return

    depth = tree.max_depth

    # Build lookup by node_id
    id_to_node: dict[int, TreeNode] = {n.node_id: n for n in tree.nodes}

    # Find root
    child_ids: set[int] = set()
    for node in tree.nodes:
        if not node.is_leaf:
            child_ids.add(node.left_child)
            child_ids.add(node.right_child)
    root_candidates = [n for n in tree.nodes if n.node_id not in child_ids]
    if not root_candidates:
        return
    root = root_candidates[0]

    next_id = max(n.node_id for n in tree.nodes) + 1

    # BFS: for every leaf above max_depth, extend with dummy internal nodes
    queue: deque[tuple[TreeNode, float]] = deque()
    queue.append((root, 0.0))

    while queue:
        node, default_leaf_val = queue.popleft()

        if node.is_leaf:
            leaf_val = node.leaf_value
            leaf_dist = node.leaf_distribution

            # If leaf is above max depth, convert to internal + extend
            if node.depth < depth:
                node.is_leaf = False
                node.feature_index = 0  # dummy feature
                node.threshold = 0.0    # always true or false, doesn't matter
                node.leaf_value = 0.0

                # Create two identical leaf children
                left = TreeNode(
                    node_id=next_id,
                    is_leaf=True,
                    leaf_value=leaf_val,
                    leaf_distribution=leaf_dist,
                    depth=node.depth + 1,
                    default_left=True,
                )
                next_id += 1
                right = TreeNode(
                    node_id=next_id,
                    is_leaf=True,
                    leaf_value=leaf_val,
                    leaf_distribution=leaf_dist,
                    depth=node.depth + 1,
                    default_left=True,
                )
                next_id += 1

                node.left_child = left.node_id
                node.right_child = right.node_id

                tree.nodes.append(left)
                tree.nodes.append(right)
                id_to_node[left.node_id] = left
                id_to_node[right.node_id] = right

                queue.append((left, leaf_val))
                queue.append((right, leaf_val))
        else:
            left_child = id_to_node.get(node.left_child)
            right_child = id_to_node.get(node.right_child)
            if left_child is not None:
                queue.append((left_child, default_leaf_val))
            if right_child is not None:
                queue.append((right_child, default_leaf_val))

    tree.recount()


def _flatten_level_order(tree: Tree) -> list[TreeNode] | None:
    """Flatten tree nodes into level-order (BFS) array.

    In a perfect binary tree of depth d, level-order indexing gives:
        - root at index 0
        - left child of node i at 2*i + 1
        - right child of node i at 2*i + 2

    This enables branchless constant-time indexed traversal:
        idx = 0
        for level in range(depth):
            go_right = (features[node[idx].feature] >= node[idx].threshold)
            idx = 2 * idx + 1 + go_right

    Returns None if the tree structure is invalid.
    """
    if not tree.nodes:
        return None

    # Find root
    id_to_node: dict[int, TreeNode] = {n.node_id: n for n in tree.nodes}
    child_ids: set[int] = set()
    for node in tree.nodes:
        if not node.is_leaf:
            child_ids.add(node.left_child)
            child_ids.add(node.right_child)
    root_candidates = [n for n in tree.nodes if n.node_id not in child_ids]
    if not root_candidates:
        return None
    root = root_candidates[0]

    # BFS to produce level-order array
    result: list[TreeNode] = []
    queue: deque[TreeNode] = deque([root])

    while queue:
        node = queue.popleft()
        new_idx = len(result)
        node.node_id = new_idx
        result.append(node)

        if not node.is_leaf:
            left = id_to_node.get(node.left_child)
            right = id_to_node.get(node.right_child)
            if left is not None:
                node.left_child = len(result) + len(queue)
                queue.append(left)
            if right is not None:
                node.right_child = len(result) + len(queue)
                queue.append(right)

    return result
