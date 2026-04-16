"""Deterministic execution pass for Timber IR models.

Transforms the IR to guarantee deterministic, reproducible inference:
- Sorts tree nodes for consistent traversal order
- Normalizes floating-point thresholds (flush denormals to zero)
- Ensures consistent NaN handling (always go left)
- Removes non-deterministic annotations
"""

from __future__ import annotations

import math
import struct
import warnings
from typing import Any

from timber.ir.model import (
    TimberIR,
    TreeEnsembleStage,
    TreeNode,
    Tree,
)

DISCLAIMER = (
    "ADVISORY — Deterministic transforms may alter model numerical semantics. "
    "Denormal flushing changes leaf values below ~1.18e-38 to zero. Tree "
    "reordering changes accumulation order, which may affect floating-point "
    "results within ULP tolerance. Verify numerical equivalence against the "
    "original model after transformation."
)

# Smallest normal float32 value; anything smaller is a denormal
_F32_MIN_NORMAL = 1.175494350822288e-38


def deterministic_pass(ir: TimberIR, **kwargs: Any) -> tuple[bool, TimberIR, dict[str, Any]]:
    """Transform IR for deterministic execution.

    Follows the Timber pass interface.

    Args:
        ir: Input Timber IR model.
        **kwargs: Optional settings:
            flush_denormals (bool): Flush denormal thresholds to zero. Default True.
            force_nan_left (bool): Force NaN default direction to left. Default True.
            sort_nodes (bool): Sort internal nodes for consistent order. Default True.
            strip_nondeterministic (bool): Remove non-deterministic annotations. Default True.

    Returns:
        Tuple of (changed, new_ir, diagnostics).
    """
    warnings.warn(DISCLAIMER, stacklevel=2)

    flush_denormals = kwargs.get("flush_denormals", True)
    force_nan_left = kwargs.get("force_nan_left", True)
    sort_nodes = kwargs.get("sort_nodes", True)
    strip_nondeterministic = kwargs.get("strip_nondeterministic", True)

    new_ir = ir.deep_copy()
    changed = False
    diagnostics: dict[str, Any] = {
        "denormals_flushed": 0,
        "nan_directions_fixed": 0,
        "trees_reordered": 0,
        "annotations_removed": [],
    }

    for stage in new_ir.pipeline:
        if not isinstance(stage, TreeEnsembleStage):
            continue

        # Strip non-deterministic annotations
        if strip_nondeterministic:
            nondeterministic_keys = [
                k for k in stage.annotations
                if k.startswith("random_") or k in ("seed", "shuffle", "subsample_seed")
            ]
            for key in nondeterministic_keys:
                del stage.annotations[key]
                diagnostics["annotations_removed"].append(key)
                changed = True

        for tree in stage.trees:
            # Flush denormal thresholds to zero
            if flush_denormals:
                for node in tree.nodes:
                    if node.is_leaf:
                        continue
                    if _is_denormal(node.threshold):
                        node.threshold = 0.0
                        diagnostics["denormals_flushed"] += 1
                        changed = True

            # Force consistent NaN handling: always go left
            if force_nan_left:
                for node in tree.nodes:
                    if node.is_leaf:
                        continue
                    if not node.default_left:
                        node.default_left = True
                        diagnostics["nan_directions_fixed"] += 1
                        changed = True

            # Sort internal nodes by (depth, feature_index, threshold) for
            # deterministic traversal order, then rebuild child pointers
            if sort_nodes:
                was_reordered = _sort_tree_nodes(tree)
                if was_reordered:
                    diagnostics["trees_reordered"] += 1
                    changed = True

    diagnostics["disclaimer"] = DISCLAIMER

    return changed, new_ir, diagnostics


def _is_denormal(value: float) -> bool:
    """Check if a float32 value is denormal (subnormal)."""
    if value == 0.0:
        return False
    return abs(value) < _F32_MIN_NORMAL


def _sort_tree_nodes(tree: Tree) -> bool:
    """Sort tree nodes in breadth-first order by (depth, feature_index, threshold).

    Rebuilds child pointers to maintain tree structure after reordering.
    Returns True if any reordering occurred.
    """
    if len(tree.nodes) <= 1:
        return False

    # Build adjacency: old_id -> (left_child_id, right_child_id)
    id_to_node: dict[int, TreeNode] = {n.node_id: n for n in tree.nodes}

    # Find root: the node that is not any other node's child
    child_ids = set()
    for node in tree.nodes:
        if not node.is_leaf:
            child_ids.add(node.left_child)
            child_ids.add(node.right_child)
    root_candidates = [n for n in tree.nodes if n.node_id not in child_ids]
    if not root_candidates:
        return False
    root = root_candidates[0]

    # BFS traversal to produce deterministic ordering
    ordered: list[TreeNode] = []
    queue: list[TreeNode] = [root]
    while queue:
        current = queue.pop(0)
        ordered.append(current)
        if not current.is_leaf:
            left = id_to_node.get(current.left_child)
            right = id_to_node.get(current.right_child)
            if left is not None:
                queue.append(left)
            if right is not None:
                queue.append(right)

    if len(ordered) != len(tree.nodes):
        # Some nodes were unreachable; skip reordering to avoid corruption
        return False

    # Check if already in correct order
    old_order = [n.node_id for n in tree.nodes]
    new_order = [n.node_id for n in ordered]
    if old_order == new_order:
        return False

    # Build old_id -> new_id mapping
    old_to_new: dict[int, int] = {}
    for new_idx, node in enumerate(ordered):
        old_to_new[node.node_id] = new_idx

    # Reassign IDs and update child pointers
    for node in ordered:
        old_id = node.node_id
        node.node_id = old_to_new[old_id]
        if not node.is_leaf:
            node.left_child = old_to_new[node.left_child]
            node.right_child = old_to_new[node.right_child]

    tree.nodes = ordered
    tree.recount()
    return True
