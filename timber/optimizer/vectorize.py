"""Vectorization pass (Pass 6) — analyzes tree ensembles for batched SIMD opportunities.

This pass annotates the IR with vectorization hints that the C99/WASM backends
can use to generate SIMD-friendly batched inference code. It groups trees by
structure similarity and marks feature access patterns for prefetch optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from timber.ir.model import TimberIR, Tree, TreeEnsembleStage, TreeNode


@dataclass
class VectorizationHint:
    """Vectorization metadata attached to a tree ensemble."""
    # Groups of structurally similar trees (same topology)
    structure_groups: list[list[int]] = field(default_factory=list)
    # Per-tree feature access order (for prefetch scheduling)
    feature_access_orders: dict[int, list[int]] = field(default_factory=dict)
    # Recommended batch tile size
    recommended_batch_tile: int = 4
    # Whether all trees have uniform depth
    uniform_depth: bool = False
    # Max feature index accessed (for memory layout optimization)
    max_feature_accessed: int = 0
    # Feature access frequency across all trees
    feature_frequencies: dict[int, int] = field(default_factory=dict)


def vectorization_analysis(ir: TimberIR, **kwargs) -> dict[str, Any]:
    """Analyze tree ensemble for vectorization opportunities.

    This pass does NOT modify the IR — it only annotates it with metadata.
    The backend emitters use these hints to generate optimized batch code.

    Returns:
        dict with pass stats and the VectorizationHint object.
    """
    ensemble = ir.get_tree_ensemble()
    if ensemble is None or ensemble.n_trees == 0:
        return {"trees_analyzed": 0, "groups_found": 0, "hint": VectorizationHint()}

    hint = VectorizationHint()

    # 1. Compute structural signatures for tree grouping
    signatures: dict[str, list[int]] = {}
    for tree in ensemble.trees:
        sig = _tree_signature(tree)
        signatures.setdefault(sig, []).append(tree.tree_id)

    hint.structure_groups = [ids for ids in signatures.values() if len(ids) > 1]

    # 2. Feature access order per tree (BFS traversal order)
    for tree in ensemble.trees:
        order = _feature_access_order(tree)
        hint.feature_access_orders[tree.tree_id] = order

    # 3. Feature frequency analysis
    freq: dict[int, int] = {}
    for tree in ensemble.trees:
        for node in tree.nodes:
            if not node.is_leaf and node.feature_index >= 0:
                freq[node.feature_index] = freq.get(node.feature_index, 0) + 1
    hint.feature_frequencies = freq
    hint.max_feature_accessed = max(freq.keys()) if freq else 0

    # 4. Uniform depth check
    depths = {tree.max_depth for tree in ensemble.trees}
    hint.uniform_depth = len(depths) <= 1

    # 5. Recommended batch tile based on tree count and depth
    avg_depth = sum(t.max_depth for t in ensemble.trees) / max(len(ensemble.trees), 1)
    if avg_depth <= 4:
        hint.recommended_batch_tile = 8  # shallow trees: more parallelism
    elif avg_depth <= 8:
        hint.recommended_batch_tile = 4
    else:
        hint.recommended_batch_tile = 2  # deep trees: less register pressure

    # 6. Store hint in IR metadata for backend consumption
    if ensemble.annotations is None:
        ensemble.annotations = {}
    ensemble.annotations["vectorization_hint"] = {
        "structure_groups": hint.structure_groups,
        "recommended_batch_tile": hint.recommended_batch_tile,
        "uniform_depth": hint.uniform_depth,
        "max_feature_accessed": hint.max_feature_accessed,
        "n_structure_groups": len(hint.structure_groups),
        "feature_hotspots": sorted(freq.items(), key=lambda x: -x[1])[:10],
    }

    return {
        "trees_analyzed": ensemble.n_trees,
        "groups_found": len(hint.structure_groups),
        "uniform_depth": hint.uniform_depth,
        "recommended_batch_tile": hint.recommended_batch_tile,
        "hint": hint,
    }


def _tree_signature(tree: Tree) -> str:
    """Compute a structural signature for a tree (topology only, ignoring values)."""
    parts = []
    for node in tree.nodes:
        if node.is_leaf:
            parts.append("L")
        else:
            parts.append(f"S{node.left_child},{node.right_child}")
    return "|".join(parts)


def _feature_access_order(tree: Tree) -> list[int]:
    """Return the feature access order via BFS traversal (for prefetch hints)."""
    if not tree.nodes:
        return []

    order = []
    queue = [0]
    while queue:
        idx = queue.pop(0)
        if idx < 0 or idx >= len(tree.nodes):
            continue
        node = tree.nodes[idx]
        if not node.is_leaf:
            if node.feature_index >= 0:
                order.append(node.feature_index)
            queue.append(node.left_child)
            queue.append(node.right_child)
    return order
