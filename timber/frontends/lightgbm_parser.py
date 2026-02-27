"""LightGBM front-end parser — reads LightGBM model.txt files and converts to Timber IR."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from timber.ir.model import (
    Field,
    FieldType,
    Metadata,
    Objective,
    Schema,
    TimberIR,
    Tree,
    TreeEnsembleStage,
    TreeNode,
)


_OBJECTIVE_MAP: dict[str, Objective] = {
    "binary": Objective.BINARY_CLASSIFICATION,
    "binary binary:logistic": Objective.BINARY_CLASSIFICATION,
    "multiclass": Objective.MULTICLASS_CLASSIFICATION,
    "multiclass num_class": Objective.MULTICLASS_CLASSIFICATION,
    "regression": Objective.REGRESSION,
    "regression_l2": Objective.REGRESSION,
    "regression_l1": Objective.REGRESSION,
    "huber": Objective.REGRESSION,
    "fair": Objective.REGRESSION,
    "poisson": Objective.REGRESSION,
    "lambdarank": Objective.RANKING,
}


def parse_lightgbm_model(path: str | Path) -> TimberIR:
    """Parse a LightGBM model.txt file and return a TimberIR."""
    path = Path(path)
    raw = path.read_bytes()
    artifact_hash = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")

    return _parse_lightgbm_text(text, artifact_hash)


def _parse_lightgbm_text(text: str, artifact_hash: str = "") -> TimberIR:
    """Parse a LightGBM model text dump."""
    if not text or not text.strip():
        raise ValueError("Empty LightGBM model file")

    lines = text.strip().split("\n")

    # Parse header parameters
    params: dict[str, str] = {}
    tree_sections: list[list[str]] = []
    current_tree_lines: list[str] | None = None
    feature_names: list[str] = []
    feature_importances_dict: dict[str, int] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Tree="):
            if current_tree_lines is not None:
                tree_sections.append(current_tree_lines)
            current_tree_lines = [line]
        elif current_tree_lines is not None:
            if line == "" and i + 1 < len(lines) and lines[i + 1].strip().startswith("Tree="):
                tree_sections.append(current_tree_lines)
                current_tree_lines = None
            elif line.startswith("end of trees"):
                tree_sections.append(current_tree_lines)
                current_tree_lines = None
            else:
                current_tree_lines.append(line)
        elif "=" in line and not line.startswith("Tree="):
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key == "feature_names":
                feature_names = val.split(" ") if val else []
            elif key == "feature_importances":
                # Not standard but handle gracefully
                pass
            else:
                params[key] = val
        i += 1

    if current_tree_lines is not None:
        tree_sections.append(current_tree_lines)

    # Validate: must have either trees or recognizable LightGBM header params
    if not tree_sections and "max_feature_idx" not in params and "num_class" not in params:
        raise ValueError("Not a valid LightGBM model file — no trees or header parameters found")

    # Extract key parameters
    n_features = int(params.get("max_feature_idx", "0")) + 1
    n_classes = int(params.get("num_class", "1"))
    n_trees_per_iter = int(params.get("num_tree_per_iteration", "1"))

    # Objective
    obj_str = params.get("objective", "regression")
    objective = Objective.REGRESSION
    for key, obj in _OBJECTIVE_MAP.items():
        if obj_str.startswith(key):
            objective = obj
            break

    if n_classes <= 1:
        if objective == Objective.BINARY_CLASSIFICATION:
            n_classes = 2

    # Learning rate
    learning_rate = float(params.get("learning_rate", params.get("shrinkage_rate", "0.1")))

    # Parse trees
    trees: list[Tree] = []
    for idx, tree_lines in enumerate(tree_sections):
        tree = _parse_single_tree(tree_lines, tree_id=idx)
        trees.append(tree)

    # Build schema
    input_fields = []
    for idx in range(n_features):
        name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        input_fields.append(Field(
            name=name,
            dtype=FieldType.FLOAT32,
            index=idx,
        ))

    n_outputs = 1 if n_classes <= 2 else n_classes
    output_fields = [
        Field(name=f"output_{i}", dtype=FieldType.FLOAT32, index=i)
        for i in range(n_outputs)
    ]

    ensemble = TreeEnsembleStage(
        stage_name="lightgbm_ensemble",
        stage_type="tree_ensemble",
        trees=trees,
        n_features=n_features,
        n_classes=n_classes,
        objective=objective,
        base_score=0.0,
        learning_rate=learning_rate,
        is_boosted=True,
    )

    metadata = Metadata(
        source_framework="lightgbm",
        source_framework_version=params.get("version", ""),
        source_artifact_hash=artifact_hash,
        feature_names=feature_names,
        objective_name=obj_str,
        training_params={k: v for k, v in params.items()
                         if k not in ("feature_names",)},
    )

    return TimberIR(
        pipeline=[ensemble],
        schema=Schema(input_fields=input_fields, output_fields=output_fields),
        metadata=metadata,
    )


def _parse_single_tree(tree_lines: list[str], tree_id: int) -> Tree:
    """Parse a single LightGBM tree section into a flat node array.

    LightGBM text format uses parallel arrays like:
        split_feature=0 2 -1 -1 1
        threshold=0.5 1.2 0 0 3.4
        decision_type=2 2 0 0 2
        left_child=1 3 -1 -1 -1
        right_child=2 4 -1 -1 -1
        leaf_value=... (only for leaf nodes)
    """
    tree_params: dict[str, str] = {}
    for line in tree_lines:
        line = line.strip()
        if "=" in line:
            key, _, val = line.partition("=")
            tree_params[key.strip()] = val.strip()

    num_leaves = int(tree_params.get("num_leaves", "0"))
    num_internal = int(tree_params.get("num_cat", "0"))  # not always accurate

    # Parse parallel arrays
    split_features = _parse_int_array(tree_params.get("split_feature", ""))
    thresholds = _parse_float_array(tree_params.get("threshold", ""))
    left_children = _parse_int_array(tree_params.get("left_child", ""))
    right_children = _parse_int_array(tree_params.get("right_child", ""))
    leaf_values = _parse_float_array(tree_params.get("leaf_value", ""))
    decision_types = _parse_int_array(tree_params.get("decision_type", ""))

    # LightGBM uses a split between internal nodes and leaf nodes
    # Internal nodes are indexed 0..num_internal-1
    # Leaf nodes are referenced as ~leaf_index (negative: ~i means leaf i)
    n_internal = len(split_features)
    n_leaves = len(leaf_values)

    if n_internal == 0 and n_leaves == 0:
        tree = Tree(tree_id=tree_id)
        return tree

    # Build flat node array: internal nodes first, then leaf nodes
    total_nodes = n_internal + n_leaves
    nodes: list[TreeNode] = []

    # Internal nodes
    for i in range(n_internal):
        left = left_children[i] if i < len(left_children) else -1
        right = right_children[i] if i < len(right_children) else -1

        # Convert LightGBM's leaf references: negative means ~leaf_index
        left_mapped = _map_child(left, n_internal)
        right_mapped = _map_child(right, n_internal)

        node = TreeNode(
            node_id=i,
            feature_index=split_features[i] if i < len(split_features) else 0,
            threshold=thresholds[i] if i < len(thresholds) else 0.0,
            left_child=left_mapped,
            right_child=right_mapped,
            is_leaf=False,
            leaf_value=0.0,
            depth=0,
        )
        nodes.append(node)

    # Leaf nodes
    for i in range(n_leaves):
        node = TreeNode(
            node_id=n_internal + i,
            feature_index=-1,
            threshold=0.0,
            left_child=-1,
            right_child=-1,
            is_leaf=True,
            leaf_value=leaf_values[i] if i < len(leaf_values) else 0.0,
            depth=0,
        )
        nodes.append(node)

    # Compute depths
    if nodes:
        _compute_depths(nodes)

    tree = Tree(tree_id=tree_id, nodes=nodes)
    tree.recount()
    return tree


def _map_child(child_ref: int, n_internal: int) -> int:
    """Map LightGBM child reference to flat array index.

    In LightGBM:
    - Non-negative values are internal node indices
    - Negative values ~i (bitwise complement) reference leaf i
    """
    if child_ref >= 0:
        return child_ref
    # ~child_ref gives the leaf index
    leaf_idx = ~child_ref
    return n_internal + leaf_idx


def _compute_depths(nodes: list[TreeNode]) -> None:
    """BFS from root to compute node depths."""
    if not nodes:
        return
    nodes[0].depth = 0
    queue = [0]
    visited = {0}
    while queue:
        idx = queue.pop(0)
        node = nodes[idx]
        if not node.is_leaf:
            for child in (node.left_child, node.right_child):
                if 0 <= child < len(nodes) and child not in visited:
                    nodes[child].depth = node.depth + 1
                    visited.add(child)
                    queue.append(child)


def _parse_int_array(s: str) -> list[int]:
    if not s.strip():
        return []
    return [int(x) for x in s.strip().split()]


def _parse_float_array(s: str) -> list[float]:
    if not s.strip():
        return []
    return [float(x) for x in s.strip().split()]
