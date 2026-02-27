"""ONNX ML opset front-end parser — reads ONNX models with tree ensemble operators.

Supports ONNX ML opset operators:
  - TreeEnsembleClassifier
  - TreeEnsembleRegressor

Requires onnx package to be installed (not a hard Timber dependency).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

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


def parse_onnx_model(path: str | Path) -> TimberIR:
    """Parse an ONNX model file containing tree ensemble operators."""
    try:
        import onnx
    except ImportError:
        raise ImportError("onnx package is required to parse ONNX models: pip install onnx")

    path = Path(path)
    raw = path.read_bytes()
    artifact_hash = hashlib.sha256(raw).hexdigest()

    model = onnx.load(str(path))
    return _convert_onnx(model, artifact_hash)


def _convert_onnx(model: Any, artifact_hash: str = "") -> TimberIR:
    """Convert an ONNX model with tree ensemble nodes to Timber IR."""
    import onnx

    graph = model.graph

    # Find tree ensemble operator
    tree_node = None
    for node in graph.node:
        if node.op_type in ("TreeEnsembleClassifier", "TreeEnsembleRegressor"):
            tree_node = node
            break

    if tree_node is None:
        raise ValueError("No TreeEnsembleClassifier or TreeEnsembleRegressor found in ONNX model")

    is_classifier = tree_node.op_type == "TreeEnsembleClassifier"

    # Extract attributes
    attrs = {a.name: a for a in tree_node.attribute}

    # Tree structure arrays
    nodes_featureids = _get_ints(attrs, "nodes_featureids")
    nodes_values = _get_floats(attrs, "nodes_values")
    nodes_hitrates = _get_floats(attrs, "nodes_hitrates") if "nodes_hitrates" in attrs else None
    nodes_modes = _get_strings(attrs, "nodes_modes")
    nodes_nodeids = _get_ints(attrs, "nodes_nodeids")
    nodes_treeids = _get_ints(attrs, "nodes_treeids")
    nodes_truenodeids = _get_ints(attrs, "nodes_truenodeids")
    nodes_falsenodeids = _get_ints(attrs, "nodes_falsenodeids")
    nodes_missing_value_tracks_true = _get_ints(attrs, "nodes_missing_value_tracks_true") if "nodes_missing_value_tracks_true" in attrs else None

    # Leaf / target info
    if is_classifier:
        class_ids = _get_ints(attrs, "class_ids")
        class_nodeids = _get_ints(attrs, "class_nodeids")
        class_treeids = _get_ints(attrs, "class_treeids")
        class_weights = _get_floats(attrs, "class_weights")
        post_transform = _get_string(attrs, "post_transform", "NONE")
        class_labels_int = _get_ints(attrs, "classlabels_int64s") if "classlabels_int64s" in attrs else []
        n_classes = len(class_labels_int) if class_labels_int else max(class_ids) + 1 if class_ids else 2
    else:
        target_ids = _get_ints(attrs, "target_ids")
        target_nodeids = _get_ints(attrs, "target_nodeids")
        target_treeids = _get_ints(attrs, "target_treeids")
        target_weights = _get_floats(attrs, "target_weights")
        post_transform = _get_string(attrs, "post_transform", "NONE")
        n_classes = 1

    base_values = _get_floats(attrs, "base_values") if "base_values" in attrs else [0.0]

    # Determine number of trees
    unique_treeids = sorted(set(nodes_treeids))
    n_trees = len(unique_treeids)

    # Determine n_features from input
    n_features = 0
    for inp in graph.input:
        shape = inp.type.tensor_type.shape
        if shape and len(shape.dim) >= 2:
            n_features = shape.dim[1].dim_value
            break
    if n_features == 0:
        n_features = max(nodes_featureids) + 1 if nodes_featureids else 0

    # Build trees
    trees: list[Tree] = []
    for tree_idx, tid in enumerate(unique_treeids):
        # Gather nodes for this tree
        tree_mask = [i for i, t in enumerate(nodes_treeids) if t == tid]
        if not tree_mask:
            continue

        # Map from ONNX node_id to local index
        onnx_nodeids = [nodes_nodeids[i] for i in tree_mask]
        id_to_local = {nid: local for local, nid in enumerate(onnx_nodeids)}

        # Build leaf value map
        leaf_values: dict[int, float] = {}
        if is_classifier:
            for i in range(len(class_treeids)):
                if class_treeids[i] == tid:
                    nid = class_nodeids[i]
                    leaf_values[nid] = class_weights[i]
        else:
            for i in range(len(target_treeids)):
                if target_treeids[i] == tid:
                    nid = target_nodeids[i]
                    leaf_values[nid] = target_weights[i]

        timber_nodes: list[TreeNode] = []
        for local_idx, global_idx in enumerate(tree_mask):
            nid = nodes_nodeids[global_idx]
            mode = nodes_modes[global_idx] if global_idx < len(nodes_modes) else "LEAF"
            is_leaf = mode == "LEAF"

            if is_leaf:
                lv = leaf_values.get(nid, 0.0)
                node = TreeNode(
                    node_id=local_idx,
                    feature_index=-1,
                    threshold=0.0,
                    left_child=-1,
                    right_child=-1,
                    is_leaf=True,
                    leaf_value=lv,
                    depth=0,
                    default_left=True,
                )
            else:
                true_nid = nodes_truenodeids[global_idx]
                false_nid = nodes_falsenodeids[global_idx]
                left = id_to_local.get(true_nid, -1)
                right = id_to_local.get(false_nid, -1)

                default_left = True
                if nodes_missing_value_tracks_true and global_idx < len(nodes_missing_value_tracks_true):
                    default_left = bool(nodes_missing_value_tracks_true[global_idx])

                node = TreeNode(
                    node_id=local_idx,
                    feature_index=nodes_featureids[global_idx],
                    threshold=nodes_values[global_idx],
                    left_child=left,
                    right_child=right,
                    is_leaf=False,
                    leaf_value=0.0,
                    depth=0,
                    default_left=default_left,
                )
            timber_nodes.append(node)

        # Compute depths
        _compute_depths(timber_nodes)
        tree = Tree(tree_id=tree_idx, nodes=timber_nodes)
        tree.recount()
        trees.append(tree)

    # Determine objective
    if is_classifier:
        if n_classes <= 2:
            objective = Objective.BINARY_CLASSIFICATION
        else:
            objective = Objective.MULTICLASS_CLASSIFICATION
    else:
        objective = Objective.REGRESSION

    base_score = float(base_values[0]) if base_values else 0.0

    ensemble = TreeEnsembleStage(
        stage_name="onnx_tree_ensemble",
        stage_type="tree_ensemble",
        trees=trees,
        n_features=n_features,
        n_classes=max(n_classes, 1),
        objective=objective,
        base_score=base_score,
        learning_rate=1.0,
        is_boosted=is_classifier or "Gradient" in str(attrs),
    )

    # Feature names from graph input
    feature_names: list[str] = []

    # Schema
    input_fields = [
        Field(name=f"f{i}", dtype=FieldType.FLOAT32, index=i)
        for i in range(n_features)
    ]
    n_outputs = 1 if n_classes <= 2 else n_classes
    output_fields = [
        Field(name=f"output_{i}", dtype=FieldType.FLOAT32, index=i)
        for i in range(n_outputs)
    ]

    onnx_version = []
    try:
        import onnx as _onnx
        onnx_version = list(map(int, _onnx.__version__.split(".")[:3]))
    except Exception:
        pass

    metadata = Metadata(
        source_framework="onnx",
        source_framework_version=onnx_version,
        source_artifact_hash=artifact_hash,
        feature_names=feature_names,
        objective_name=tree_node.op_type,
        training_params={"post_transform": post_transform},
    )

    return TimberIR(
        pipeline=[ensemble],
        schema=Schema(input_fields=input_fields, output_fields=output_fields),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Attribute extraction helpers
# ---------------------------------------------------------------------------

def _get_ints(attrs: dict, name: str) -> list[int]:
    if name not in attrs:
        return []
    return list(attrs[name].ints)


def _get_floats(attrs: dict, name: str) -> list[float]:
    if name not in attrs:
        return []
    return list(attrs[name].floats)


def _get_strings(attrs: dict, name: str) -> list[str]:
    if name not in attrs:
        return []
    return [s.decode("utf-8") if isinstance(s, bytes) else s for s in attrs[name].strings]


def _get_string(attrs: dict, name: str, default: str = "") -> str:
    if name not in attrs:
        return default
    val = attrs[name].s
    return val.decode("utf-8") if isinstance(val, bytes) else str(val)


def _compute_depths(nodes: list[TreeNode]) -> None:
    if not nodes:
        return
    nodes[0].depth = 0
    queue = [0]
    while queue:
        idx = queue.pop(0)
        node = nodes[idx]
        if not node.is_leaf:
            if 0 <= node.left_child < len(nodes):
                nodes[node.left_child].depth = node.depth + 1
                queue.append(node.left_child)
            if 0 <= node.right_child < len(nodes):
                nodes[node.right_child].depth = node.depth + 1
                queue.append(node.right_child)
