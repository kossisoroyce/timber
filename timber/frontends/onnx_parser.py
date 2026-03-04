"""ONNX ML opset front-end parser — reads ONNX models with ML opset operators.

Supports ONNX ML opset operators:
  - TreeEnsembleClassifier / TreeEnsembleRegressor
  - LinearClassifier / LinearRegressor
  - SVMClassifier / SVMRegressor
  - Normalizer (L1 / L2 / MAX normalization preprocessing)
  - Scaler (ZipMap-style offset+scale preprocessing)

Requires onnx package to be installed (not a hard Timber dependency).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from timber.ir.model import (
    Field,
    FieldType,
    LinearStage,
    Metadata,
    NormalizerStage,
    Objective,
    ScalerStage,
    Schema,
    SVMStage,
    TimberIR,
    Tree,
    TreeEnsembleStage,
    TreeNode,
)


_SUPPORTED_OPS = {
    "TreeEnsembleClassifier", "TreeEnsembleRegressor",
    "LinearClassifier", "LinearRegressor",
    "SVMClassifier", "SVMRegressor",
    "Normalizer", "Scaler",
}


def parse_onnx_model(path: str | Path) -> TimberIR:
    """Parse an ONNX model file containing ML opset operators."""
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
    """Dispatch to the correct converter based on the primary ML opset operator."""

    graph = model.graph

    # Collect preprocessing stages (Normalizer, Scaler) and the primary op
    pre_stages: list = []
    primary_node = None

    for node in graph.node:
        if node.op_type in ("Normalizer", "Scaler"):
            pre_stages.append(node)
        elif node.op_type in _SUPPORTED_OPS and primary_node is None:
            primary_node = node

    if primary_node is None:
        supported = ", ".join(sorted(_SUPPORTED_OPS))
        raise ValueError(
            f"No supported ONNX ML opset operator found. Supported: {supported}"
        )

    n_features = _infer_n_features(graph)

    # Build preprocessing pipeline stages
    pipeline_stages = []
    for pre in pre_stages:
        attrs = {a.name: a for a in pre.attribute}
        if pre.op_type == "Normalizer":
            norm_type = _get_string(attrs, "norm", "L2").lower()
            pipeline_stages.append(NormalizerStage(
                stage_name="normalizer",
                stage_type="normalizer",
                norm=norm_type,
            ))
        elif pre.op_type == "Scaler":
            offsets = list(_get_floats(attrs, "offset"))
            scales = list(_get_floats(attrs, "scale"))
            feat_indices = list(range(len(offsets) or len(scales)))
            pipeline_stages.append(ScalerStage(
                stage_name="scaler",
                stage_type="scaler",
                means=offsets,
                scales=scales,
                feature_indices=feat_indices,
            ))

    op = primary_node.op_type
    if op in ("TreeEnsembleClassifier", "TreeEnsembleRegressor"):
        main_stages, schema, metadata = _convert_tree_ensemble(
            primary_node, graph, n_features, artifact_hash
        )
    elif op in ("LinearClassifier", "LinearRegressor"):
        main_stages, schema, metadata = _convert_linear(
            primary_node, graph, n_features, artifact_hash
        )
    elif op in ("SVMClassifier", "SVMRegressor"):
        main_stages, schema, metadata = _convert_svm(
            primary_node, graph, n_features, artifact_hash
        )
    else:
        raise ValueError(f"Unhandled operator: {op}")

    return TimberIR(
        pipeline=pipeline_stages + main_stages,
        schema=schema,
        metadata=metadata,
    )


def _infer_n_features(graph: Any) -> int:
    """Infer the number of input features from the ONNX graph inputs."""
    for inp in graph.input:
        shape = inp.type.tensor_type.shape
        if shape and len(shape.dim) >= 2:
            dim_val = shape.dim[1].dim_value
            if dim_val > 0:
                return dim_val
    return 0


def _convert_tree_ensemble(
    tree_node: Any, graph: Any, n_features: int, artifact_hash: str
) -> tuple:
    """Convert TreeEnsembleClassifier / TreeEnsembleRegressor to IR."""

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

    return (
        [ensemble],
        Schema(input_fields=input_fields, output_fields=output_fields),
        metadata,
    )


def _convert_linear(
    node: Any, graph: Any, n_features: int, artifact_hash: str
) -> tuple:
    """Convert LinearClassifier / LinearRegressor to LinearStage IR."""
    is_classifier = node.op_type == "LinearClassifier"
    attrs = {a.name: a for a in node.attribute}

    coefficients = list(_get_floats(attrs, "coefficients"))
    intercepts = list(_get_floats(attrs, "intercepts"))
    post_transform = _get_string(attrs, "post_transform", "NONE")

    if is_classifier:
        # ONNX ML opset uses "classlabels_ints" for LinearClassifier
        class_labels = _get_ints(attrs, "classlabels_ints")
        if not class_labels:
            # Fallback: infer from biases length or weight rows
            n_bias = len(intercepts)
            n_classes = n_bias if n_bias > 1 else 2
        else:
            n_classes = len(class_labels)
        if n_classes <= 2:
            objective = Objective.BINARY_CLASSIFICATION
        else:
            objective = Objective.MULTICLASS_CLASSIFICATION
    else:
        n_classes = 1
        objective = Objective.REGRESSION

    # Infer n_features from coefficient length
    if n_features == 0 and coefficients:
        n_features = len(coefficients) // max(n_classes, 1)

    # Determine activation from post_transform
    pt = post_transform.upper()
    if pt in ("LOGISTIC", "SIGMOID"):
        activation = "sigmoid"
    elif pt in ("SOFTMAX", "SOFTMAX_ZERO"):
        # Binary SOFTMAX from ONNX is logistic — use sigmoid for scalar output
        activation = "softmax" if n_classes > 2 else "sigmoid"
    elif pt == "PROBIT":
        activation = "probit"
    else:
        activation = "none"

    # For binary classification (n_classes <= 2): always use a single weight row
    # and multi_weights=False to emit a scalar decision function.
    # ONNX may store 2 rows (one per class) — take the positive-class row.
    if n_classes <= 2 and is_classifier:
        if n_features > 0 and len(coefficients) >= n_features:
            weights_1d = coefficients[:n_features]
        else:
            weights_1d = coefficients
        bias_1d = intercepts[0] if intercepts else 0.0
        stage = LinearStage(
            stage_name="onnx_linear",
            stage_type="linear",
            weights=weights_1d,
            bias=bias_1d,
            activation=activation,
            n_classes=1,
            multi_weights=False,
            biases=[],
        )
    else:
        multi = n_classes > 2
        stage = LinearStage(
            stage_name="onnx_linear",
            stage_type="linear",
            weights=coefficients,
            bias=intercepts[0] if (intercepts and not multi) else 0.0,
            activation=activation,
            n_classes=n_classes,
            multi_weights=multi,
            biases=intercepts if multi else [],
        )

    input_fields = [
        Field(name=f"f{i}", dtype=FieldType.FLOAT32, index=i)
        for i in range(n_features)
    ]
    n_outputs = 1 if n_classes <= 2 else n_classes
    output_fields = [
        Field(name=f"output_{i}", dtype=FieldType.FLOAT32, index=i)
        for i in range(n_outputs)
    ]

    onnx_version = _onnx_version()
    metadata = Metadata(
        source_framework="onnx",
        source_framework_version=onnx_version,
        source_artifact_hash=artifact_hash,
        objective_name=node.op_type,
        training_params={"post_transform": post_transform},
    )
    return (
        [stage],
        Schema(input_fields=input_fields, output_fields=output_fields),
        metadata,
    )


def _convert_svm(
    node: Any, graph: Any, n_features: int, artifact_hash: str
) -> tuple:
    """Convert SVMClassifier / SVMRegressor to SVMStage IR."""
    is_classifier = node.op_type == "SVMClassifier"
    attrs = {a.name: a for a in node.attribute}

    kernel_type = _get_string(attrs, "kernel_type", "RBF").lower()
    support_vectors_flat = list(_get_floats(attrs, "support_vectors"))
    coefficients = list(_get_floats(attrs, "coefficients"))
    rho = list(_get_floats(attrs, "rho"))
    n_support = list(_get_ints(attrs, "vector_count")) or [len(support_vectors_flat) // max(n_features, 1)]
    gamma = float(attrs["gamma"].f) if "gamma" in attrs else 1.0
    coef0 = float(attrs["coef0"].f) if "coef0" in attrs else 0.0
    degree = int(attrs["degree"].i) if "degree" in attrs else 3
    post_transform = _get_string(attrs, "post_transform", "NONE")

    if is_classifier:
        class_labels = _get_ints(attrs, "classlabels_ints")
        n_classes = len(class_labels) if class_labels else 2
        objective = Objective.BINARY_CLASSIFICATION if n_classes <= 2 else Objective.MULTICLASS_CLASSIFICATION
    else:
        n_classes = 1
        objective = Objective.REGRESSION

    # Reshape flat support_vectors to list-of-lists
    n_sv = len(support_vectors_flat) // max(n_features, 1) if n_features > 0 else 0
    sv_matrix: list[list[float]] = []
    for i in range(n_sv):
        sv_matrix.append(support_vectors_flat[i * n_features:(i + 1) * n_features])

    pt = post_transform.upper()
    if pt in ("LOGISTIC", "SIGMOID"):
        pt_norm = "logistic"
    elif pt in ("SOFTMAX",):
        pt_norm = "softmax"
    else:
        pt_norm = "none"

    stage = SVMStage(
        stage_name="onnx_svm",
        stage_type="svm",
        kernel_type=kernel_type,
        support_vectors=sv_matrix,
        dual_coef=coefficients,
        rho=rho,
        n_support=n_support,
        gamma=gamma,
        coef0=coef0,
        degree=degree,
        n_features=n_features,
        n_classes=n_classes,
        objective=objective,
        post_transform=pt_norm,
    )

    input_fields = [
        Field(name=f"f{i}", dtype=FieldType.FLOAT32, index=i)
        for i in range(n_features)
    ]
    n_outputs = 1 if n_classes <= 2 else n_classes
    output_fields = [
        Field(name=f"output_{i}", dtype=FieldType.FLOAT32, index=i)
        for i in range(n_outputs)
    ]

    onnx_version = _onnx_version()
    metadata = Metadata(
        source_framework="onnx",
        source_framework_version=onnx_version,
        source_artifact_hash=artifact_hash,
        objective_name=node.op_type,
        training_params={"kernel_type": kernel_type, "post_transform": post_transform},
    )
    return (
        [stage],
        Schema(input_fields=input_fields, output_fields=output_fields),
        metadata,
    )


def _onnx_version() -> list:
    try:
        import onnx as _onnx
        return list(map(int, _onnx.__version__.split(".")[:3]))
    except Exception:
        return []


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
