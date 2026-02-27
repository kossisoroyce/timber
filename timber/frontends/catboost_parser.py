"""CatBoost front-end parser — reads CatBoost JSON model dumps and converts to Timber IR.

CatBoost exports models via `model.save_model("model.json", format="json")`.
The JSON contains oblivious decision trees with symmetric structure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from timber.ir.model import (
    Field, FieldType, Metadata, Objective, Schema,
    TimberIR, Tree, TreeEnsembleStage, TreeNode,
)

_OBJECTIVE_MAP: dict[str, Objective] = {
    "Logloss": Objective.BINARY_CLASSIFICATION,
    "CrossEntropy": Objective.BINARY_CLASSIFICATION,
    "MultiClass": Objective.MULTICLASS_CLASSIFICATION,
    "RMSE": Objective.REGRESSION,
    "MAE": Objective.REGRESSION,
}


def parse_catboost_json(path: str | Path) -> TimberIR:
    path = Path(path)
    raw = path.read_bytes()
    artifact_hash = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    return _parse_catboost_dict(data, artifact_hash)


def _parse_catboost_dict(data: dict[str, Any], artifact_hash: str = "") -> TimberIR:
    if not isinstance(data, dict):
        raise ValueError("CatBoost model must be a JSON object")

    model_info = data.get("model_info", {})
    oblivious_trees = data.get("oblivious_trees", [])
    if not oblivious_trees:
        raise ValueError("No oblivious_trees found in CatBoost model JSON")

    params = model_info.get("params", {})
    loss_fn = params.get("loss_function", {})
    obj_name = loss_fn.get("type", "RMSE") if isinstance(loss_fn, dict) else str(loss_fn)
    objective = _OBJECTIVE_MAP.get(obj_name, Objective.REGRESSION)

    features_info = data.get("features_info", {})
    float_features = features_info.get("float_features", [])
    n_features = len(float_features) or int(model_info.get("num_features", 0))

    n_classes = 1
    if objective == Objective.BINARY_CLASSIFICATION:
        n_classes = 2
    elif objective == Objective.MULTICLASS_CLASSIFICATION:
        n_classes = int(model_info.get("class_count", 2))

    bias_data = data.get("scale_and_bias", [[], [0.0]])
    scale = float(bias_data[0][0]) if bias_data and bias_data[0] else 1.0
    bias = float(bias_data[1][0]) if len(bias_data) > 1 and bias_data[1] else 0.0

    trees: list[Tree] = []
    for i, otree in enumerate(oblivious_trees):
        trees.append(_parse_oblivious_tree(otree, tree_id=i, scale=scale))

    feature_names = [ff.get("feature_name", f"f{j}") for j, ff in enumerate(float_features)]

    ensemble = TreeEnsembleStage(
        stage_name="catboost_ensemble", stage_type="tree_ensemble",
        trees=trees, n_features=n_features, n_classes=max(n_classes, 1),
        objective=objective, base_score=bias,
        learning_rate=float(params.get("boosting_options", {}).get("learning_rate", 0.03)),
        is_boosted=True,
    )

    input_fields = [Field(name=feature_names[i] if i < len(feature_names) else f"f{i}",
                          dtype=FieldType.FLOAT32, index=i) for i in range(n_features)]
    n_outputs = 1 if n_classes <= 2 else n_classes
    output_fields = [Field(name=f"output_{i}", dtype=FieldType.FLOAT32, index=i) for i in range(n_outputs)]

    metadata = Metadata(
        source_framework="catboost", source_framework_version=[0, 0, 0],
        source_artifact_hash=artifact_hash, feature_names=feature_names,
        objective_name=obj_name, training_params={},
    )

    return TimberIR(
        pipeline=[ensemble],
        schema=Schema(input_fields=input_fields, output_fields=output_fields),
        metadata=metadata,
    )


def _level_of_node(idx: int) -> int:
    level = 0
    while (1 << (level + 1)) - 1 <= idx:
        level += 1
    return level


def _parse_oblivious_tree(otree: dict, tree_id: int, scale: float = 1.0) -> Tree:
    splits = otree.get("splits", [])
    leaf_values = otree.get("leaf_values", [])
    depth = len(splits)

    if depth == 0:
        val = leaf_values[0] * scale if leaf_values else 0.0
        return Tree(tree_id=tree_id, nodes=[
            TreeNode(node_id=0, feature_index=-1, threshold=0.0,
                     left_child=-1, right_child=-1, is_leaf=True,
                     leaf_value=val, depth=0, default_left=True)
        ])

    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    nodes: list[TreeNode] = []

    for i in range(n_internal):
        level = _level_of_node(i)
        split_idx = depth - 1 - level
        split = splits[split_idx] if split_idx < len(splits) else splits[-1]
        feat_idx = split.get("float_feature_index", split.get("feature_index", 0))
        threshold = float(split.get("border", split.get("threshold", 0.0)))
        nodes.append(TreeNode(
            node_id=i, feature_index=feat_idx, threshold=threshold,
            left_child=2 * i + 1, right_child=2 * i + 2,
            is_leaf=False, leaf_value=0.0, depth=level, default_left=True,
        ))

    for i in range(n_leaves):
        val = leaf_values[i] * scale if i < len(leaf_values) else 0.0
        nodes.append(TreeNode(
            node_id=n_internal + i, feature_index=-1, threshold=0.0,
            left_child=-1, right_child=-1, is_leaf=True,
            leaf_value=val, depth=depth, default_left=True,
        ))

    tree = Tree(tree_id=tree_id, nodes=nodes)
    tree.recount()
    return tree
