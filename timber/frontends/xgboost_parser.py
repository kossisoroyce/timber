"""XGBoost front-end parser — reads XGBoost JSON model dumps and converts to Timber IR."""

from __future__ import annotations

import hashlib
import json
import math
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
    "binary:logistic": Objective.BINARY_CLASSIFICATION,
    "binary:logitraw": Objective.BINARY_CLASSIFICATION,
    "multi:softprob": Objective.MULTICLASS_CLASSIFICATION,
    "multi:softmax": Objective.MULTICLASS_CLASSIFICATION,
    "reg:squarederror": Objective.REGRESSION,
    "reg:linear": Objective.REGRESSION,
    "reg:logistic": Objective.REGRESSION_LOGISTIC,
    "rank:pairwise": Objective.RANKING,
    "rank:ndcg": Objective.RANKING,
}


def _coerce_float(val: Any) -> float:
    """Convert XGBoost numeric fields that may be stored as strings/lists into float."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        # XGBoost sometimes serializes as "[6.28E-1]"
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        try:
            return float(s)
        except Exception:
            pass
    if isinstance(val, (list, tuple)) and val:
        # Find first convertible element
        for el in val:
            try:
                return _coerce_float(el)
            except Exception:
                continue
    try:
        return float(str(val).strip("[] "))
    except Exception:
        raise ValueError(f"Cannot coerce value to float: {val!r}")


def parse_xgboost_json(path: str | Path) -> TimberIR:
    """Parse an XGBoost JSON model file and return a TimberIR."""
    path = Path(path)
    raw = path.read_bytes()
    artifact_hash = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)

    return _parse_xgboost_dict(data, artifact_hash)


def _parse_xgboost_dict(data: dict[str, Any], artifact_hash: str = "") -> TimberIR:
    """Parse the XGBoost JSON dict structure."""
    if not isinstance(data, dict):
        raise ValueError("XGBoost model must be a JSON object")

    learner = data.get("learner")
    if learner is None:
        # Check if this is a flat dict with gradient_booster at top level
        if "gradient_booster" not in data and "gbtree_model_param" not in data:
            raise ValueError("Missing 'learner' key — not a valid XGBoost JSON model")
        learner = data

    gradient_booster = learner.get("gradient_booster", {})
    if not gradient_booster:
        raise ValueError("Missing 'gradient_booster' in XGBoost model")
    model = gradient_booster.get("model", gradient_booster)

    # Extract learner model param
    learner_params = learner.get("learner_model_param", {})
    n_features = int(learner_params.get("num_feature", 0))
    n_classes = int(learner_params.get("num_class", 0))
    base_score_raw = _coerce_float(learner_params.get("base_score", 0.5))

    # Objective
    obj_info = learner.get("objective", {})
    obj_name = obj_info.get("name", "reg:squarederror")
    objective = _OBJECTIVE_MAP.get(obj_name, Objective.REGRESSION)

    if n_classes <= 1:
        n_classes = 1
        if objective in (Objective.BINARY_CLASSIFICATION,):
            n_classes = 2

    # XGBoost >= 2.0 stores base_score in probability space for logistic objectives.
    # We need to convert to margin (logit) space for the accumulator.
    if objective in (Objective.BINARY_CLASSIFICATION, Objective.REGRESSION_LOGISTIC):
        if 0.0 < base_score_raw < 1.0:
            base_score = math.log(base_score_raw / (1.0 - base_score_raw))
        else:
            base_score = base_score_raw
    else:
        base_score = base_score_raw

    # Parse trees
    gbtree_params = gradient_booster.get("gbtree_model_param",
                                          model.get("gbtree_model_param", {}))
    n_trees_total = int(gbtree_params.get("num_trees", 0))

    tree_info = model.get("tree_info", [])
    raw_trees = model.get("trees", [])

    trees: list[Tree] = []
    for i, raw_tree in enumerate(raw_trees):
        tree = _parse_single_tree(raw_tree, tree_id=i)
        trees.append(tree)

    # Feature names
    feature_names = learner.get("feature_names", [])
    feature_types = learner.get("feature_types", [])

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

    # Extract training params
    training_params: dict[str, Any] = {}
    if "learner_train_param" in learner:
        training_params.update(learner["learner_train_param"])

    # Determine learning rate
    learning_rate = _coerce_float(training_params.get("learning_rate",
                          learner_params.get("learning_rate", 0.3)))

    ensemble = TreeEnsembleStage(
        stage_name="xgboost_ensemble",
        stage_type="tree_ensemble",
        trees=trees,
        n_features=n_features,
        n_classes=n_classes,
        objective=objective,
        base_score=base_score,
        learning_rate=learning_rate,
        is_boosted=True,
    )

    metadata = Metadata(
        source_framework="xgboost",
        source_framework_version=data.get("version", [0, 0, 0]),
        source_artifact_hash=artifact_hash,
        feature_names=feature_names,
        objective_name=obj_name,
        training_params=training_params,
    )

    return TimberIR(
        pipeline=[ensemble],
        schema=Schema(input_fields=input_fields, output_fields=output_fields),
        metadata=metadata,
    )


def _parse_single_tree(raw: dict[str, Any], tree_id: int) -> Tree:
    """Parse a single XGBoost tree from the JSON representation into flat node array."""
    # XGBoost JSON stores trees with parallel arrays
    split_indices = raw.get("split_indices", [])
    split_conditions = raw.get("split_conditions", [])
    left_children = raw.get("left_children", [])
    right_children = raw.get("right_children", [])
    default_left_arr = raw.get("default_left", [])

    # Leaf values may be in split_conditions for leaf nodes
    # XGBoost marks leaves by left_children[i] == -1
    n_nodes = len(split_indices) if split_indices else 0

    # Handle the alternative "nodes" format
    if n_nodes == 0 and "nodes" in raw:
        return _parse_tree_nodes_format(raw["nodes"], tree_id)

    nodes: list[TreeNode] = []
    for i in range(n_nodes):
        lc_raw = left_children[i]
        rc_raw = right_children[i]
        lc = int(lc_raw) if isinstance(lc_raw, str) else lc_raw
        rc = int(rc_raw) if isinstance(rc_raw, str) else rc_raw
        is_leaf = lc == -1
        node = TreeNode(
            node_id=i,
            feature_index=split_indices[i] if not is_leaf else -1,
            threshold=_coerce_float(split_conditions[i]) if not is_leaf else 0.0,
            left_child=lc if not is_leaf else -1,
            right_child=rc if not is_leaf else -1,
            is_leaf=is_leaf,
            leaf_value=_coerce_float(split_conditions[i]) if is_leaf else 0.0,
            depth=0,  # computed below
            default_left=bool(int(default_left_arr[i])) if i < len(default_left_arr) else True,
        )
        nodes.append(node)

    # Compute depths via BFS from root
    if nodes:
        _compute_depths(nodes)

    tree = Tree(tree_id=tree_id, nodes=nodes)
    tree.recount()
    return tree


def _parse_tree_nodes_format(raw_nodes: list[dict[str, Any]], tree_id: int) -> Tree:
    """Parse tree from the older XGBoost 'nodes' list format."""
    nodes: list[TreeNode] = []
    for i, rn in enumerate(raw_nodes):
        is_leaf = rn.get("leaf", None) is not None
        node = TreeNode(
            node_id=i,
            feature_index=rn.get("split", -1) if not is_leaf else -1,
            threshold=_coerce_float(rn.get("split_condition", 0.0)) if not is_leaf else 0.0,
            left_child=rn.get("yes", -1) if not is_leaf else -1,
            right_child=rn.get("no", -1) if not is_leaf else -1,
            is_leaf=is_leaf,
            leaf_value=_coerce_float(rn.get("leaf", 0.0)) if is_leaf else 0.0,
            depth=rn.get("depth", 0),
            default_left=rn.get("missing", rn.get("yes", -1)) == rn.get("yes", -1),
        )
        nodes.append(node)

    if nodes and nodes[0].depth == 0:
        _compute_depths(nodes)

    tree = Tree(tree_id=tree_id, nodes=nodes)
    tree.recount()
    return tree


def _compute_depths(nodes: list[TreeNode]) -> None:
    """BFS from root to compute node depths."""
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
