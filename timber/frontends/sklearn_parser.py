"""scikit-learn front-end parser — reads pickled sklearn Pipelines / estimators and converts to Timber IR.

Supports:
  - RandomForestClassifier / RandomForestRegressor
  - GradientBoostingClassifier / GradientBoostingRegressor
  - HistGradientBoostingClassifier / HistGradientBoostingRegressor
  - Standalone DecisionTreeClassifier / DecisionTreeRegressor
  - Pipeline objects containing a scaler + one of the above

Requires scikit-learn to be installed (not a hard Timber dependency).
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from timber.ir.model import (
    Field,
    FieldType,
    Metadata,
    Objective,
    PipelineStage,
    ScalerStage,
    Schema,
    TimberIR,
    Tree,
    TreeEnsembleStage,
    TreeNode,
)


def parse_sklearn_model(path: str | Path) -> TimberIR:
    """Parse a pickled sklearn model / pipeline and return a TimberIR."""
    path = Path(path)
    raw = path.read_bytes()
    artifact_hash = hashlib.sha256(raw).hexdigest()

    model = pickle.loads(raw)
    return _convert_sklearn(model, artifact_hash)


def _convert_sklearn(model: Any, artifact_hash: str = "") -> TimberIR:
    """Convert a sklearn estimator or pipeline to Timber IR."""
    try:
        from sklearn.pipeline import Pipeline as SkPipeline
    except ImportError:
        raise ImportError("scikit-learn is required to parse sklearn models: pip install scikit-learn")

    stages: list[PipelineStage] = []
    estimator = model
    feature_names: list[str] = []
    n_features: Optional[int] = None

    # Unwrap Pipeline
    if isinstance(model, SkPipeline):
        for name, step in model.steps[:-1]:
            scaler_stage = _try_parse_scaler(step, name)
            if scaler_stage is not None:
                stages.append(scaler_stage)
        estimator = model.steps[-1][1]

    # Parse the tree estimator
    ensemble_stage = _parse_estimator(estimator)
    stages.append(ensemble_stage)

    n_features = ensemble_stage.n_features
    if hasattr(estimator, "feature_names_in_"):
        feature_names = list(estimator.feature_names_in_)

    # Schema
    input_fields = [
        Field(name=feature_names[i] if i < len(feature_names) else f"f{i}",
              dtype=FieldType.FLOAT32, index=i)
        for i in range(n_features)
    ]
    n_outputs = 1 if ensemble_stage.n_classes <= 2 else ensemble_stage.n_classes
    output_fields = [
        Field(name=f"output_{i}", dtype=FieldType.FLOAT32, index=i)
        for i in range(n_outputs)
    ]

    metadata = Metadata(
        source_framework="sklearn",
        source_framework_version=[0, 0, 0],
        source_artifact_hash=artifact_hash,
        feature_names=feature_names,
        objective_name=ensemble_stage.objective.value,
        training_params={},
    )

    try:
        import sklearn
        metadata.source_framework_version = list(map(int, sklearn.__version__.split(".")[:3]))
    except Exception:
        pass

    return TimberIR(
        pipeline=stages,
        schema=Schema(input_fields=input_fields, output_fields=output_fields),
        metadata=metadata,
    )


def _try_parse_scaler(step: Any, name: str) -> Optional[ScalerStage]:
    """Try to convert a sklearn scaler step into a ScalerStage."""
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
    except ImportError:
        return None

    if isinstance(step, StandardScaler):
        mean = step.mean_.astype(np.float64).tolist() if step.with_mean else [0.0] * step.n_features_in_
        scale = step.scale_.astype(np.float64).tolist() if step.with_std else [1.0] * step.n_features_in_
        return ScalerStage(
            stage_name=name,
            stage_type="scaler",
            means=mean,
            scales=scale,
            feature_indices=list(range(step.n_features_in_)),
        )

    return None


def _parse_estimator(estimator: Any) -> TreeEnsembleStage:
    """Convert a sklearn tree-based estimator to TreeEnsembleStage."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    cls_name = type(estimator).__name__

    # --- GradientBoosting ---
    if cls_name in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        return _parse_gradient_boosting(estimator)

    # --- HistGradientBoosting ---
    if cls_name in ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
        return _parse_hist_gradient_boosting(estimator)

    # --- RandomForest ---
    if cls_name in ("RandomForestClassifier", "RandomForestRegressor"):
        return _parse_random_forest(estimator)

    # --- Single DecisionTree ---
    if isinstance(estimator, (DecisionTreeClassifier, DecisionTreeRegressor)):
        return _parse_single_decision_tree(estimator)

    raise ValueError(f"Unsupported sklearn estimator: {cls_name}")


# ---------------------------------------------------------------------------
# GradientBoosting
# ---------------------------------------------------------------------------

def _parse_gradient_boosting(est: Any) -> TreeEnsembleStage:
    is_classifier = hasattr(est, "classes_")
    n_classes = len(est.classes_) if is_classifier else 1

    if is_classifier and n_classes == 2:
        objective = Objective.BINARY_CLASSIFICATION
    elif is_classifier and n_classes > 2:
        objective = Objective.MULTICLASS_CLASSIFICATION
    else:
        objective = Objective.REGRESSION

    trees: list[Tree] = []
    estimators = est.estimators_
    tree_id = 0
    for stage_idx in range(len(estimators)):
        for cls_idx in range(estimators[stage_idx].shape[0] if hasattr(estimators[stage_idx], 'shape') else len(estimators[stage_idx])):
            dt = estimators[stage_idx][cls_idx]
            t = _sklearn_tree_to_timber(dt.tree_, tree_id)
            trees.append(t)
            tree_id += 1

    n_features = est.n_features_in_ if hasattr(est, 'n_features_in_') else est.n_features_
    base_score = 0.0
    if hasattr(est, 'init_'):
        init = est.init_
        if hasattr(init, 'constant_'):
            c = init.constant_
            if isinstance(c, np.ndarray):
                base_score = float(c.flat[0])
            else:
                base_score = float(c)

    return TreeEnsembleStage(
        stage_name="sklearn_gradient_boosting",
        stage_type="tree_ensemble",
        trees=trees,
        n_features=n_features,
        n_classes=max(n_classes, 1),
        objective=objective,
        base_score=base_score,
        learning_rate=float(est.learning_rate),
        is_boosted=True,
    )


# ---------------------------------------------------------------------------
# HistGradientBoosting
# ---------------------------------------------------------------------------

def _parse_hist_gradient_boosting(est: Any) -> TreeEnsembleStage:
    is_classifier = hasattr(est, "classes_")
    n_classes = len(est.classes_) if is_classifier else 1

    if is_classifier and n_classes == 2:
        objective = Objective.BINARY_CLASSIFICATION
    elif is_classifier and n_classes > 2:
        objective = Objective.MULTICLASS_CLASSIFICATION
    else:
        objective = Objective.REGRESSION

    trees: list[Tree] = []
    tree_id = 0

    # _predictors is a list of lists of TreePredictor objects
    for iteration_predictors in est._predictors:
        for predictor in iteration_predictors:
            nodes = predictor.nodes
            t = _hist_tree_to_timber(nodes, tree_id)
            trees.append(t)
            tree_id += 1

    n_features = est.n_features_in_ if hasattr(est, 'n_features_in_') else est.max_features_

    # Base prediction
    base_score = 0.0
    if hasattr(est, '_baseline_prediction'):
        bp = est._baseline_prediction
        if isinstance(bp, np.ndarray):
            base_score = float(bp.flat[0])
        else:
            base_score = float(bp)

    return TreeEnsembleStage(
        stage_name="sklearn_hist_gradient_boosting",
        stage_type="tree_ensemble",
        trees=trees,
        n_features=n_features,
        n_classes=max(n_classes, 1),
        objective=objective,
        base_score=base_score,
        learning_rate=float(est.learning_rate),
        is_boosted=True,
    )


# ---------------------------------------------------------------------------
# RandomForest
# ---------------------------------------------------------------------------

def _parse_random_forest(est: Any) -> TreeEnsembleStage:
    is_classifier = hasattr(est, "classes_")
    n_classes = len(est.classes_) if is_classifier else 1

    if is_classifier and n_classes == 2:
        objective = Objective.BINARY_CLASSIFICATION
    elif is_classifier and n_classes > 2:
        objective = Objective.MULTICLASS_CLASSIFICATION
    else:
        objective = Objective.REGRESSION

    trees: list[Tree] = []
    for i, dt in enumerate(est.estimators_):
        t = _sklearn_tree_to_timber(dt.tree_, tree_id=i)
        trees.append(t)

    n_features = est.n_features_in_ if hasattr(est, 'n_features_in_') else est.n_features_

    return TreeEnsembleStage(
        stage_name="sklearn_random_forest",
        stage_type="tree_ensemble",
        trees=trees,
        n_features=n_features,
        n_classes=max(n_classes, 1),
        objective=objective,
        base_score=0.0,
        learning_rate=1.0 / len(est.estimators_),  # averaging
        is_boosted=False,
    )


# ---------------------------------------------------------------------------
# Single DecisionTree
# ---------------------------------------------------------------------------

def _parse_single_decision_tree(est: Any) -> TreeEnsembleStage:
    from sklearn.tree import DecisionTreeClassifier
    is_classifier = isinstance(est, DecisionTreeClassifier)
    n_classes = len(est.classes_) if is_classifier else 1

    if is_classifier and n_classes == 2:
        objective = Objective.BINARY_CLASSIFICATION
    elif is_classifier and n_classes > 2:
        objective = Objective.MULTICLASS_CLASSIFICATION
    else:
        objective = Objective.REGRESSION

    t = _sklearn_tree_to_timber(est.tree_, tree_id=0)
    n_features = est.n_features_in_ if hasattr(est, 'n_features_in_') else est.n_features_

    return TreeEnsembleStage(
        stage_name="sklearn_decision_tree",
        stage_type="tree_ensemble",
        trees=[t],
        n_features=n_features,
        n_classes=max(n_classes, 1),
        objective=objective,
        base_score=0.0,
        learning_rate=1.0,
        is_boosted=False,
    )


# ---------------------------------------------------------------------------
# Tree conversion helpers
# ---------------------------------------------------------------------------

def _sklearn_tree_to_timber(sk_tree: Any, tree_id: int) -> Tree:
    """Convert a sklearn Tree (Cython struct) into a Timber Tree."""
    children_left = sk_tree.children_left
    children_right = sk_tree.children_right
    feature = sk_tree.feature
    threshold = sk_tree.threshold
    value = sk_tree.value  # shape: (n_nodes, n_outputs, max_n_classes)
    n_nodes = sk_tree.node_count

    TREE_LEAF = -1  # sklearn sentinel

    nodes: list[TreeNode] = []
    for i in range(n_nodes):
        is_leaf = children_left[i] == TREE_LEAF
        # Leaf value: for regression, value[i,0,0]; for classification, use weighted probability
        if is_leaf:
            v = value[i].flatten()
            total = v.sum()
            if total > 0:
                leaf_val = float(v[0] / total) if len(v) == 1 else float(v.max() / total)
                # For regression trees in boosting, use raw value
                if len(v) == 1:
                    leaf_val = float(v[0])
            else:
                leaf_val = 0.0
        else:
            leaf_val = 0.0

        node = TreeNode(
            node_id=i,
            feature_index=int(feature[i]) if not is_leaf else -1,
            threshold=float(threshold[i]) if not is_leaf else 0.0,
            left_child=int(children_left[i]) if not is_leaf else -1,
            right_child=int(children_right[i]) if not is_leaf else -1,
            is_leaf=is_leaf,
            leaf_value=leaf_val,
            depth=0,
            default_left=True,
        )
        nodes.append(node)

    # Compute depths
    if nodes:
        _compute_depths(nodes)

    tree = Tree(tree_id=tree_id, nodes=nodes)
    tree.recount()
    return tree


def _hist_tree_to_timber(nodes_arr: np.ndarray, tree_id: int) -> Tree:
    """Convert a HistGradientBoosting TreePredictor nodes array to Timber Tree."""
    nodes: list[TreeNode] = []

    for i in range(len(nodes_arr)):
        n = nodes_arr[i]
        # HistGBT node dtype has: value, count, feature_idx, num_threshold,
        #   missing_go_to_left, left, right, gain, depth, is_leaf, bin_threshold
        is_leaf = bool(n['is_leaf'])

        node = TreeNode(
            node_id=i,
            feature_index=int(n['feature_idx']) if not is_leaf else -1,
            threshold=float(n['num_threshold']) if not is_leaf else 0.0,
            left_child=int(n['left']) if not is_leaf else -1,
            right_child=int(n['right']) if not is_leaf else -1,
            is_leaf=is_leaf,
            leaf_value=float(n['value']),
            depth=int(n['depth']) if 'depth' in n.dtype.names else 0,
            default_left=bool(n['missing_go_to_left']) if 'missing_go_to_left' in n.dtype.names else True,
        )
        nodes.append(node)

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
