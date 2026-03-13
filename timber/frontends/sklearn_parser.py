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

import math

import numpy as np

from timber.ir.model import (
    Field,
    FieldType,
    GPRStage,
    IsolationForestStage,
    KNNStage,
    Metadata,
    NaiveBayesStage,
    Objective,
    PipelineStage,
    SVMStage,
    ScalerStage,
    Schema,
    TimberIR,
    Tree,
    TreeEnsembleStage,
    TreeNode,
    _c_factor,
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

    # Detect non-tree estimators first
    cls_name = type(estimator).__name__
    if cls_name == "IsolationForest":
        primary_stage = _parse_isolation_forest(estimator)
        n_features = primary_stage.n_features
        n_outputs = 1
        objective_name = "anomaly:isolation_forest"
    elif cls_name == "OneClassSVM":
        primary_stage = _parse_one_class_svm(estimator)
        n_features = primary_stage.n_features
        n_outputs = 1
        objective_name = "anomaly:one_class_svm"
    elif cls_name == "GaussianNB":
        primary_stage = _parse_naive_bayes(estimator)
        n_features = primary_stage.n_features
        n_outputs = primary_stage.n_classes
        objective_name = "multi:naive_bayes"
    elif cls_name == "GaussianProcessRegressor":
        primary_stage = _parse_gpr(estimator)
        n_features = primary_stage.n_features
        n_outputs = 1
        objective_name = "reg:gpr"
    elif cls_name in ("KNeighborsClassifier", "KNeighborsRegressor"):
        primary_stage = _parse_knn(estimator)
        n_features = primary_stage.n_features
        n_outputs = primary_stage.n_outputs
        objective_name = "classify:knn" if primary_stage.task_type == "classifier" else "reg:knn"
    else:
        primary_stage = _parse_estimator(estimator)
        n_features = primary_stage.n_features
        n_outputs = 1 if primary_stage.n_classes <= 2 else primary_stage.n_classes
        objective_name = primary_stage.objective.value

    stages.append(primary_stage)

    if hasattr(estimator, "feature_names_in_"):
        feature_names = list(estimator.feature_names_in_)

    # Schema
    input_fields = [
        Field(name=feature_names[i] if i < len(feature_names) else f"f{i}",
              dtype=FieldType.FLOAT32, index=i)
        for i in range(n_features)
    ]
    output_fields = [
        Field(name=f"output_{i}", dtype=FieldType.FLOAT32, index=i)
        for i in range(n_outputs)
    ]

    metadata = Metadata(
        source_framework="sklearn",
        source_framework_version=[0, 0, 0],
        source_artifact_hash=artifact_hash,
        feature_names=feature_names,
        objective_name=objective_name,
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


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def _parse_isolation_forest(est: Any) -> IsolationForestStage:
    """Convert sklearn IsolationForest to IsolationForestStage.

    Each isolation-tree leaf stores a pre-computed path-length contribution:
        leaf_value = leaf_depth + c(n_node_samples_at_leaf)
    so C99 inference needs no log() at runtime.
    """
    n_features = int(est.n_features_in_)
    max_samples = int(est.max_samples_)
    offset = float(est.offset_)

    trees: list[Tree] = []
    for tree_idx, (sk_est, feat_map) in enumerate(
        zip(est.estimators_, est.estimators_features_)
    ):
        t = _iforest_tree_to_timber(sk_est.tree_, tree_idx, feat_map)
        trees.append(t)

    return IsolationForestStage(
        stage_name="sklearn_isolation_forest",
        stage_type="isolation_forest",
        trees=trees,
        n_features=n_features,
        max_samples=max_samples,
        offset=offset,
    )


def _iforest_tree_to_timber(sk_tree: Any, tree_id: int, feat_map: Any) -> Tree:
    """Convert an isolation tree to a Timber Tree.

    Leaf values = traversal_depth + c(n_node_samples), pre-computed at
    parse time so C99 needs no floating-point log().
    """
    children_left  = sk_tree.children_left
    children_right = sk_tree.children_right
    feature        = sk_tree.feature
    threshold      = sk_tree.threshold
    n_node_samples = sk_tree.n_node_samples
    n_nodes        = sk_tree.node_count
    TREE_LEAF = -1

    # BFS to compute depths
    depths = [0] * n_nodes
    queue = [0]
    while queue:
        idx = queue.pop(0)
        if children_left[idx] != TREE_LEAF:
            depths[children_left[idx]]  = depths[idx] + 1
            depths[children_right[idx]] = depths[idx] + 1
            queue.append(children_left[idx])
            queue.append(children_right[idx])

    nodes: list[TreeNode] = []
    for i in range(n_nodes):
        is_leaf = children_left[i] == TREE_LEAF
        # Remap subsampled feature index → original feature index
        orig_feat = int(feat_map[int(feature[i])]) if not is_leaf else -1
        # Pre-compute path-length contribution for this leaf
        leaf_val = float(depths[i]) + _c_factor(int(n_node_samples[i])) if is_leaf else 0.0
        nodes.append(TreeNode(
            node_id=i,
            feature_index=orig_feat,
            threshold=float(threshold[i]) if not is_leaf else 0.0,
            left_child=int(children_left[i])  if not is_leaf else -1,
            right_child=int(children_right[i]) if not is_leaf else -1,
            is_leaf=is_leaf,
            leaf_value=leaf_val,
            depth=depths[i],
            default_left=True,
        ))

    max_depth_val = max(depths) if depths else 0
    tree = Tree(tree_id=tree_id, nodes=nodes, max_depth=max_depth_val)
    tree.recount()
    return tree


# ---------------------------------------------------------------------------
# One-Class SVM
# ---------------------------------------------------------------------------

def _parse_one_class_svm(est: Any) -> SVMStage:
    """Convert sklearn OneClassSVM to SVMStage(is_one_class=True)."""
    kernel = est.kernel.lower()
    n_features = int(est.n_features_in_)
    sv = est.support_vectors_.astype(np.float64).tolist()
    # dual_coef_ shape is (1, n_sv) for one-class
    dc = est.dual_coef_.astype(np.float64).flatten().tolist()
    # C99 emitter computes: decision = rho[0] + sum(dual_coef * kernel)
    # sklearn: decision_function = sum(...) + intercept_  => store rho = intercept_
    rho = [float(est.intercept_[0])]

    gamma_val: float
    if est.gamma == "scale":
        gamma_val = float(est._gamma)
    elif est.gamma == "auto":
        gamma_val = 1.0 / n_features
    else:
        gamma_val = float(est.gamma)

    return SVMStage(
        stage_name="sklearn_one_class_svm",
        stage_type="svm",
        kernel_type=kernel,
        support_vectors=sv,
        dual_coef=dc,
        rho=rho,
        n_support=[len(sv)],
        gamma=gamma_val,
        coef0=float(est.coef0),
        degree=int(est.degree),
        n_features=n_features,
        n_classes=1,
        objective=Objective.CUSTOM,
        post_transform="none",
        is_one_class=True,
    )


# ---------------------------------------------------------------------------
# Gaussian Naive Bayes
# ---------------------------------------------------------------------------

def _parse_naive_bayes(est: Any) -> NaiveBayesStage:
    """Convert sklearn GaussianNB to NaiveBayesStage.

    Pre-computes log-variance constants and inverse-2-variance to avoid
    log() and division at C99 inference time.
    """
    n_classes  = int(est.class_count_.shape[0])
    n_features = int(est.theta_.shape[1])
    TWO_PI     = 2.0 * math.pi

    log_prior   = np.log(est.class_prior_).tolist()
    theta       = est.theta_.astype(np.float64).tolist()
    # var_ already includes var_smoothing
    var         = est.var_.astype(np.float64)
    log_vc      = (-0.5 * np.log(TWO_PI * var)).tolist()
    inv_2v      = (1.0 / (2.0 * var)).tolist()

    return NaiveBayesStage(
        stage_name="sklearn_gaussian_nb",
        stage_type="naive_bayes",
        log_prior=log_prior,
        theta=theta,
        log_var_const=log_vc,
        inv_2var=inv_2v,
        n_classes=n_classes,
        n_features=n_features,
    )


# ---------------------------------------------------------------------------
# Gaussian Process Regressor
# ---------------------------------------------------------------------------

def _parse_gpr(est: Any) -> GPRStage:
    """Convert sklearn GaussianProcessRegressor (RBF kernel) to GPRStage.

    Supports: RBF, ConstantKernel * RBF, WhiteKernel additive noise.
    """
    n_features = int(est.X_train_.shape[1])

    # Extract kernel hyperparameters from the *fitted* kernel
    length_scale, amplitude = _extract_rbf_params(est.kernel_)

    # alpha_ = K_inv @ (y - y_mean)  (already computed by sklearn)
    alpha_vec = est.alpha_.astype(np.float64).flatten().tolist()
    X_tr = est.X_train_.astype(np.float64).tolist()

    _ym = getattr(est, "_y_train_mean", 0.0)
    _ys = getattr(est, "_y_train_std",  1.0)
    y_mean = float(np.ravel(_ym)[0]) if hasattr(_ym, "__len__") else float(_ym)
    y_std  = (float(np.ravel(_ys)[0]) if hasattr(_ys, "__len__") else float(_ys)) or 1.0

    return GPRStage(
        stage_name="sklearn_gpr",
        stage_type="gpr",
        X_train=X_tr,
        alpha=alpha_vec,
        length_scale=length_scale,
        amplitude=amplitude,
        y_train_mean=y_mean,
        y_train_std=y_std,
        n_features=n_features,
    )


def _extract_rbf_params(kernel: Any) -> tuple[float, float]:
    """Walk a sklearn kernel tree and return (length_scale, amplitude)."""
    cls = type(kernel).__name__
    if cls == "RBF":
        return float(kernel.length_scale), 1.0
    if cls == "ConstantKernel":
        return 1.0, float(kernel.constant_value ** 0.5)
    if cls in ("Product", "Sum"):
        ls, amp = 1.0, 1.0
        for part in (kernel.k1, kernel.k2):
            pn = type(part).__name__
            if pn == "RBF":
                ls = float(part.length_scale)
            elif pn == "ConstantKernel":
                amp = float(part.constant_value ** 0.5)
            elif pn == "WhiteKernel":
                pass  # noise handled by alpha_
            elif pn in ("Product", "Sum"):
                sub_ls, sub_amp = _extract_rbf_params(part)
                if sub_ls != 1.0:
                    ls = sub_ls
                if sub_amp != 1.0:
                    amp = sub_amp
        return ls, amp
    # Fallback
    if hasattr(kernel, "length_scale"):
        return float(kernel.length_scale), 1.0
    return 1.0, 1.0


# ---------------------------------------------------------------------------
# k-Nearest Neighbours
# ---------------------------------------------------------------------------

def _parse_knn(est: Any) -> KNNStage:
    """Convert sklearn KNeighborsClassifier/Regressor to KNNStage."""
    cls_name   = type(est).__name__
    is_clf     = cls_name == "KNeighborsClassifier"
    n_features = int(est.n_features_in_)
    k          = int(est.n_neighbors)
    metric_map = {"euclidean": "euclidean", "l2": "euclidean",
                  "manhattan": "manhattan", "l1": "manhattan",
                  "minkowski": "euclidean"}
    metric = metric_map.get(str(est.metric).lower(), "euclidean")

    X_tr = est._fit_X.astype(np.float64).tolist()

    if is_clf:
        n_classes = int(len(est.classes_))
        # Store one-hot encoded class index as y_train
        y_raw = est._y.astype(np.float64)
        y_tr  = [[float(v)] for v in y_raw]
        n_outputs = 1
    else:
        n_classes = 0
        y_raw = np.array(est._y, dtype=np.float64)
        if y_raw.ndim == 1:
            y_raw = y_raw.reshape(-1, 1)
        y_tr      = y_raw.tolist()
        n_outputs = int(y_raw.shape[1])

    return KNNStage(
        stage_name="sklearn_knn",
        stage_type="knn",
        X_train=X_tr,
        y_train=y_tr,
        k=k,
        metric=metric,
        task_type="classifier" if is_clf else "regressor",
        n_classes=n_classes,
        n_features=n_features,
        n_outputs=n_outputs,
    )


def _try_parse_scaler(step: Any, name: str) -> Optional[ScalerStage]:
    """Try to convert a sklearn scaler step into a ScalerStage."""
    try:
        from sklearn.preprocessing import StandardScaler
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
