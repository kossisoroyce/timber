# IR Reference

The **Timber IR** (Internal Representation) is a typed, serializable Python
dataclass tree defined in `timber/ir/model.py`. All frontend parsers produce
a `TimberIR`; all backends consume one.

---

## TimberIR

The root object.

```python
@dataclass
class TimberIR:
    schema: Schema
    pipeline: list[PipelineStage]
    metadata: Metadata
```

| Field | Type | Description |
|-------|------|-------------|
| `schema` | `Schema` | Input feature definitions |
| `pipeline` | `list[PipelineStage]` | Ordered processing stages applied left-to-right |
| `metadata` | `Metadata` | Provenance, source hash, framework version |

---

## Schema

```python
@dataclass
class Schema:
    fields: list[Field]
```

Describes the model's expected input vector.

## Field

```python
@dataclass
class Field:
    name: str
    dtype: FieldType
    index: int
```

| Field | Description |
|-------|-------------|
| `name` | Feature name as declared in the source model |
| `dtype` | `FieldType.FLOAT` or `FieldType.INT` |
| `index` | 0-based position in the input feature vector |

## FieldType

```python
class FieldType(str, Enum):
    FLOAT = "float"
    INT   = "int"
```

---

## Metadata

```python
@dataclass
class Metadata:
    source_framework: str
    source_framework_version: list[int]
    source_artifact_hash: str
    feature_names: list[str]
    objective_name: str
```

| Field | Description |
|-------|-------------|
| `source_framework` | `"xgboost"`, `"lightgbm"`, `"sklearn"`, `"catboost"`, `"onnx"`, `"urdf"` |
| `source_framework_version` | e.g. `[2, 1, 0]` |
| `source_artifact_hash` | SHA-256 hex digest of the source file |
| `feature_names` | Feature names in index order |
| `objective_name` | Raw objective string, e.g. `"binary:logistic"` |

---

## PipelineStage (base)

```python
@dataclass
class PipelineStage:
    stage_name: str
    stage_type: str
```

All concrete stages inherit from this. `stage_type` is set by `__post_init__`.

### Concrete stage types

| stage_type | Class | Purpose |
|------------|-------|---------|
| `"scaler"` | `ScalerStage` | Elementwise affine transform |
| `"encoder"` | `EncoderStage` | Categorical feature encoding |
| `"imputer"` | `ImputerStage` | Missing value imputation |
| `"tree_ensemble"` | `TreeEnsembleStage` | GBT or random forest inference |
| `"linear"` | `LinearStage` | Linear classifier / regressor (logistic regression, linear SVM) |
| `"svm"` | `SVMStage` | Support vector machine (RBF or linear kernel) |
| `"normalizer"` | `NormalizerStage` | Row-wise L1 / L2 / Max normalization |
| `"aggregator"` | `AggregatorStage` | Voting / stacking aggregation |
| `"kinematics"` | `KinematicsStage` | Robot forward kinematics (URDF source) |
| `"isolation_forest"` | `IsolationForestStage` | Isolation Forest anomaly detection |
| `"naive_bayes"` | `NaiveBayesStage` | Gaussian Naive Bayes classifier |
| `"gpr"` | `GPRStage` | Gaussian Process Regressor (RBF kernel) |
| `"knn"` | `KNNStage` | k-Nearest Neighbours classifier / regressor |

---

## ScalerStage

Applied before tree inference when the source model includes a preprocessing
scaler (e.g., `StandardScaler` in a scikit-learn `Pipeline`).

```python
@dataclass
class ScalerStage(PipelineStage):
    means: list[float]
    scales: list[float]
    feature_indices: list[int]
```

Transform applied: `output[i] = (input[i] - means[i]) / scales[i]`

When the optimizer's **pipeline_fusion** pass is active, `ScalerStage` is
folded into the tree thresholds and removed from the pipeline entirely.

---

## TreeEnsembleStage

The primary inference stage for all supported tree-based frameworks.

```python
@dataclass
class TreeEnsembleStage(PipelineStage):
    trees: list[Tree]
    n_features: int
    n_classes: int
    objective: Objective
    base_score: float
    per_class_base_scores: list[float]
    learning_rate: float
    is_boosted: bool
    annotations: dict[str, Any]
```

| Field | Description |
|-------|-------------|
| `trees` | Ordered list of `Tree` objects |
| `n_features` | Number of input features |
| `n_classes` | 1 for regression, 2 for binary classification, K for K-class multiclass |
| `objective` | `Objective` enum value |
| `base_score` | Scalar initial prediction (binary / regression) |
| `per_class_base_scores` | Per-class initial predictions (multiclass). Empty list if not applicable. Length == `n_classes` when present. |
| `learning_rate` | Shrinkage factor applied to each tree's leaf values |
| `is_boosted` | `True` for gradient boosted trees, `False` for random forest (bagged) |
| `annotations` | Arbitrary key-value metadata from the parser |

### Properties

```python
ensemble.n_trees    # int — len(trees)
ensemble.max_depth  # int — max(tree.max_depth for tree in trees)
```

### Tree ordering for multiclass

For `multi:softprob` / `multi:softmax` with `n_classes = K`:
- Trees are interleaved: tree `i` contributes to class `i % K`
- Accumulation: `scores[c] += traverse(tree_i)` for all `i` where `i % K == c`
- Initial values: `scores[c] = per_class_base_scores[c]` (or 0.0 if empty)
- Output transform: softmax over `scores[0..K-1]`

---

## Objective

```python
class Objective(str, Enum):
    REGRESSION                = "reg:squarederror"
    REGRESSION_LOGISTIC       = "reg:logistic"
    BINARY_CLASSIFICATION     = "binary:logistic"
    MULTICLASS_CLASSIFICATION = "multi:softprob"
    RANKING                   = "rank:pairwise"
```

Output transforms by objective:

| Objective | C transform | Output range |
|-----------|-------------|--------------|
| `REGRESSION` | identity (`sum`) | ℝ |
| `REGRESSION_LOGISTIC` | sigmoid | (0, 1) |
| `BINARY_CLASSIFICATION` | sigmoid | (0, 1) |
| `MULTICLASS_CLASSIFICATION` | softmax | (0, 1)^K, sums to 1 |
| `RANKING` | identity | ℝ |

---

## Tree

```python
@dataclass
class Tree:
    tree_id: int
    nodes: list[TreeNode]
```

`nodes` is a flat list. Node indices match list positions (`nodes[i].node_id == i`).

### Properties

```python
tree.n_nodes    # int — len(nodes)
tree.max_depth  # int — max(node.depth for node in nodes)
```

---

## TreeNode

```python
@dataclass
class TreeNode:
    node_id: int
    feature_index: int
    threshold: float
    left_child: int
    right_child: int
    leaf_value: float
    is_leaf: bool
    default_left: bool
    depth: int
```

| Field | Leaf node value | Internal node value |
|-------|-----------------|---------------------|
| `feature_index` | `-1` | `0`-based feature index |
| `threshold` | `0.0` | split threshold; go left if `input[feature_index] < threshold` |
| `left_child` | `-1` | index into `tree.nodes` |
| `right_child` | `-1` | index into `tree.nodes` |
| `leaf_value` | prediction value | `0.0` |
| `is_leaf` | `True` | `False` |
| `default_left` | `False` | `True` if NaN routes left, `False` if NaN routes right |
| `depth` | depth from root | depth from root |

### Traversal pseudocode

```
node = nodes[0]  # root
while not node.is_leaf:
    if input[node.feature_index] is NaN:
        go to nodes[node.left_child if node.default_left else node.right_child]
    elif input[node.feature_index] < node.threshold:
        go to nodes[node.left_child]
    else:
        go to nodes[node.right_child]
return node.leaf_value
```

---

## LinearStage

Represents a fully-connected linear layer: logistic regression (binary or multiclass) or linear regression.
Produced by the ONNX `LinearClassifier` and `LinearRegressor` parsers.

```python
@dataclass
class LinearStage(PipelineStage):
    weights: list[list[float]]  # shape [n_outputs, n_features] when multi_weights=True
                                # shape [1, n_features] when multi_weights=False
    biases: list[float]         # length n_outputs
    activation: str             # "none" | "sigmoid" | "softmax"
    n_classes: int
    multi_weights: bool
    n_features: int
    n_outputs: int
```

| Field | Description |
|-------|-------------|
| `weights` | Weight matrix. `multi_weights=False`: single row used for binary sigmoid. `multi_weights=True`: one row per class for softmax. |
| `biases` | Intercept / bias vector. |
| `activation` | `"sigmoid"` for binary classification, `"softmax"` for multiclass, `"none"` for regression. |
| `n_classes` | Number of target classes (1 for regression, 2 for binary, K for K-class). |
| `multi_weights` | `True` when the weight matrix has one row per output class. |

**C99 output:**
- `"sigmoid"`: `outputs[0] = 1 / (1 + exp(-(w · x + b)))`
- `"softmax"`: `outputs[c] = exp(score[c]) / Σ exp(score[k])` (numerically stable, double precision)
- `"none"`: `outputs[0] = w · x + b`

---

## SVMStage

Represents a support vector machine classifier or regressor.
Produced by the ONNX `SVMClassifier` and `SVMRegressor` parsers.

```python
@dataclass
class SVMStage(PipelineStage):
    support_vectors: list[list[float]]  # shape [n_support_vectors, n_features]
    dual_coef: list[float]              # shape [n_support_vectors]
    rho: float                          # intercept
    kernel: str                         # "rbf" | "linear" | "poly" | "sigmoid"
    gamma: float                        # RBF kernel parameter
    coef0: float                        # polynomial / sigmoid kernel parameter
    degree: int                         # polynomial degree
    n_features: int
    n_outputs: int
    is_classifier: bool
```

| Field | Description |
|-------|-------------|
| `support_vectors` | Training support vectors retained from `model.support_vectors_`. |
| `dual_coef` | Dual coefficients `α_i · y_i` from `model.dual_coef_`. |
| `rho` | Intercept term (`−model.intercept_`). |
| `kernel` | Kernel function. `"rbf"` and `"linear"` fully supported in C99. |
| `gamma` | Bandwidth for RBF: `exp(−gamma · ‖x − sv‖²)`. |

**C99 output (RBF):** `output = tanh(Σ dual_coef[i] · exp(−gamma · ‖x − sv_i‖²) + rho)`

---

## NormalizerStage

Row-wise normalization applied as a preprocessing step before the primary stage.
Produced by the ONNX `Normalizer` operator.

```python
@dataclass
class NormalizerStage(PipelineStage):
    norm: str        # "L1" | "L2" | "Max"
    n_features: int
```

| `norm` | Transform |
|--------|-----------|
| `"L1"` | divide each element by the sum of absolute values |
| `"L2"` | divide each element by the Euclidean norm |
| `"Max"` | divide each element by the maximum absolute value |

---

## EncoderStage

```python
@dataclass
class EncoderStage(PipelineStage):
    encoder_type: EncoderType
    categories_per_feature: dict[int, list[str]]
    feature_indices: list[int]
```

```python
class EncoderType(str, Enum):
    ORDINAL = "ordinal"
    ONEHOT  = "onehot"
```

---

## ImputerStage

```python
@dataclass
class ImputerStage(PipelineStage):
    strategy: ImputerStrategy
    fill_values: list[float]
    feature_indices: list[int]
```

```python
class ImputerStrategy(str, Enum):
    MEAN     = "mean"
    MEDIAN   = "median"
    CONSTANT = "constant"
```

---

## KinematicsStage

Represents a robot's forward-kinematics chain parsed from a URDF file.
Produced by `URDFParser` from `.urdf` XML.

```python
@dataclass
class KinematicsStage(PipelineStage):
    joints: list[JointSpec]
    base_link: str
    end_effector: str
```

| Field | Description |
|-------|-------------|
| `joints` | Ordered list of `JointSpec` objects (active DOF joints only; `fixed` joints included for transform chaining) |
| `base_link` | Name of the root link in the kinematic chain |
| `end_effector` | Name of the terminal link |

### Properties

```python
kin.n_dof  # int — number of actuated (non-fixed) joints
```

**C99 output:** `timber_fk(q, T, ctx)` fills a 16-element row-major `float[16]` buffer with the 4×4 homogeneous transform from base to end-effector. `timber_infer_single` delegates to `timber_fk` (same ABI as all other stages).

---

## JointSpec

One joint in a kinematic chain.

```python
@dataclass
class JointSpec:
    name: str
    joint_type: str          # "revolute" | "prismatic" | "continuous" | "fixed"
    axis: list[float]        # unit vector [x, y, z]
    origin_xyz: list[float]  # translation [x, y, z] from parent frame
    origin_rpy: list[float]  # rotation [r, p, y] from parent frame (radians)
    parent: str              # parent link name
    child: str               # child link name
    limit_lower: float       # lower position limit (rad or m)
    limit_upper: float       # upper position limit (rad or m)
```

| Field | Description |
|-------|-------------|
| `joint_type` | `"revolute"` / `"continuous"` — Rodrigues rotation about `axis`; `"prismatic"` — translation along `axis`; `"fixed"` — constant transform (no DOF) |
| `axis` | Normalized rotation / translation axis in the joint's local frame |
| `origin_xyz` | Position of the joint frame origin relative to the parent link frame |
| `origin_rpy` | Orientation of the joint frame expressed as roll-pitch-yaw (intrinsic XYZ) |
| `limit_lower` / `limit_upper` | Position limits; `0.0` / `0.0` for `continuous` and `fixed` joints |

---

## IsolationForestStage

Anomaly detection via average isolation path length. Output is
`−anomaly_score − offset` (positive → inlier, negative → outlier), matching
sklearn's `decision_function` sign convention.

```python
@dataclass
class IsolationForestStage(PipelineStage):
    trees: list[Tree]
    n_features: int
    max_samples: int    # training subsample size used for c(n) normalisation
    offset: float       # sklearn offset_ (typically negative)
```

| Field | Description |
|-------|-------------|
| `trees` | Flat-node isolation trees; leaf `leaf_value` = `depth + c(n_node_samples)` |
| `max_samples` | Size of the training subsample per tree (used to compute `c_max`) |
| `offset` | Decision threshold offset; added to `−anomaly_score` to form decision value |

**C99 output formula:** `outputs[0] = (float)(−exp2(−mean_path / c(max_samples)) − offset)`

---

## NaiveBayesStage

Gaussian Naive Bayes classifier. All log-arithmetic is pre-computed at
parse time; C99 inference performs only multiply-accumulate and `exp()` for softmax.

```python
@dataclass
class NaiveBayesStage(PipelineStage):
    log_prior: list[float]          # [n_classes]
    theta: list[list[float]]        # [n_classes, n_features]  mean per class/feature
    log_var_const: list[list[float]]# [n_classes, n_features]  −0.5·log(2π·var)
    inv_2var: list[list[float]]     # [n_classes, n_features]  1/(2·var)
    n_classes: int
    n_features: int
```

**C99 output formula:** `outputs[c] = softmax(log_prior[c] + Σ_f (log_var_const[c,f] − inv_2var[c,f]·(x[f]−theta[c,f])²))`

---

## GPRStage

Gaussian Process Regressor with RBF kernel. Prediction is a weighted sum of
kernel evaluations against the training set.

```python
@dataclass
class GPRStage(PipelineStage):
    X_train: list[list[float]]  # [n_train, n_features]
    alpha: list[float]          # [n_train]  K⁻¹(y − y_mean)
    length_scale: float
    amplitude: float
    y_train_mean: float         # target normalisation mean
    y_train_std: float          # target normalisation std
    n_features: int
```

| Field | Description |
|-------|-------------|
| `alpha` | Dual coefficients from `GaussianProcessRegressor.alpha_` |
| `length_scale` | RBF kernel length scale (extracted from fitted kernel tree) |
| `amplitude` | RBF amplitude (square-root of `ConstantKernel.constant_value`) |
| `y_train_mean` / `y_train_std` | Used when `normalize_y=True` |

**C99 output formula:** `outputs[0] = (float)((Σ_i amp²·exp(−inv2ls²·‖x−X_train[i]‖²)·alpha[i])·y_std + y_mean)`

---

## KNNStage

k-Nearest Neighbours lookup table — stores the full training set and labels.

```python
@dataclass
class KNNStage(PipelineStage):
    X_train: list[list[float]]  # [n_train, n_features]
    y_train: list[list[float]]  # [n_train, n_outputs]
    k: int
    metric: str          # "euclidean" | "manhattan"
    task_type: str       # "classifier" | "regressor"
    n_classes: int       # 0 for regressors
    n_features: int
    n_outputs: int
```

**Classifier C99 output:** majority vote → `outputs[0] = (float)winning_class_index`  
**Regressor C99 output:** mean of k-nearest labels per output dimension

---

## SVMStage (`is_one_class`)

`SVMStage` gained an `is_one_class: bool` field for `sklearn.svm.OneClassSVM`.
When `True`, `n_classes = 1` and `rho[0]` stores `intercept_[0]` directly
(not negated) so the C99 formula `decision = rho[0] + Σ(dual_coef·K)` matches
sklearn's `decision_function`.

---

## IR Serialization

`TimberIR` can be round-tripped to/from JSON:

```python
from timber.ir.model import TimberIR, to_dict, from_dict
import json

ir = parse_model("model.json")

# Serialize
d = to_dict(ir)
json_str = json.dumps(d, indent=2)

# Deserialize
d2 = json.loads(json_str)
ir2 = from_dict(d2)
```

The serialized JSON is written to `model.timber.json` in every compiled
artifact directory for inspection and debugging.

---

## Extending the IR

To add a new preprocessing stage:

1. Define a new `@dataclass` inheriting from `PipelineStage` in `timber/ir/model.py`
2. Add serialization in `_stage_to_dict()`
3. Add deserialization in `_stage_from_dict()`
4. Handle the new stage in `C99Emitter._emit_inference()` (or ignore it if the
   optimizer folds it away)

To add a new frontend parser:

1. Create `timber/frontends/<framework>_parser.py`
2. Implement `parse_<framework>_model(path: str) -> TimberIR`
3. Register in `timber/frontends/auto_detect.py`
