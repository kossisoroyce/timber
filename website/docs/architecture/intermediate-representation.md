---
sidebar_position: 2
title: Intermediate Representation
---

# Intermediate Representation (IR)

The Timber IR is the central data structure connecting front-ends to back-ends. It provides a framework-agnostic representation of tree-based models.

## Structure

A `TimberIR` is a pipeline of **stages**, where each stage represents a transformation:

```python
@dataclass
class TimberIR:
    stages: list[Stage]
    metadata: dict
```

## Stage Types

### `TreeEnsembleStage`

The core stage — an ensemble of decision trees.

```python
@dataclass
class TreeEnsembleStage:
    trees: list[Tree]
    n_features: int
    n_outputs: int
    objective: str           # "binary:logistic", "regression", "multiclass"
    base_score: float        # Bias term (in logit space for classification)
    feature_names: list[str]
```

### `ScalerStage`

A feature scaler (from sklearn Pipeline).

```python
@dataclass
class ScalerStage:
    means: list[float]       # Per-feature means
    scales: list[float]      # Per-feature standard deviations
```

The Pipeline Fusion optimizer pass absorbs this into tree thresholds.

### `VotingEnsembleStage`

Weighted average of multiple sub-models.

```python
@dataclass
class VotingEnsembleStage:
    sub_models: list[TimberIR]
    weights: list[float]
    voting: str              # "soft" or "hard"
```

### `StackingEnsembleStage`

Stacking with a meta-learner.

```python
@dataclass
class StackingEnsembleStage:
    base_models: list[TimberIR]
    meta_model: TimberIR
    passthrough: bool        # Also pass original features to meta-learner
```

## Tree Representation

Each tree is a list of nodes:

```python
@dataclass
class TreeNode:
    feature_index: int       # -1 for leaf nodes
    threshold: float         # Split threshold
    left_child: int          # Index of left child
    right_child: int         # Index of right child
    leaf_value: float        # Only valid for leaf nodes
    default_left: bool       # Missing value direction
```

Trees are stored in **depth-first pre-order** as flat arrays, enabling iterative traversal without recursion.

## Design Decisions

### Why a Pipeline of Stages?

Many real-world deployments use sklearn `Pipeline(StandardScaler, GradientBoosting)`. By representing the scaler as an explicit stage, the optimizer can fuse it into the tree thresholds, eliminating preprocessing overhead.

### Why Flat Arrays?

The flat array representation maps directly to C's `static const` arrays, enabling zero-copy code generation. No pointer chasing, no dynamic allocation, cache-friendly layout.

### Why Double-Precision Accumulation?

Tree outputs are summed in `double` precision before final `float` cast. This prevents catastrophic cancellation when many small leaf values are summed, especially in models with hundreds of trees. The generated C code uses `double sum = 0.0;` for accumulation.
