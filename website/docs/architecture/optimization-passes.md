---
sidebar_position: 3
title: Optimization Passes
---

# Optimization Passes

Timber runs 6 domain-specific optimization passes on the IR. Each pass is designed for tree-based models and operates on the IR in-place.

## Pass 1: Dead Leaf Elimination

**File:** `timber/optimizer/dead_leaf.py`

Prunes leaves whose contribution is negligible relative to the maximum leaf value in the tree.

**Algorithm:**
1. Compute `max_leaf_value` across all leaves in the tree
2. If `|leaf_value| / |max_leaf_value| < threshold` (default: 1e-6), mark the leaf as dead
3. If both children of a node are dead, collapse the node to a leaf with the weighted average of children

**Effect:** Reduces tree depth and generated code size. A 50-tree model may lose 2-5% of nodes.

## Pass 2: Constant Feature Detection

**File:** `timber/optimizer/constant_feature.py`

Identifies internal nodes where both children have identical leaf values — the split is redundant regardless of the input.

**Algorithm:**
1. Post-order traversal of each tree
2. If `left_child.leaf_value == right_child.leaf_value` for a node, replace the node with a leaf

**Effect:** Eliminates unnecessary comparisons. Common in models trained with early stopping where some splits are noise.

## Pass 3: Threshold Quantization

**File:** `timber/optimizer/threshold_quant.py`

Analyzes all split thresholds per feature to determine the minimum precision required.

**Algorithm:**
1. Collect all thresholds for each feature across all trees
2. Compute the minimum gap between consecutive thresholds
3. Determine if int8, int16, float16, or float32 is sufficient
4. Store precision metadata as `QuantizationHint` annotations

**Effect:** Produces metadata for potential future narrow-type SIMD backends. Currently informational.

## Pass 4: Frequency-Ordered Branch Sorting

**File:** `timber/optimizer/branch_sort.py`

Reorders tree children so the most frequently taken branch is the fall-through path.

**Algorithm:**
1. Feed calibration data through each tree
2. Count left/right branch frequencies at each node
3. If `right_count > left_count`, swap children and invert the comparison
4. The "fall-through" (more common) branch becomes the first clause in generated `if/else`

**Effect:** Improves CPU branch prediction hit rate. Requires calibration data (`--calibration-data`). When no data is provided, the pass is a no-op.

## Pass 5: Pipeline Fusion

**File:** `timber/optimizer/pipeline_fusion.py`

Absorbs a preceding `ScalerStage` into tree thresholds.

**Algorithm:**
1. If the IR has `[ScalerStage(μ, σ), TreeEnsembleStage]`
2. For each split on feature `i`: `θ' = θ × σ_i + μ_i`
3. Remove the `ScalerStage` from the pipeline

**Effect:** Eliminates the entire preprocessing step at inference time. The compiled code operates directly on raw features. This is the key optimization for sklearn Pipeline deployments.

**Mathematical guarantee:** The transformation `(x - μ) / σ < θ` is equivalent to `x < θ × σ + μ`, so predictions are numerically identical.

## Pass 6: Vectorization Analysis

**File:** `timber/optimizer/vectorize.py`

Analyzes tree structure to identify SIMD batching opportunities.

**Algorithm:**
1. Compute depth profile of each tree
2. Analyze feature access patterns (sequential vs. random)
3. Identify groups of structurally identical trees
4. Produce `VectorizationHint` annotations

**Effect:** Produces metadata for future SIMD code generation. Trees with identical structure can share a single control flow and process multiple inputs via SIMD lanes.

## Pass Execution Order

The passes run in the order listed above. This order matters:

1. **Dead leaf** first — reduces tree size before other passes analyze it
2. **Constant feature** second — further simplification
3. **Threshold quant** third — needs final threshold values
4. **Branch sort** fourth — needs final tree structure
5. **Pipeline fusion** fifth — modifies thresholds, must run before quant analysis is consumed
6. **Vectorization** last — needs the final, fully optimized tree structure

## Audit Integration

Each pass logs its result to the audit trail:

```json
{
  "name": "dead_leaf_elimination",
  "changed": true,
  "nodes_before": 1550,
  "nodes_after": 1523,
  "duration_ms": 1.2
}
```
