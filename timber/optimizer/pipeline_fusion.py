"""Pass 5: Pipeline Fusion.

Analyze preprocessing stages for fusion opportunities. Where a scaler is
followed directly by a tree ensemble, absorb the scaler's per-feature affine
transforms into the split thresholds — eliminating the scaling step at runtime.
"""

from __future__ import annotations

from typing import Any

from timber.ir.model import (
    TimberIR,
    ScalerStage,
    TreeEnsembleStage,
    PipelineStage,
)


def pipeline_fusion(
    ir: TimberIR,
) -> tuple[bool, TimberIR, dict[str, Any]]:
    """Fuse scaler stages into downstream tree ensemble split thresholds.

    For a StandardScaler with mean/scale, the transform is:
        x_scaled = (x - mean) / scale

    A tree split: x_scaled < threshold
    Becomes:      x < threshold * scale + mean

    This eliminates the scaler stage entirely at inference time.

    Returns (changed, new_ir, details).
    """
    if len(ir.pipeline) < 2:
        return False, ir, {"skipped": "fewer than 2 pipeline stages"}

    fused_scalers: list[str] = []
    stages_to_remove: list[int] = []

    for i in range(len(ir.pipeline) - 1):
        stage = ir.pipeline[i]
        next_stage = ir.pipeline[i + 1]

        if isinstance(stage, ScalerStage) and isinstance(next_stage, TreeEnsembleStage):
            success = _fuse_scaler_into_ensemble(stage, next_stage)
            if success:
                fused_scalers.append(stage.stage_name)
                stages_to_remove.append(i)

    # Remove fused scaler stages (reverse order to preserve indices)
    for idx in reversed(stages_to_remove):
        ir.pipeline.pop(idx)

    changed = len(fused_scalers) > 0
    details: dict[str, Any] = {
        "scalers_fused": fused_scalers,
        "stages_removed": len(stages_to_remove),
    }
    return changed, ir, details


def _fuse_scaler_into_ensemble(
    scaler: ScalerStage,
    ensemble: TreeEnsembleStage,
) -> bool:
    """Absorb scaler transforms into tree thresholds.

    Returns True if fusion was applied.
    """
    if not scaler.means or not scaler.scales:
        return False

    # Build a map: feature_index -> (mean, scale)
    transform_map: dict[int, tuple[float, float]] = {}
    for idx, feat_idx in enumerate(scaler.feature_indices):
        if idx < len(scaler.means) and idx < len(scaler.scales):
            scale = scaler.scales[idx]
            if scale == 0.0:
                continue  # skip zero-variance features
            transform_map[feat_idx] = (scaler.means[idx], scale)

    if not transform_map:
        return False

    # Adjust thresholds in all trees
    modified = False
    for tree in ensemble.trees:
        for node in tree.nodes:
            if node.is_leaf:
                continue
            if node.feature_index in transform_map:
                mean, scale = transform_map[node.feature_index]
                # Reverse the scaling: threshold_original = threshold_scaled * scale + mean
                node.threshold = node.threshold * scale + mean
                modified = True

    return modified
