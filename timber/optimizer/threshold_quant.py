"""Pass 3: Threshold Quantization.

Analyze split thresholds for precision requirements. Where float32 precision
is unnecessary (integer counts, binary flags, bounded ordinals), quantize
thresholds to float16 or int8.
"""

from __future__ import annotations

import math
from typing import Any

from timber.ir.model import TimberIR, TreeEnsembleStage


def threshold_quantization(
    ir: TimberIR,
) -> tuple[bool, TimberIR, dict[str, Any]]:
    """Analyze and tag thresholds for quantization.

    This pass does not modify the actual threshold values — it annotates
    features with their minimum required precision, which the code generator
    uses to select storage types.

    Returns (changed, new_ir, details).
    """
    ensemble = ir.get_tree_ensemble()
    if ensemble is None:
        return False, ir, {"skipped": "no tree ensemble found"}

    # Collect all thresholds per feature
    feature_thresholds: dict[int, list[float]] = {}
    for tree in ensemble.trees:
        for node in tree.nodes:
            if not node.is_leaf and node.feature_index >= 0:
                feat = node.feature_index
                if feat not in feature_thresholds:
                    feature_thresholds[feat] = []
                feature_thresholds[feat].append(node.threshold)

    if not feature_thresholds:
        return False, ir, {"skipped": "no split thresholds found"}

    # Classify each feature's precision
    quantization_map: dict[int, str] = {}
    int8_features: list[int] = []
    float16_features: list[int] = []
    float32_features: list[int] = []

    for feat, thresholds in sorted(feature_thresholds.items()):
        precision = _classify_precision(thresholds)
        quantization_map[feat] = precision
        if precision == "int8":
            int8_features.append(feat)
        elif precision == "float16":
            float16_features.append(feat)
        else:
            float32_features.append(feat)

    # Store quantization hints in metadata
    ir.metadata.compilation_hints["quantization_map"] = {
        str(k): v for k, v in quantization_map.items()
    }

    changed = len(int8_features) + len(float16_features) > 0
    details: dict[str, Any] = {
        "features_analyzed": len(feature_thresholds),
        "int8_features": int8_features,
        "float16_features": float16_features,
        "float32_features": float32_features,
    }
    return changed, ir, details


def _classify_precision(thresholds: list[float]) -> str:
    """Determine minimum precision required for a set of thresholds.

    - int8: all thresholds are integers in [-128, 127]
    - float16: all thresholds fit in float16 range with acceptable precision
    - float32: default
    """
    if not thresholds:
        return "float32"

    all_integer = True
    all_small = True

    for t in thresholds:
        if not math.isfinite(t):
            return "float32"

        # Check if integer
        if t != math.floor(t):
            all_integer = False

        # Check float16 range: ~[-65504, 65504] with limited precision
        if abs(t) > 65504.0:
            all_small = False

    if all_integer:
        min_t = min(thresholds)
        max_t = max(thresholds)
        if -128 <= min_t and max_t <= 127:
            return "int8"

    if all_small:
        # Check if float16 precision is sufficient
        # float16 has ~3.3 decimal digits of precision
        for t in thresholds:
            if t == 0.0:
                continue
            # Check if float16 round-trip preserves the value
            import struct
            try:
                import numpy as np
                f16 = np.float16(t)
                if abs(float(f16) - t) / max(abs(t), 1e-10) > 0.001:
                    return "float32"
            except (ImportError, OverflowError):
                # Without numpy, use a heuristic: if fewer than 4 significant digits
                if _significant_digits(t) > 3:
                    return "float32"
        return "float16"

    return "float32"


def _significant_digits(value: float) -> int:
    """Estimate the number of significant digits in a float."""
    if value == 0.0:
        return 1
    s = f"{abs(value):.15g}"
    s = s.lstrip("0").replace(".", "").rstrip("0")
    return len(s)
