"""Auto-detection of model formats and unified parse entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from timber.ir.model import TimberIR


SUPPORTED_FORMATS = ("xgboost", "lightgbm", "sklearn", "onnx", "catboost")


def detect_format(path: str | Path) -> Optional[str]:
    """Attempt to auto-detect the model format from the file contents.

    Returns one of: 'xgboost', 'lightgbm', or None if unrecognized.
    """
    path = Path(path)
    if not path.exists():
        return None

    suffix = path.suffix.lower()

    # Try JSON-based detection first
    if suffix == ".json":
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if "learner" in data or "gradient_booster" in data:
                return "xgboost"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # Try LightGBM text format detection
    if suffix in (".txt", ".model", ".lgb", ""):
        try:
            with open(path, "r") as f:
                head = f.read(2048)
            if "tree" in head.lower() and ("num_leaves" in head or "split_feature" in head):
                return "lightgbm"
            if "objective" in head and "max_feature_idx" in head:
                return "lightgbm"
        except UnicodeDecodeError:
            pass

    # sklearn pickle
    if suffix in (".pkl", ".pickle", ".joblib"):
        return "sklearn"

    # ONNX
    if suffix == ".onnx":
        return "onnx"

    # CatBoost JSON (check content)
    if suffix == ".json":
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if "oblivious_trees" in data:
                return "catboost"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # Fallback: try XGBoost binary format
    if suffix in (".bin", ".ubj", ".xgb"):
        return "xgboost"

    return None


def parse_model(path: str | Path, format_hint: Optional[str] = None) -> TimberIR:
    """Parse a model artifact and return a TimberIR.

    Args:
        path: Path to the model file.
        format_hint: Optional format hint ('xgboost', 'lightgbm').
            If omitted, the format is auto-detected.

    Returns:
        TimberIR

    Raises:
        ValueError: If the format cannot be detected or is unsupported.
    """
    path = Path(path)
    fmt = format_hint or detect_format(path)

    if fmt is None:
        raise ValueError(
            f"Cannot detect model format for '{path}'. "
            f"Provide --format hint. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    if fmt == "xgboost":
        from timber.frontends.xgboost_parser import parse_xgboost_json
        return parse_xgboost_json(path)
    elif fmt == "lightgbm":
        from timber.frontends.lightgbm_parser import parse_lightgbm_model
        return parse_lightgbm_model(path)
    elif fmt == "sklearn":
        from timber.frontends.sklearn_parser import parse_sklearn_model
        return parse_sklearn_model(path)
    elif fmt == "onnx":
        from timber.frontends.onnx_parser import parse_onnx_model
        return parse_onnx_model(path)
    elif fmt == "catboost":
        from timber.frontends.catboost_parser import parse_catboost_json
        return parse_catboost_json(path)
    else:
        raise ValueError(f"Unsupported format: '{fmt}'. Supported: {', '.join(SUPPORTED_FORMATS)}")
