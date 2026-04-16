"""Worst-Case Execution Time (WCET) analysis for compiled inference models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from timber.ir.model import (
    GPRStage,
    ImputerStage,
    KNNStage,
    LinearStage,
    NaiveBayesStage,
    NormalizerStage,
    ScalerStage,
    SVMStage,
    TimberIR,
    TreeEnsembleStage,
)

DISCLAIMER = (
    "ADVISORY ONLY — WCET estimates use a simplified analytical model that does "
    "not account for cache effects, pipeline stalls, branch misprediction penalties, "
    "or memory hierarchy latency. Actual worst-case execution time may be 3-10x "
    "higher. For safety-critical applications, use hardware-in-the-loop measurement "
    "tools (e.g., aiT, RapiTime, Bound-T) to obtain certified WCET bounds."
)


@dataclass
class WCETResult:
    """WCET analysis result."""
    raw_total_cycles_worst: int
    raw_total_cycles_avg: int
    total_cycles_worst: int
    total_cycles_avg: int
    total_time_us_worst: float
    total_time_us_avg: float
    per_stage: list[dict[str, Any]]
    arch: str
    clock_mhz: float
    safety_margin: float
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_total_cycles_worst": self.raw_total_cycles_worst,
            "raw_total_cycles_avg": self.raw_total_cycles_avg,
            "total_cycles_worst": self.total_cycles_worst,
            "total_cycles_avg": self.total_cycles_avg,
            "total_time_us_worst": self.total_time_us_worst,
            "total_time_us_avg": self.total_time_us_avg,
            "per_stage": self.per_stage,
            "arch": self.arch,
            "clock_mhz": self.clock_mhz,
            "safety_margin": self.safety_margin,
            "disclaimer": self.disclaimer,
        }


# Architecture-specific cycle costs per operation
ARCH_COSTS = {
    "cortex-m4": {
        "compare": 1, "branch": 3, "load": 2, "store": 2,
        "add_f32": 1, "mul_f32": 1, "div_f32": 14, "exp_f32": 40,
        "sqrt_f32": 14, "call_overhead": 6, "loop_overhead": 3,
    },
    "cortex-m7": {
        "compare": 1, "branch": 2, "load": 1, "store": 1,
        "add_f32": 1, "mul_f32": 1, "div_f32": 14, "exp_f32": 35,
        "sqrt_f32": 14, "call_overhead": 4, "loop_overhead": 2,
    },
    "x86_64": {
        "compare": 1, "branch": 1, "load": 4, "store": 4,
        "add_f32": 1, "mul_f32": 1, "div_f32": 11, "exp_f32": 20,
        "sqrt_f32": 11, "call_overhead": 3, "loop_overhead": 1,
    },
    "aarch64": {
        "compare": 1, "branch": 1, "load": 3, "store": 3,
        "add_f32": 1, "mul_f32": 1, "div_f32": 10, "exp_f32": 25,
        "sqrt_f32": 10, "call_overhead": 4, "loop_overhead": 1,
    },
    "riscv64": {
        "compare": 1, "branch": 2, "load": 3, "store": 3,
        "add_f32": 1, "mul_f32": 1, "div_f32": 12, "exp_f32": 30,
        "sqrt_f32": 12, "call_overhead": 4, "loop_overhead": 2,
    },
}


def analyze_wcet(
    ir: TimberIR,
    arch: str = "cortex-m4",
    clock_mhz: float = 100.0,
    safety_margin: float = 3.0,
) -> dict[str, Any]:
    """Analyze worst-case execution time for a compiled model.

    Args:
        ir: Timber IR model
        arch: Target architecture (cortex-m4, cortex-m7, x86_64, aarch64, riscv64)
        clock_mhz: Clock frequency in MHz
        safety_margin: Multiplier applied to raw estimates (default 3.0)

    Returns:
        Dict with raw and margined cycle/time estimates, per_stage, and disclaimer
    """
    warnings.warn(DISCLAIMER, stacklevel=2)
    costs = ARCH_COSTS.get(arch)
    if costs is None:
        raise ValueError(f"Unknown architecture: {arch!r}. Supported: {list(ARCH_COSTS)}")

    per_stage = []
    total_worst = 0
    total_avg = 0

    for stage in ir.pipeline:
        if isinstance(stage, TreeEnsembleStage):
            worst, avg = _wcet_tree_ensemble(stage, costs)
        elif isinstance(stage, LinearStage):
            worst, avg = _wcet_linear(stage, costs)
        elif isinstance(stage, SVMStage):
            worst, avg = _wcet_svm(stage, costs)
        elif isinstance(stage, KNNStage):
            worst, avg = _wcet_knn(stage, costs)
        elif isinstance(stage, GPRStage):
            worst, avg = _wcet_gpr(stage, costs)
        elif isinstance(stage, NaiveBayesStage):
            worst, avg = _wcet_naive_bayes(stage, costs)
        elif isinstance(stage, ScalerStage):
            worst, avg = _wcet_scaler(stage, costs)
        elif isinstance(stage, NormalizerStage):
            worst, avg = _wcet_normalizer(stage, costs)
        elif isinstance(stage, ImputerStage):
            worst, avg = _wcet_imputer(stage, costs)
        else:
            worst = avg = 100  # conservative estimate for unknown stages

        per_stage.append({
            "stage": stage.stage_name,
            "stage_type": stage.stage_type,
            "cycles_worst": worst,
            "cycles_avg": avg,
        })
        total_worst += worst
        total_avg += avg

    us_per_cycle = 1.0 / clock_mhz

    margined_worst = int(total_worst * safety_margin)
    margined_avg = int(total_avg * safety_margin)

    return WCETResult(
        raw_total_cycles_worst=total_worst,
        raw_total_cycles_avg=total_avg,
        total_cycles_worst=margined_worst,
        total_cycles_avg=margined_avg,
        total_time_us_worst=margined_worst * us_per_cycle,
        total_time_us_avg=margined_avg * us_per_cycle,
        per_stage=per_stage,
        arch=arch,
        clock_mhz=clock_mhz,
        safety_margin=safety_margin,
        disclaimer=DISCLAIMER,
    ).to_dict()


def _wcet_tree_ensemble(stage: TreeEnsembleStage, costs: dict) -> tuple[int, int]:
    """WCET for tree ensemble: per-tree traversal * n_trees + aggregation."""
    n_trees = len(stage.trees)
    if n_trees == 0:
        return 0, 0

    # Per node: load feature, compare threshold, branch
    per_node = costs["load"] + costs["compare"] + costs["branch"]

    # Worst case: traverse max_depth nodes per tree
    max_depth = max(t.max_depth for t in stage.trees)
    per_tree_worst = (max_depth + 1) * per_node + costs["load"]  # +1 for leaf load

    # Average: ~60% of max depth
    avg_depth = max(1, int(max_depth * 0.6))
    per_tree_avg = (avg_depth + 1) * per_node + costs["load"]

    # Tree loop overhead
    tree_loop = n_trees * costs["loop_overhead"]

    # Aggregation: sum + possible sigmoid/softmax
    agg = n_trees * costs["add_f32"]
    if stage.n_classes > 1:
        # Softmax: exp per class + sum + div per class
        agg += stage.n_classes * (costs["exp_f32"] + costs["add_f32"] + costs["div_f32"])
    elif stage.objective.value in ("binary:logistic", "reg:logistic"):
        agg += costs["exp_f32"] + costs["add_f32"] + costs["div_f32"]

    worst = n_trees * per_tree_worst + tree_loop + agg + costs["call_overhead"]
    avg = n_trees * per_tree_avg + tree_loop + agg + costs["call_overhead"]
    return worst, avg


def _wcet_linear(stage: LinearStage, costs: dict) -> tuple[int, int]:
    """WCET for linear stage: dot product + bias + optional activation."""
    n = len(stage.weights)
    # dot product: n loads + n multiplies + n-1 adds + bias add
    ops = n * (costs["load"] + costs["mul_f32"]) + n * costs["add_f32"] + costs["add_f32"]
    if stage.activation == "sigmoid":
        ops += costs["exp_f32"] + costs["add_f32"] + costs["div_f32"]
    return ops + costs["call_overhead"], ops + costs["call_overhead"]


def _wcet_svm(stage: SVMStage, costs: dict) -> tuple[int, int]:
    """WCET for SVM: kernel evaluation per support vector."""
    n_sv = len(stage.support_vectors)
    n_feat = stage.n_features
    if stage.kernel_type == "rbf":
        # Per SV: n_feat subtracts + n_feat muls + sum + exp + mul coef
        per_sv = n_feat * (costs["add_f32"] + costs["mul_f32"]) + costs["add_f32"] + costs["exp_f32"] + costs["mul_f32"]
    elif stage.kernel_type == "linear":
        per_sv = n_feat * (costs["mul_f32"] + costs["add_f32"]) + costs["mul_f32"]
    else:
        per_sv = n_feat * 3 * costs["mul_f32"]
    total = n_sv * per_sv + costs["call_overhead"]
    return total, total


def _wcet_knn(stage: KNNStage, costs: dict) -> tuple[int, int]:
    """WCET for KNN: distance computation + partial sort."""
    n_train = len(stage.X_train)
    n_feat = stage.n_features
    # Distance computation per training sample
    per_sample = n_feat * (costs["add_f32"] + costs["mul_f32"]) + costs["sqrt_f32"]
    # Partial sort for k nearest
    sort_ops = stage.k * n_train * costs["compare"]
    total = n_train * per_sample + sort_ops + costs["call_overhead"]
    return total, total


def _wcet_gpr(stage: GPRStage, costs: dict) -> tuple[int, int]:
    """WCET for GPR: kernel evaluation per training point."""
    n_train = len(stage.X_train)
    n_feat = stage.n_features
    # Kernel evaluation per training point
    per_point = n_feat * (costs["add_f32"] + costs["mul_f32"]) + costs["exp_f32"] + costs["mul_f32"]
    total = n_train * per_point + costs["call_overhead"]
    return total, total


def _wcet_naive_bayes(stage: NaiveBayesStage, costs: dict) -> tuple[int, int]:
    """WCET for Naive Bayes: per-class likelihood computation."""
    n_cls = stage.n_classes
    n_feat = stage.n_features
    # Per class: n_feat * (sub + mul + add) + prior add
    per_class = n_feat * (costs["add_f32"] + costs["mul_f32"] + costs["add_f32"]) + costs["add_f32"]
    total = n_cls * per_class + costs["call_overhead"]
    return total, total


def _wcet_scaler(stage: ScalerStage, costs: dict) -> tuple[int, int]:
    """WCET for scaler: affine transform per feature."""
    n = len(stage.feature_indices)
    ops = n * (costs["add_f32"] + costs["mul_f32"] + costs["load"] + costs["store"])
    return ops + costs["call_overhead"], ops + costs["call_overhead"]


def _wcet_normalizer(stage: NormalizerStage, costs: dict) -> tuple[int, int]:
    """WCET for normalizer: norm computation + division."""
    n_feat = getattr(stage, 'n_features', 100)
    ops = n_feat * (costs["load"] + costs["mul_f32"] + costs["add_f32"]) + costs["sqrt_f32"]
    return ops + costs["call_overhead"], ops + costs["call_overhead"]


def _wcet_imputer(stage: ImputerStage, costs: dict) -> tuple[int, int]:
    """WCET for imputer: NaN check + conditional store per feature."""
    n = len(stage.feature_indices)
    # Per feature: NaN check + conditional store
    ops = n * (costs["compare"] + costs["branch"] + costs["store"])
    return ops + costs["call_overhead"], ops + costs["call_overhead"]
