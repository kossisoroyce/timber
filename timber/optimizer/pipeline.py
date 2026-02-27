"""Optimizer pipeline — orchestrates pass execution on Timber IR."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from timber.ir.model import TimberIR


@dataclass
class PassResult:
    """Result of a single optimizer pass."""
    pass_name: str
    changed: bool
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Aggregate result of the full optimization pipeline."""
    ir: TimberIR
    passes: list[PassResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def any_changed(self) -> bool:
        return any(p.changed for p in self.passes)

    def summary(self) -> dict[str, Any]:
        return {
            "total_passes": len(self.passes),
            "passes_applied": sum(1 for p in self.passes if p.changed),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "pass_details": [
                {
                    "name": p.pass_name,
                    "changed": p.changed,
                    "duration_ms": round(p.duration_ms, 2),
                    "details": p.details,
                }
                for p in self.passes
            ],
        }


class OptimizerPipeline:
    """Runs a sequence of optimization passes on a TimberIR."""

    def __init__(
        self,
        dead_leaf_threshold: float = 0.001,
        enable_quantization: bool = True,
        calibration_data: Optional[Any] = None,
        enable_fusion: bool = True,
    ):
        self.dead_leaf_threshold = dead_leaf_threshold
        self.enable_quantization = enable_quantization
        self.calibration_data = calibration_data
        self.enable_fusion = enable_fusion

    def run(self, ir: TimberIR) -> OptimizationResult:
        """Run all optimizer passes on the given IR and return an OptimizationResult."""
        from timber.optimizer.dead_leaf import dead_leaf_elimination
        from timber.optimizer.constant_feature import constant_feature_detection
        from timber.optimizer.threshold_quant import threshold_quantization
        from timber.optimizer.branch_sort import frequency_branch_sort
        from timber.optimizer.pipeline_fusion import pipeline_fusion

        working = ir.deep_copy()
        results: list[PassResult] = []
        t0 = time.monotonic()

        # Pass 1: Dead Leaf Elimination
        result, working = self._run_pass(
            "dead_leaf_elimination",
            lambda ir_: dead_leaf_elimination(ir_, threshold=self.dead_leaf_threshold),
            working,
        )
        results.append(result)

        # Pass 2: Constant Feature Detection
        result, working = self._run_pass(
            "constant_feature_detection",
            constant_feature_detection,
            working,
        )
        results.append(result)

        # Pass 3: Threshold Quantization
        if self.enable_quantization:
            result, working = self._run_pass(
                "threshold_quantization",
                threshold_quantization,
                working,
            )
            results.append(result)

        # Pass 4: Frequency-Ordered Branch Sorting
        if self.calibration_data is not None:
            result, working = self._run_pass(
                "frequency_branch_sort",
                lambda ir_: frequency_branch_sort(ir_, self.calibration_data),
                working,
            )
            results.append(result)

        # Pass 5: Pipeline Fusion
        if self.enable_fusion:
            result, working = self._run_pass(
                "pipeline_fusion",
                pipeline_fusion,
                working,
            )
            results.append(result)

        # Pass 6: Vectorization Analysis (annotation-only)
        from timber.optimizer.vectorize import vectorization_analysis
        result, working = self._run_pass(
            "vectorization_analysis",
            lambda ir_: (True, ir_, vectorization_analysis(ir_)),
            working,
        )
        results.append(result)

        total_ms = (time.monotonic() - t0) * 1000.0

        return OptimizationResult(
            ir=working,
            passes=results,
            total_duration_ms=total_ms,
        )

    @staticmethod
    def _run_pass(name: str, fn, ir: TimberIR) -> tuple[PassResult, TimberIR]:
        t0 = time.monotonic()
        try:
            changed, new_ir, details = fn(ir)
        except Exception as exc:
            duration = (time.monotonic() - t0) * 1000.0
            return PassResult(
                pass_name=name,
                changed=False,
                duration_ms=duration,
                details={"error": str(exc)},
            ), ir

        duration = (time.monotonic() - t0) * 1000.0
        return PassResult(
            pass_name=name,
            changed=changed,
            duration_ms=duration,
            details=details,
        ), new_ir
