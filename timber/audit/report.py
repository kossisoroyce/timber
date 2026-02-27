"""Audit report generation.

Every compilation produces an audit_report.json documenting the complete
compilation history for regulatory review (financial services, healthcare,
automotive).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import timber
from timber.ir.model import TimberIR
from timber.optimizer.pipeline import OptimizationResult


@dataclass
class AuditReport:
    """Deterministic compilation audit report."""

    # Input artifact
    input_artifact_hash: str = ""
    input_artifact_path: str = ""
    input_format: str = ""

    # Timber version and target
    timber_version: str = timber.__version__
    target_spec: dict[str, Any] = field(default_factory=dict)

    # Compilation timestamp
    compiled_at: str = ""

    # Model summary
    model_summary: dict[str, Any] = field(default_factory=dict)

    # Optimizer pass log
    optimizer_log: dict[str, Any] = field(default_factory=dict)

    # Pruning summary
    pruning_summary: dict[str, Any] = field(default_factory=dict)

    # Quantization decisions
    quantization_decisions: dict[str, Any] = field(default_factory=dict)

    # Calibration data statistics
    calibration_stats: dict[str, Any] = field(default_factory=dict)

    # Output artifact
    output_artifact_hash: str = ""
    output_files: list[str] = field(default_factory=list)

    # Determinism guarantee
    deterministic: bool = True

    @staticmethod
    def generate(
        ir: TimberIR,
        optimization_result: Optional[OptimizationResult],
        input_path: str,
        input_format: str,
        target_spec: dict[str, Any],
        output_files: list[str],
    ) -> AuditReport:
        """Generate a complete audit report from compilation artifacts."""
        report = AuditReport()

        # Input artifact hash
        try:
            raw = Path(input_path).read_bytes()
            report.input_artifact_hash = hashlib.sha256(raw).hexdigest()
        except (FileNotFoundError, OSError):
            report.input_artifact_hash = ir.metadata.source_artifact_hash

        report.input_artifact_path = input_path
        report.input_format = input_format
        report.target_spec = target_spec
        report.compiled_at = datetime.now(timezone.utc).isoformat()

        # Model summary
        report.model_summary = ir.summary()

        # Optimizer log
        if optimization_result:
            report.optimizer_log = optimization_result.summary()

            # Extract pruning details
            for p in optimization_result.passes:
                if p.pass_name == "dead_leaf_elimination" and p.changed:
                    report.pruning_summary = p.details
                if p.pass_name == "threshold_quantization":
                    report.quantization_decisions = p.details

        # Output artifact hashes
        report.output_files = output_files
        if output_files:
            report.output_artifact_hash = _hash_output_files(output_files)

        return report

    def to_dict(self) -> dict[str, Any]:
        return {
            "timber_audit_report_version": "0.1",
            "input": {
                "artifact_hash_sha256": self.input_artifact_hash,
                "artifact_path": self.input_artifact_path,
                "format": self.input_format,
            },
            "compiler": {
                "timber_version": self.timber_version,
                "target_spec": self.target_spec,
                "compiled_at": self.compiled_at,
                "deterministic": self.deterministic,
            },
            "model_summary": self.model_summary,
            "optimization": self.optimizer_log,
            "pruning_summary": self.pruning_summary,
            "quantization_decisions": self.quantization_decisions,
            "calibration_stats": self.calibration_stats,
            "output": {
                "artifact_hash_sha256": self.output_artifact_hash,
                "files": self.output_files,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def write(self, path: str | Path) -> str:
        """Write the audit report to a JSON file. Returns the file path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return str(p)


def _hash_output_files(paths: list[str]) -> str:
    """Compute a combined SHA-256 hash of all output files."""
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            h.update(Path(p).read_bytes())
        except (FileNotFoundError, OSError):
            pass
    return h.hexdigest()
