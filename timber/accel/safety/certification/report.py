"""Certification report generator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from timber.accel.version import __version__
from timber.ir.model import TimberIR

GENERIC_DISCLAIMER = (
    "ADVISORY ONLY \u2014 All compliance analyses in this report use heuristic "
    "pattern matching and are NOT a substitute for formal certification tooling. "
    "Results must be independently verified by qualified assessors before use "
    "in any safety-critical system."
)


@dataclass
class CertificationReport:
    """Comprehensive certification report for safety-critical deployment."""
    standard: str
    level: str
    model_summary: dict[str, Any]
    compliance_result: dict[str, Any]
    wcet_result: Optional[dict[str, Any]] = None
    generated_at: str = ""
    timber_accel_version: str = __version__
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_compliant(self) -> bool:
        return self.compliance_result.get("compliant", False)

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "DISCLAIMER: " + GENERIC_DISCLAIMER,
            "=" * 72,
            "",
            f"Certification Report \u2014 {self.standard} Level {self.level}",
            f"Generated: {self.generated_at}",
            f"Timber Accelerate: v{self.timber_accel_version}",
            f"Compliant: {'YES' if self.is_compliant else 'NO'}",
            f"Rules checked: {self.compliance_result.get('rules_checked', 0)}",
            f"Violations: {len(self.compliance_result.get('violations', []))}",
        ]
        if self.wcet_result:
            lines.append(f"WCET (worst): {self.wcet_result.get('total_time_us_worst', 'N/A')} µs")
        if self.recommendations:
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": GENERIC_DISCLAIMER,
            "standard": self.standard,
            "level": self.level,
            "generated_at": self.generated_at,
            "timber_accel_version": self.timber_accel_version,
            "compliant": self.is_compliant,
            "model_summary": self.model_summary,
            "compliance_result": self.compliance_result,
            "wcet_result": self.wcet_result,
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: str | Path) -> str:
        p = Path(path)
        p.write_text(self.to_json())
        return str(p)


def generate_certification_report(
    ir: TimberIR,
    profile_name: str,
    include_wcet: bool = False,
    arch: str = "cortex-m4",
    clock_mhz: float = 100.0,
) -> CertificationReport:
    """Generate a full certification report.

    Args:
        ir: Timber IR model
        profile_name: Compliance profile (do_178c, iso_26262, iec_62304)
        include_wcet: Whether to include WCET analysis
        arch: Architecture for WCET (if included)
        clock_mhz: Clock frequency for WCET

    Returns:
        CertificationReport instance
    """
    from timber.accel.safety.certification.profiles import check_compliance, load_compliance_profile
    from timber.codegen.c99 import C99Emitter

    # Generate baseline C code for analysis
    emitter = C99Emitter()
    output = emitter.emit(ir)

    # Load profile
    profile_data = load_compliance_profile(profile_name)

    # Run compliance checks
    compliance_result = check_compliance(output.model_c, profile_name)

    # Optional WCET
    wcet_result = None
    if include_wcet:
        from timber.accel.safety.realtime.wcet import analyze_wcet
        wcet_result = analyze_wcet(ir, arch=arch, clock_mhz=clock_mhz)

    # Build recommendations
    recommendations = []
    if not compliance_result["compliant"]:
        recommendations.append("Address all compliance violations before deployment.")
    if wcet_result and wcet_result.get("total_time_us_worst", 0) > 1000:
        recommendations.append("WCET exceeds 1ms — consider model simplification for real-time use.")

    # Standard-specific checks
    standard = profile_data.get("profile", {}).get("standard", profile_name)
    level = profile_data.get("profile", {}).get("level", "")

    if standard == "DO-178C":
        from timber.accel.safety.certification.do_178c import do_178c_checks
        extra = do_178c_checks(ir, output.model_c, level)
        compliance_result["do_178c_specific"] = extra
        if extra.get("recommendations"):
            recommendations.extend(extra["recommendations"])
    elif standard == "ISO 26262":
        from timber.accel.safety.certification.iso_26262 import iso_26262_checks
        extra = iso_26262_checks(ir, output.model_c, level)
        compliance_result["iso_26262_specific"] = extra
        if extra.get("recommendations"):
            recommendations.extend(extra["recommendations"])
    elif standard == "IEC 62304":
        from timber.accel.safety.certification.iec_62304 import iec_62304_checks
        extra = iec_62304_checks(ir, output.model_c, level)
        compliance_result["iec_62304_specific"] = extra
        if extra.get("recommendations"):
            recommendations.extend(extra["recommendations"])

    return CertificationReport(
        standard=standard,
        level=level,
        model_summary=ir.summary(),
        compliance_result=compliance_result,
        wcet_result=wcet_result,
        recommendations=recommendations,
    )
