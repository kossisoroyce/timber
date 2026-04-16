"""ISO 26262 automotive functional safety checker.

Extends the generic profile checker with automotive-specific requirements
including ASIL-level mapping, freedom from interference analysis,
diagnostic coverage estimation, and safe-state analysis.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

from timber.ir.model import TimberIR

# ---------------------------------------------------------------------------
# Advisory disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "ADVISORY ONLY \u2014 This analysis uses heuristic pattern matching and is NOT "
    "a substitute for formal ISO 26262 certification tooling. Results must be "
    "independently verified by a qualified functional safety assessor before "
    "use in any automotive safety-related system."
)

# ---------------------------------------------------------------------------
# ASIL level definitions (A = lowest, D = highest)
# ---------------------------------------------------------------------------

ASIL_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "ASIL-A": {
        "description": "Low severity / low probability",
        "structural_coverage": "statement",
        "require_back_to_back_testing": False,
        "require_static_analysis": True,
        "diagnostic_coverage_min": 60.0,
        "fault_tolerance_time_ms": 1000,
    },
    "ASIL-B": {
        "description": "Medium severity / medium probability",
        "structural_coverage": "branch",
        "require_back_to_back_testing": False,
        "require_static_analysis": True,
        "diagnostic_coverage_min": 90.0,
        "fault_tolerance_time_ms": 500,
    },
    "ASIL-C": {
        "description": "High severity / medium probability",
        "structural_coverage": "MC/DC",
        "require_back_to_back_testing": True,
        "require_static_analysis": True,
        "diagnostic_coverage_min": 97.0,
        "fault_tolerance_time_ms": 200,
    },
    "ASIL-D": {
        "description": "Highest severity / highest probability",
        "structural_coverage": "MC/DC",
        "require_back_to_back_testing": True,
        "require_static_analysis": True,
        "diagnostic_coverage_min": 99.0,
        "fault_tolerance_time_ms": 100,
    },
}

# Map short-hand levels to full names
_LEVEL_ALIASES: dict[str, str] = {
    "A": "ASIL-A", "B": "ASIL-B", "C": "ASIL-C", "D": "ASIL-D",
    "ASIL-A": "ASIL-A", "ASIL-B": "ASIL-B", "ASIL-C": "ASIL-C", "ASIL-D": "ASIL-D",
}

_FUNC_RE = re.compile(r"^[a-zA-Z_]\w*\s+\*?\s*([a-zA-Z_]\w*)\s*\(", re.MULTILINE)
_GLOBAL_RE = re.compile(r"^(?:static\s+)?(?:volatile\s+)?(?:const\s+)?\w+\s+(\w+)\s*[=;]", re.MULTILINE)


def _resolve_level(level: str) -> str:
    return _LEVEL_ALIASES.get(level.upper().strip(), "ASIL-D")


# ---------------------------------------------------------------------------
# Freedom from interference
# ---------------------------------------------------------------------------

def analyze_freedom_from_interference(code: str) -> dict[str, Any]:
    """Analyse generated code for potential interference between components.

    Checks for shared global state, volatile accesses, and unprotected
    resources that could cause interference between software partitions.
    """
    globals_found = _GLOBAL_RE.findall(code)
    volatile_count = len(re.findall(r"\bvolatile\b", code))
    shared_warnings: list[str] = []

    for gname in globals_found:
        # Flag globals that are not declared static (visible across TUs)
        pattern = re.compile(rf"^(?!static)\w.*\b{re.escape(gname)}\b", re.MULTILINE)
        if pattern.search(code):
            shared_warnings.append(
                f"Global '{gname}' is not static — may interfere across partitions."
            )

    return {
        "global_variables": globals_found,
        "volatile_accesses": volatile_count,
        "shared_warnings": shared_warnings,
        "interference_risk": "high" if shared_warnings else "low",
    }


# ---------------------------------------------------------------------------
# Diagnostic coverage estimation
# ---------------------------------------------------------------------------

def estimate_diagnostic_coverage(code: str) -> dict[str, Any]:
    """Estimate diagnostic coverage based on defensive programming patterns.

    Looks for null-pointer guards, range checks, assertion macros, and
    error return paths.
    """
    lines = code.splitlines()
    total_functions = len(_FUNC_RE.findall(code))
    null_checks = sum(1 for ln in lines if re.search(r"if\s*\(.*==\s*NULL", ln))
    range_checks = sum(1 for ln in lines if re.search(r"if\s*\(.*[<>]=?\s*\d+", ln))
    assertions = sum(1 for ln in lines if re.search(r"\b(assert|ASSERT|CHECK)\b", ln))
    error_returns = sum(1 for ln in lines if re.search(r"return\s+(-1|NULL|ERR|false)", ln))

    diagnostic_mechanisms = null_checks + range_checks + assertions + error_returns
    # Simple heuristic: coverage % based on mechanisms per function
    if total_functions > 0:
        coverage_pct = min(100.0, (diagnostic_mechanisms / total_functions) * 25.0)
    else:
        coverage_pct = 0.0

    return {
        "null_checks": null_checks,
        "range_checks": range_checks,
        "assertions": assertions,
        "error_returns": error_returns,
        "total_mechanisms": diagnostic_mechanisms,
        "estimated_coverage_pct": round(coverage_pct, 1),
    }


# ---------------------------------------------------------------------------
# Safe state analysis
# ---------------------------------------------------------------------------

def analyze_safe_state(ir: TimberIR, code: str) -> dict[str, Any]:
    """Analyse whether the generated code can reach a safe state on failure.

    Checks for error handling paths, output clamping, and default/fallback
    values that would bring the system to a safe state.
    """
    has_output_clamp = bool(re.search(r"\b(fminf?|fmaxf?|clamp|CLAMP)\b", code))
    has_default_output = bool(re.search(r"(default_output|safe_value|fallback)", code, re.IGNORECASE))
    has_error_handler = bool(re.search(r"(error_handler|on_error|fail_safe)", code, re.IGNORECASE))
    has_watchdog = bool(re.search(r"(watchdog|wdt|WDT)", code))

    mechanisms: list[str] = []
    if has_output_clamp:
        mechanisms.append("output_clamping")
    if has_default_output:
        mechanisms.append("default_safe_output")
    if has_error_handler:
        mechanisms.append("error_handler")
    if has_watchdog:
        mechanisms.append("watchdog_support")

    return {
        "safe_state_mechanisms": mechanisms,
        "has_output_clamping": has_output_clamp,
        "has_default_output": has_default_output,
        "has_error_handler": has_error_handler,
        "has_watchdog_support": has_watchdog,
        "safe_state_achievable": len(mechanisms) >= 1,
    }


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------

def iso_26262_checks(
    ir: TimberIR,
    code: str,
    level: str = "ASIL-D",
) -> dict[str, Any]:
    """Run all ISO 26262 specific checks.

    Parameters
    ----------
    ir:
        The Timber IR model.
    code:
        Generated C source code.
    level:
        ASIL level (A-D or ASIL-A through ASIL-D).

    Returns
    -------
    dict
        Contains ``asil_info``, ``freedom_from_interference``,
        ``diagnostic_coverage``, ``safe_state``, and ``recommendations``.
    """
    warnings.warn(DISCLAIMER, stacklevel=2)

    asil = _resolve_level(level)
    asil_info = ASIL_REQUIREMENTS.get(asil, ASIL_REQUIREMENTS["ASIL-D"])

    interference = analyze_freedom_from_interference(code)
    diagnostics = estimate_diagnostic_coverage(code)
    safe_state = analyze_safe_state(ir, code)

    recommendations: list[str] = []

    if interference["interference_risk"] == "high":
        recommendations.append(
            "ISO 26262: high interference risk — add memory protection or "
            "partition shared state."
        )

    min_coverage = asil_info["diagnostic_coverage_min"]
    if diagnostics["estimated_coverage_pct"] < min_coverage:
        recommendations.append(
            f"ISO 26262 {asil}: diagnostic coverage {diagnostics['estimated_coverage_pct']}% "
            f"below required {min_coverage}% — add defensive checks."
        )

    if not safe_state["safe_state_achievable"]:
        recommendations.append(
            "ISO 26262: no safe-state mechanism detected — add output clamping or "
            "error handler."
        )

    if asil_info["require_back_to_back_testing"]:
        recommendations.append(
            f"ISO 26262 {asil}: back-to-back testing required between model and "
            f"generated code."
        )

    return {
        "disclaimer": DISCLAIMER,
        "asil": asil,
        "asil_info": asil_info,
        "freedom_from_interference": interference,
        "diagnostic_coverage": diagnostics,
        "safe_state": safe_state,
        "recommendations": recommendations,
    }
