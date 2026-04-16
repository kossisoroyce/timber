"""IEC 62304 medical device software checker.

Extends the generic profile checker with medical-device-specific
requirements including software safety classification, unit verification,
and risk-based testing requirements.
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
    "a substitute for formal IEC 62304 compliance assessment. Results must be "
    "independently verified by a qualified regulatory affairs specialist before "
    "use in any medical device software."
)

# ---------------------------------------------------------------------------
# Software safety class definitions
# ---------------------------------------------------------------------------

SAFETY_CLASS_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "A": {
        "description": "No injury or damage to health is possible",
        "require_unit_testing": False,
        "require_integration_testing": False,
        "require_static_analysis": False,
        "require_risk_analysis": True,
        "require_traceability": False,
        "documentation_level": "basic",
    },
    "B": {
        "description": "Non-serious injury is possible",
        "require_unit_testing": True,
        "require_integration_testing": True,
        "require_static_analysis": True,
        "require_risk_analysis": True,
        "require_traceability": True,
        "documentation_level": "detailed",
    },
    "C": {
        "description": "Death or serious injury is possible",
        "require_unit_testing": True,
        "require_integration_testing": True,
        "require_static_analysis": True,
        "require_risk_analysis": True,
        "require_traceability": True,
        "documentation_level": "comprehensive",
    },
}

_CLASS_ALIASES: dict[str, str] = {
    "A": "A", "B": "B", "C": "C",
    "CLASS A": "A", "CLASS B": "B", "CLASS C": "C",
    "Class A": "A", "Class B": "B", "Class C": "C",
}

_FUNC_RE = re.compile(r"^[a-zA-Z_]\w*\s+\*?\s*([a-zA-Z_]\w*)\s*\(", re.MULTILINE)


def _resolve_class(level: str) -> str:
    normalized = level.strip()
    return _CLASS_ALIASES.get(normalized, _CLASS_ALIASES.get(normalized.upper(), "C"))


# ---------------------------------------------------------------------------
# Unit verification analysis
# ---------------------------------------------------------------------------

def analyze_unit_verification(code: str) -> dict[str, Any]:
    """Assess how well the generated code supports unit-level verification.

    Checks for testable function boundaries, input validation,
    deterministic behaviour, and documentation annotations.
    """
    functions = _FUNC_RE.findall(code)
    lines = code.splitlines()

    input_validations = sum(
        1 for ln in lines if re.search(r"if\s*\(.*[<>!=]=?\s*", ln)
    )
    doc_comments = sum(
        1 for ln in lines if ln.strip().startswith("/**") or ln.strip().startswith("///")
    )
    static_functions = len(re.findall(r"\bstatic\s+\w+\s+\w+\s*\(", code))

    testability_score = 0.0
    if functions:
        # Heuristic score 0-100
        validation_ratio = min(1.0, input_validations / len(functions))
        doc_ratio = min(1.0, doc_comments / len(functions))
        encapsulation_ratio = static_functions / len(functions) if functions else 0
        testability_score = round(
            (validation_ratio * 40 + doc_ratio * 30 + encapsulation_ratio * 30), 1
        )

    return {
        "total_functions": len(functions),
        "functions": functions,
        "input_validations": input_validations,
        "documented_functions": doc_comments,
        "static_functions": static_functions,
        "testability_score": testability_score,
    }


# ---------------------------------------------------------------------------
# Risk-based testing requirements
# ---------------------------------------------------------------------------

def assess_risk_based_testing(ir: TimberIR, code: str, safety_class: str) -> dict[str, Any]:
    """Determine testing requirements based on software safety class and
    code characteristics.
    """
    class_reqs = SAFETY_CLASS_REQUIREMENTS.get(safety_class, SAFETY_CLASS_REQUIREMENTS["C"])
    functions = _FUNC_RE.findall(code)
    ir_summary = ir.summary()

    # Classify functions by risk
    high_risk: list[str] = []
    medium_risk: list[str] = []
    low_risk: list[str] = []

    for fname in functions:
        # Heuristic: functions touching outputs or critical paths are higher risk
        if any(kw in fname.lower() for kw in ("predict", "infer", "output", "classify")):
            high_risk.append(fname)
        elif any(kw in fname.lower() for kw in ("init", "setup", "config", "param")):
            medium_risk.append(fname)
        else:
            low_risk.append(fname)

    testing_requirements: list[str] = []
    if class_reqs["require_unit_testing"]:
        testing_requirements.append("Unit testing required for all software units.")
    if class_reqs["require_integration_testing"]:
        testing_requirements.append("Integration testing required for all interfaces.")
    if safety_class == "C":
        testing_requirements.append("Regression testing required after every change.")
        testing_requirements.append("Boundary value analysis required for all inputs.")
        testing_requirements.append("Equivalence class testing required.")

    return {
        "safety_class": safety_class,
        "high_risk_functions": high_risk,
        "medium_risk_functions": medium_risk,
        "low_risk_functions": low_risk,
        "testing_requirements": testing_requirements,
        "documentation_level": class_reqs["documentation_level"],
    }


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------

def iec_62304_checks(
    ir: TimberIR,
    code: str,
    level: str = "C",
) -> dict[str, Any]:
    """Run all IEC 62304 specific checks.

    Parameters
    ----------
    ir:
        The Timber IR model.
    code:
        Generated C source code.
    level:
        Software safety class (A, B, or C).

    Returns
    -------
    dict
        Contains ``safety_class_info``, ``unit_verification``,
        ``risk_based_testing``, and ``recommendations``.
    """
    warnings.warn(DISCLAIMER, stacklevel=2)

    safety_class = _resolve_class(level)
    class_info = SAFETY_CLASS_REQUIREMENTS.get(safety_class, SAFETY_CLASS_REQUIREMENTS["C"])

    unit_ver = analyze_unit_verification(code)
    risk_testing = assess_risk_based_testing(ir, code, safety_class)

    recommendations: list[str] = []

    if unit_ver["testability_score"] < 50.0:
        recommendations.append(
            f"IEC 62304 Class {safety_class}: testability score "
            f"{unit_ver['testability_score']}% is low — add input validation "
            f"and documentation to generated code."
        )

    if class_info["require_traceability"]:
        recommendations.append(
            f"IEC 62304 Class {safety_class}: full requirements traceability "
            f"is required — map each software unit to a requirement."
        )

    if class_info["require_risk_analysis"]:
        if risk_testing["high_risk_functions"]:
            recommendations.append(
                f"IEC 62304: {len(risk_testing['high_risk_functions'])} high-risk "
                f"functions identified — ensure hazard analysis covers these."
            )

    if safety_class == "C" and unit_ver["documented_functions"] < unit_ver["total_functions"]:
        recommendations.append(
            "IEC 62304 Class C: all software units must be fully documented."
        )

    return {
        "disclaimer": DISCLAIMER,
        "safety_class": safety_class,
        "safety_class_info": class_info,
        "unit_verification": unit_ver,
        "risk_based_testing": risk_testing,
        "recommendations": recommendations,
    }
