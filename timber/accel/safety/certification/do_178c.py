"""DO-178C specific certification checks.

Extends the generic profile checker with avionics-specific requirements
including software level mapping, traceability matrix generation, MC/DC
coverage annotations, and structural coverage analysis.
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
    "a substitute for formal DO-178C certification tooling (e.g., LDRA, Polyspace, "
    "Astr\u00e9e, VectorCAST). Results must be independently verified by a qualified DER "
    "before use in any airborne system certification."
)

# ---------------------------------------------------------------------------
# Software level definitions (A = most stringent, E = no guidance)
# ---------------------------------------------------------------------------

LEVEL_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "A": {
        "description": "Catastrophic failure condition",
        "structural_coverage": "MC/DC",
        "require_independence": True,
        "require_formal_methods": True,
        "objectives": 71,
    },
    "B": {
        "description": "Hazardous/Severe-Major failure condition",
        "structural_coverage": "decision",
        "require_independence": True,
        "require_formal_methods": False,
        "objectives": 69,
    },
    "C": {
        "description": "Major failure condition",
        "structural_coverage": "statement",
        "require_independence": False,
        "require_formal_methods": False,
        "objectives": 62,
    },
    "D": {
        "description": "Minor failure condition",
        "structural_coverage": "none",
        "require_independence": False,
        "require_formal_methods": False,
        "objectives": 28,
    },
    "E": {
        "description": "No effect on aircraft safety",
        "structural_coverage": "none",
        "require_independence": False,
        "require_formal_methods": False,
        "objectives": 0,
    },
}

# ---------------------------------------------------------------------------
# Traceability helpers
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(r"^[a-zA-Z_]\w*\s+\*?\s*([a-zA-Z_]\w*)\s*\(", re.MULTILINE)
_DECISION_RE = re.compile(r"\b(if|else\s+if|switch|while|for)\b")
_CONDITION_RE = re.compile(r"(&&|\|\|)")


def _extract_functions(code: str) -> list[dict[str, Any]]:
    """Extract function names and line numbers from C source."""
    funcs: list[dict[str, Any]] = []
    for m in _FUNC_RE.finditer(code):
        lineno = code[: m.start()].count("\n") + 1
        funcs.append({"name": m.group(1), "line": lineno})
    return funcs


def generate_traceability_matrix(
    ir: TimberIR,
    code: str,
) -> dict[str, Any]:
    """Build a basic traceability matrix between IR layers and generated code.

    Maps each layer in the Timber IR to the C functions it produces.
    """
    ir_summary = ir.summary()
    layers = ir_summary.get("pipeline", [])
    functions = _extract_functions(code)

    matrix: list[dict[str, Any]] = []
    for idx, layer in enumerate(layers):
        layer_name = layer if isinstance(layer, str) else layer.get("op", f"layer_{idx}")
        # Heuristic: generated C functions are typically named after the layer
        linked_funcs = [
            f["name"] for f in functions
            if layer_name.lower().replace(" ", "_") in f["name"].lower()
        ]
        matrix.append({
            "requirement_id": f"SWR-{idx + 1:04d}",
            "ir_layer": layer_name,
            "generated_functions": linked_funcs,
            "verified": len(linked_funcs) > 0,
        })

    return {
        "entries": matrix,
        "total_requirements": len(matrix),
        "traced": sum(1 for e in matrix if e["verified"]),
    }


# ---------------------------------------------------------------------------
# MC/DC coverage requirements
# ---------------------------------------------------------------------------

def analyze_mcdc_requirements(code: str) -> dict[str, Any]:
    """Estimate MC/DC coverage requirements for the generated code.

    Counts decision points and boolean conditions to estimate the number
    of MC/DC test obligations.
    """
    decisions = 0
    conditions = 0

    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        decisions += len(_DECISION_RE.findall(line))
        conditions += len(_CONDITION_RE.findall(line))

    # Each condition within a decision requires independence pairs
    test_obligations = decisions + 2 * conditions

    return {
        "decision_points": decisions,
        "boolean_conditions": conditions,
        "estimated_test_obligations": test_obligations,
        "coverage_type": "MC/DC",
    }


# ---------------------------------------------------------------------------
# Structural coverage
# ---------------------------------------------------------------------------

def analyze_structural_coverage(code: str, level: str) -> dict[str, Any]:
    """Determine structural coverage requirements based on software level."""
    req = LEVEL_REQUIREMENTS.get(level, LEVEL_REQUIREMENTS["E"])
    functions = _extract_functions(code)

    return {
        "coverage_required": req["structural_coverage"],
        "functions_count": len(functions),
        "functions": [f["name"] for f in functions],
        "require_independence": req["require_independence"],
        "require_formal_methods": req["require_formal_methods"],
        "total_objectives": req["objectives"],
    }


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------

def do_178c_checks(
    ir: TimberIR,
    code: str,
    level: str = "A",
) -> dict[str, Any]:
    """Run all DO-178C specific checks.

    Parameters
    ----------
    ir:
        The Timber IR model.
    code:
        Generated C source code.
    level:
        DO-178C software level (A-E).

    Returns
    -------
    dict
        Contains ``level_info``, ``traceability``, ``mcdc``,
        ``structural_coverage``, and ``recommendations``.
    """
    warnings.warn(DISCLAIMER, stacklevel=2)

    level = level.upper() if level else "A"
    if level not in LEVEL_REQUIREMENTS:
        level = "A"

    level_info = LEVEL_REQUIREMENTS[level]
    traceability = generate_traceability_matrix(ir, code)
    mcdc = analyze_mcdc_requirements(code)
    structural = analyze_structural_coverage(code, level)

    recommendations: list[str] = []

    # Traceability gap warnings
    if traceability["traced"] < traceability["total_requirements"]:
        gap = traceability["total_requirements"] - traceability["traced"]
        recommendations.append(
            f"DO-178C: {gap} requirements lack traceability to generated code."
        )

    # MC/DC obligation warning for Level A
    if level == "A" and mcdc["estimated_test_obligations"] > 0:
        recommendations.append(
            f"DO-178C Level A: {mcdc['estimated_test_obligations']} MC/DC test "
            f"obligations identified — ensure test suite covers all."
        )

    if level_info["require_formal_methods"]:
        recommendations.append(
            "DO-178C Level A: formal methods supplement recommended."
        )

    return {
        "disclaimer": DISCLAIMER,
        "level": level,
        "level_info": level_info,
        "traceability": traceability,
        "mcdc": mcdc,
        "structural_coverage": structural,
        "recommendations": recommendations,
    }
