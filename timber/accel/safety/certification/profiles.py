"""Compliance profile loader and generic rule checker.

Loads TOML compliance profiles from the ``compliance_profiles/`` directory and
validates generated C code against the rule-set defined by each profile.

.. note:: **Limitations** -- All rule checks in this module rely on regex-based
   heuristic pattern matching against generated C source code.  This approach
   can produce both false positives and false negatives.  It is NOT equivalent
   to the rigorous static analysis performed by formal certification tools
   (e.g., LDRA, Polyspace, Astree, VectorCAST, Coverity).  Results should be
   treated as advisory only and must be independently verified by qualified
   assessors before use in any safety-critical system.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

ANALYSIS_LIMITATIONS = (
    "All rule checks in this module rely on regex-based heuristic pattern "
    "matching against generated C source code. This approach can produce both "
    "false positives and false negatives and is NOT equivalent to formal "
    "certification tooling. Results are advisory only."
)

# ---------------------------------------------------------------------------
# Profile discovery
# ---------------------------------------------------------------------------

_PROFILES_DIR = Path(__file__).resolve().parents[2] / "compliance_profiles"

_profile_cache: dict[str, dict[str, Any]] = {}


def _profiles_dir() -> Path:
    """Return the directory that holds ``*.toml`` compliance profiles."""
    return _PROFILES_DIR


def list_profiles() -> list[str]:
    """Return available profile names (stem of each ``.toml`` file)."""
    return sorted(p.stem for p in _profiles_dir().glob("*.toml"))


def load_compliance_profile(name: str) -> dict[str, Any]:
    """Load and cache a compliance profile by name.

    Parameters
    ----------
    name:
        Profile name — corresponds to the TOML filename without extension.
        Example: ``"do_178c"``, ``"iso_26262"``, ``"iec_62304"``.

    Returns
    -------
    dict
        Parsed TOML content.
    """
    if name in _profile_cache:
        return _profile_cache[name]

    path = _profiles_dir() / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"Compliance profile '{name}' not found at {path}. "
            f"Available: {list_profiles()}"
        )

    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    _profile_cache[name] = data
    return data


# ---------------------------------------------------------------------------
# Individual rule checkers
# ---------------------------------------------------------------------------

_DYN_ALLOC_RE = re.compile(r"\b(malloc|calloc|realloc|free)\s*\(")
_PTR_ARITH_RE = re.compile(r"(\w+\s*\+\+|\w+\s*--|\w+\s*\+=|\w+\s*-=|\*\s*\(.*?\+\s*\d+\))")
_FLOAT_CMP_RE = re.compile(
    r"(?:float|double)\s+\w+.*?(?:==|!=)"
    r"|(?:==|!=)\s*[\d.]+[fF]?\b"
)
_LOOP_RE = re.compile(r"\b(for|while|do)\b")
_FUNC_RE = re.compile(r"^[a-zA-Z_]\w*\s+\*?\s*([a-zA-Z_]\w*)\s*\(", re.MULTILINE)
_RETURN_RE = re.compile(r"\breturn\b")


def _check_no_dynamic_allocation(code: str) -> list[str]:
    violations: list[str] = []
    for m in _DYN_ALLOC_RE.finditer(code):
        lineno = code[: m.start()].count("\n") + 1
        violations.append(f"Line {lineno}: dynamic allocation ({m.group(1)})")
    return violations


def _check_no_recursion(code: str) -> list[str]:
    violations: list[str] = []
    for m in _FUNC_RE.finditer(code):
        fname = m.group(1)
        # Search for a call to the same function inside its body
        body_start = code.index("{", m.end())
        depth = 1
        idx = body_start + 1
        while idx < len(code) and depth > 0:
            if code[idx] == "{":
                depth += 1
            elif code[idx] == "}":
                depth -= 1
            idx += 1
        body = code[body_start:idx]
        call_pat = re.compile(rf"\b{re.escape(fname)}\s*\(")
        if call_pat.search(body):
            lineno = code[: m.start()].count("\n") + 1
            violations.append(f"Line {lineno}: possible recursion in '{fname}'")
    return violations


def _check_no_unbounded_loops(code: str) -> list[str]:
    violations: list[str] = []
    for m in _LOOP_RE.finditer(code):
        lineno = code[: m.start()].count("\n") + 1
        keyword = m.group(1)
        # ``while (1)`` / ``while (true)`` / ``for (;;)``
        rest = code[m.end(): m.end() + 80]
        if keyword == "while" and re.match(r"\s*\(\s*(1|true)\s*\)", rest):
            violations.append(f"Line {lineno}: unbounded loop (while true)")
        elif keyword == "for" and re.match(r"\s*\(\s*;\s*;\s*\)", rest):
            violations.append(f"Line {lineno}: unbounded loop (for ;;)")
    return violations


def _check_single_entry_exit(code: str) -> list[str]:
    violations: list[str] = []
    for m in _FUNC_RE.finditer(code):
        fname = m.group(1)
        try:
            body_start = code.index("{", m.end())
        except ValueError:
            continue
        depth = 1
        idx = body_start + 1
        while idx < len(code) and depth > 0:
            if code[idx] == "{":
                depth += 1
            elif code[idx] == "}":
                depth -= 1
            idx += 1
        body = code[body_start:idx]
        returns = _RETURN_RE.findall(body)
        if len(returns) > 1:
            lineno = code[: m.start()].count("\n") + 1
            violations.append(
                f"Line {lineno}: function '{fname}' has {len(returns)} return statements"
            )
    return violations


def _check_no_pointer_arithmetic(code: str) -> list[str]:
    violations: list[str] = []
    for i, line in enumerate(code.splitlines(), 1):
        if _PTR_ARITH_RE.search(line) and ("*" in line or "ptr" in line.lower()):
            violations.append(f"Line {i}: possible pointer arithmetic")
    return violations


def _check_no_floating_point_comparison(code: str) -> list[str]:
    violations: list[str] = []
    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        if _FLOAT_CMP_RE.search(line):
            violations.append(f"Line {i}: floating-point equality comparison")
    return violations


def _cyclomatic_complexity(body: str) -> int:
    """Approximate McCabe cyclomatic complexity of a function body."""
    cc = 1
    keywords = re.findall(r"\b(else\s+if|elif|if|for|while|case|catch)\b|\?\s*:", body)
    cc += len(keywords)
    return cc


def _check_max_function_complexity(code: str, limit: int) -> list[str]:
    violations: list[str] = []
    for m in _FUNC_RE.finditer(code):
        fname = m.group(1)
        try:
            body_start = code.index("{", m.end())
        except ValueError:
            continue
        depth = 1
        idx = body_start + 1
        while idx < len(code) and depth > 0:
            if code[idx] == "{":
                depth += 1
            elif code[idx] == "}":
                depth -= 1
            idx += 1
        body = code[body_start:idx]
        cc = _cyclomatic_complexity(body)
        if cc > limit:
            lineno = code[: m.start()].count("\n") + 1
            violations.append(
                f"Line {lineno}: function '{fname}' complexity {cc} exceeds limit {limit}"
            )
    return violations


def _check_max_nesting_depth(code: str, limit: int) -> list[str]:
    violations: list[str] = []
    max_depth = 0
    current_depth = 0
    worst_line = 0
    for i, ch in enumerate(code):
        if ch == "{":
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
                worst_line = code[:i].count("\n") + 1
        elif ch == "}":
            current_depth = max(0, current_depth - 1)
    if max_depth > limit:
        violations.append(
            f"Line {worst_line}: max nesting depth {max_depth} exceeds limit {limit}"
        )
    return violations


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_RULE_DISPATCH: dict[str, Any] = {
    "no_dynamic_allocation": _check_no_dynamic_allocation,
    "no_recursion": _check_no_recursion,
    "no_unbounded_loops": _check_no_unbounded_loops,
    "single_entry_exit": _check_single_entry_exit,
    "no_pointer_arithmetic": _check_no_pointer_arithmetic,
    "no_floating_point_comparison": _check_no_floating_point_comparison,
}

_PARAMETERISED_RULES: dict[str, Any] = {
    "max_function_complexity": _check_max_function_complexity,
    "max_nesting_depth": _check_max_nesting_depth,
}


def check_compliance(code: str, profile_name: str) -> dict[str, Any]:
    """Validate *code* against every rule in the named compliance profile.

    Parameters
    ----------
    code:
        Generated C source code to analyse.
    profile_name:
        Name of the compliance profile to load (e.g. ``"do_178c"``).

    Returns
    -------
    dict
        ``compliant`` (bool), ``violations`` (list[str]),
        ``warnings`` (list[str]), ``rules_checked`` (int).
    """
    profile = load_compliance_profile(profile_name)
    rules: dict[str, Any] = profile.get("rules", {})

    all_violations: list[str] = []
    warnings: list[str] = []
    rules_checked = 0

    for rule_name, rule_value in rules.items():
        # Boolean rules that must be True to enable the check
        if rule_name in _RULE_DISPATCH:
            if rule_value:
                rules_checked += 1
                hits = _RULE_DISPATCH[rule_name](code)
                all_violations.extend(hits)
            continue

        # Parameterised rules (integer limits)
        if rule_name in _PARAMETERISED_RULES:
            if isinstance(rule_value, (int, float)) and rule_value > 0:
                rules_checked += 1
                hits = _PARAMETERISED_RULES[rule_name](code, int(rule_value))
                all_violations.extend(hits)
            continue

        # Meta-rules (require_*) are informational warnings when not met
        if rule_name.startswith("require_"):
            if rule_value:
                warnings.append(f"Profile requires '{rule_name}' — verify externally.")
            continue

        # Unknown rules
        warnings.append(f"Unknown rule '{rule_name}' in profile '{profile_name}'.")

    return {
        "compliant": len(all_violations) == 0,
        "violations": all_violations,
        "warnings": warnings,
        "rules_checked": rules_checked,
    }
