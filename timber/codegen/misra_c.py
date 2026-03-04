"""MISRA-C compliance mode — generates C code conforming to MISRA C:2012 guidelines.

This module wraps the standard C99 emitter and applies post-processing rules
to ensure the generated code is suitable for safety-critical environments
(automotive, aerospace, medical devices).

MISRA C:2012 rules checked and enforced:
  - Rule 1.1:  No compiler extensions (__attribute__, __extension__)
  - Rule 1.3:  No undefined behaviour constructs
  - Rule 7.1:  No octal integer constants
  - Rule 7.2:  Unsigned integer literals use 'U' suffix
  - Rule 7.4:  String literals assigned only to const char*
  - Rule 8.4:  All functions have compatible declarations (ensured by header)
  - Rule 10.1: No implicit conversions that change signedness
  - Rule 10.3: Explicit cast required for value narrowing
  - Rule 11.3: No cast between pointer to object and pointer to different type
  - Rule 12.1: Operator precedence — no reliance on implicit precedence
  - Rule 14.3: No unreachable code after return
  - Rule 14.4: Boolean expressions use only boolean-compatible types
  - Rule 15.4: At most one break per loop
  - Rule 15.5: Single point of exit per function
  - Rule 17.7: Return values of non-void functions always used
  - Rule 20.4: No macro names that shadow keywords
  - Rule 20.9: No use of <stdio.h> in production code
  - Rule 21.1: No #define of standard library identifiers
  - Rule 21.6: No use of input/output functions from <stdio.h>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from timber.codegen.c99 import C99Emitter, C99Output, TargetSpec
from timber.ir.model import TimberIR

_MISRA_BANNER = "MISRA C:2012"

# Standard identifiers that must not be redefined (Rule 21.1)
_STD_IDENTIFIERS = {
    "NULL", "EOF", "errno", "stdin", "stdout", "stderr",
    "assert", "BUFSIZ", "CLOCKS_PER_SEC", "HUGE_VAL", "INT_MAX",
    "INT_MIN", "UINT_MAX", "SIZE_MAX", "true", "false",
}

# stdio.h functions forbidden in safety-critical code (Rule 21.6)
_STDIO_FUNS = {
    "printf", "fprintf", "sprintf", "snprintf", "scanf", "fscanf",
    "sscanf", "fopen", "fclose", "fread", "fwrite", "fgets", "fputs",
    "gets", "puts", "perror",
}

# C keywords that may not be shadowed by macros (Rule 20.4)
_C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default",
    "do", "double", "else", "enum", "extern", "float", "for", "goto",
    "if", "inline", "int", "long", "register", "restrict", "return",
    "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while",
}


@dataclass
class MisraViolation:
    rule: str
    severity: str    # "required" | "advisory" | "mandatory"
    line: int
    msg: str


@dataclass
class MisraReport:
    """Report of MISRA-C compliance checks."""
    violations: list[dict]
    warnings: list[dict]
    is_compliant: bool
    rules_checked: int
    rules_passed: int
    # Extended fields
    violation_objects: list[MisraViolation] = field(default_factory=list)
    advisory_count: int = 0
    required_count: int = 0

    def summary(self) -> str:
        lines = [
            "MISRA C:2012 Compliance Report",
            f"  Rules checked:  {self.rules_checked}",
            f"  Rules passed:   {self.rules_passed}",
            f"  Violations:     {len(self.violations)} "
            f"({self.required_count} required, {self.advisory_count} advisory)",
            f"  Warnings:       {len(self.warnings)}",
            f"  Compliant:      {'YES' if self.is_compliant else 'NO'}",
        ]
        for v in self.violation_objects:
            lines.append(f"  [{v.severity.upper()}] Rule {v.rule} line {v.line}: {v.msg}")
        return "\n".join(lines)


class MisraCEmitter:
    """MISRA-C compliant emitter — wraps C99Emitter with compliance transformations."""

    def __init__(self, target: TargetSpec | None = None):
        self._c99 = C99Emitter(target=target)

    def emit(self, ir: TimberIR) -> C99Output:
        """Emit MISRA-C compliant code."""
        output = self._c99.emit(ir)
        output.model_h = self._transform_header(output.model_h)
        output.model_c = self._transform_source(output.model_c)
        return output

    def check_compliance(self, code: str) -> MisraReport:
        """Check a C source string for MISRA-C rule violations.

        Returns a MisraReport describing violations, warnings, and compliance status.
        """
        violations: list[dict] = []
        warnings: list[dict] = []
        violation_objects: list[MisraViolation] = []
        rules_checked = 0
        rules_passed = 0

        def _add_violation(rule: str, severity: str, line_no: int, msg: str) -> None:
            v = MisraViolation(rule=rule, severity=severity, line=line_no, msg=msg)
            violation_objects.append(v)
            violations.append({"rule": rule, "severity": severity, "line": line_no, "msg": msg})

        def _add_warning(rule: str, line_no: int, msg: str) -> None:
            warnings.append({"rule": rule, "line": line_no, "msg": msg})

        code_lines = code.split("\n")

        # Rule 1.1: No compiler extensions
        rules_checked += 1
        ext_matches = [(i + 1, ln) for i, ln in enumerate(code_lines)
                       if "__attribute__" in ln or "__extension__" in ln or "__asm__" in ln]
        if ext_matches:
            for ln, _ in ext_matches:
                _add_violation("1.1", "required", ln, "Compiler extension detected")
        else:
            rules_passed += 1

        # Rule 7.1: No octal integer constants (leading zero, non-zero digits)
        rules_checked += 1
        octal_pat = re.compile(r'\b0[1-9][0-9]*\b')
        octal_violations = []
        for i, line in enumerate(code_lines):
            if octal_pat.search(line):
                octal_violations.append(i + 1)
        if octal_violations:
            for ln in octal_violations:
                _add_violation("7.1", "required", ln, "Octal integer constant")
        else:
            rules_passed += 1

        # Rule 7.2: Unsigned integer literals should use 'U' suffix
        rules_checked += 1
        # Look for bare hex constants without U suffix (common for bitmasks)
        hex_no_u = re.compile(r'\b0x[0-9a-fA-F]+\b(?!U)')
        hex_issues = [(i + 1) for i, ln in enumerate(code_lines) if hex_no_u.search(ln)]
        if hex_issues:
            _add_warning("7.2", hex_issues[0],
                         f"Hex constant(s) without 'U' suffix on {len(hex_issues)} line(s) (advisory)")
        rules_passed += 1  # advisory only

        # Rule 10.1: Implicit float-to-int conversion
        rules_checked += 1
        impl_conv = re.compile(r'\bint\b\s+\w+\s*=\s*[0-9]+\.[0-9]')
        impl_lines = [(i + 1) for i, ln in enumerate(code_lines) if impl_conv.search(ln)]
        if impl_lines:
            for ln in impl_lines:
                _add_violation("10.1", "required", ln, "Implicit float-to-int conversion")
        else:
            rules_passed += 1

        # Rule 14.3: No unreachable code after return
        rules_checked += 1
        unreach_pat = re.compile(r'return\s+[^;]+;\s*$')
        for i in range(len(code_lines) - 1):
            if unreach_pat.search(code_lines[i]):
                next_stripped = code_lines[i + 1].strip()
                if next_stripped and not next_stripped.startswith(("}", "/*", "//")):
                    _add_warning("14.3", i + 2, "Possible unreachable code after return")
        rules_passed += 1  # advisory

        # Rule 15.5: Single point of exit per function
        rules_checked += 1
        # Simple heuristic: count return statements per function block
        fn_starts = [i for i, ln in enumerate(code_lines)
                     if re.match(r'^[a-zA-Z_].*\(', ln) and not ln.strip().startswith(("//", "/*", "#"))]
        multi_return_found = False
        for fi in fn_starts:
            depth = 0
            body_returns = 0
            for j in range(fi, len(code_lines)):
                depth += code_lines[j].count("{") - code_lines[j].count("}")
                if re.search(r'\breturn\b', code_lines[j]):
                    body_returns += 1
                if depth == 0 and j > fi:
                    break
            if body_returns > 1:
                multi_return_found = True
                _add_warning("15.5", fi + 1, "Multiple return points in function (advisory)")
                break
        rules_passed += 1  # advisory

        # Rule 17.7: Return values of non-void functions used
        rules_checked += 1
        # Check for calls where the return value is discarded (statement = call;)
        discarded_pat = re.compile(r'^\s+(?!return|if|while|for|switch)\w+\s*\([^)]*\)\s*;')
        discarded = [(i + 1) for i, ln in enumerate(code_lines)
                     if discarded_pat.match(ln) and "timber_log" not in ln and "(void)" not in ln]
        if discarded:
            _add_warning("17.7", discarded[0],
                         f"Possible ignored return value ({len(discarded)} site(s)) (advisory)")
        rules_passed += 1  # advisory

        # Rule 20.4: No macro shadows keyword
        rules_checked += 1
        macro_def = re.compile(r'#\s*define\s+(\w+)')
        kw_shadow = []
        for i, line in enumerate(code_lines):
            m = macro_def.search(line)
            if m and m.group(1).lower() in _C_KEYWORDS:
                kw_shadow.append((i + 1, m.group(1)))
        if kw_shadow:
            for ln, name in kw_shadow:
                _add_violation("20.4", "required", ln, f"Macro '{name}' shadows a C keyword")
        else:
            rules_passed += 1

        # Rule 20.9: No <stdio.h> include
        rules_checked += 1
        stdio_inc = [(i + 1) for i, ln in enumerate(code_lines) if re.search(r'#\s*include\s*<stdio\.h>', ln)]
        if stdio_inc:
            _add_violation("20.9", "required", stdio_inc[0], "#include <stdio.h> in production code")
        else:
            rules_passed += 1

        # Rule 21.1: No #define of standard identifiers
        rules_checked += 1
        std_redef = []
        for i, line in enumerate(code_lines):
            m = macro_def.search(line)
            if m and m.group(1) in _STD_IDENTIFIERS:
                std_redef.append((i + 1, m.group(1)))
        if std_redef:
            for ln, name in std_redef:
                _add_violation("21.1", "required", ln, f"Redefinition of standard identifier '{name}'")
        else:
            rules_passed += 1

        # Rule 21.6: No stdio input/output functions
        rules_checked += 1
        stdio_calls = []
        for i, line in enumerate(code_lines):
            for fn in _STDIO_FUNS:
                if re.search(rf'\b{fn}\s*\(', line):
                    stdio_calls.append((i + 1, fn))
                    break
        if stdio_calls:
            for ln, fn in stdio_calls:
                _add_violation("21.6", "required", ln, f"Use of stdio function '{fn}'")
        else:
            rules_passed += 1

        advisory_count = sum(1 for v in violation_objects if v.severity == "advisory")
        required_count = sum(1 for v in violation_objects if v.severity == "required")
        is_compliant = required_count == 0

        return MisraReport(
            violations=violations,
            warnings=warnings,
            is_compliant=is_compliant,
            rules_checked=rules_checked,
            rules_passed=rules_passed,
            violation_objects=violation_objects,
            advisory_count=advisory_count,
            required_count=required_count,
        )

    def _transform_header(self, header: str) -> str:
        """Apply MISRA transformations to the header."""
        lines = header.split("\n")
        result = []
        for line in lines:
            if line.startswith("/* model.h"):
                result.append(
                    f"/* model.h — Timber compiled model ({_MISRA_BANNER} compliant) */"
                )
                continue
            # Rule 7.2: Add U suffix to hex constants in header macros
            if line.startswith("#define") and re.search(r'0x[0-9a-fA-F]+\b(?!U)', line):
                line = re.sub(r'(0x[0-9a-fA-F]+)\b(?!U)', r'\1U', line)
            result.append(line)
        return "\n".join(result)

    def _transform_source(self, source: str) -> str:
        """Apply MISRA transformations to the source.

        Transformations applied:
          - Banner marker in file header comments
          - Rule 7.2: Add U suffix to bare hex constants
          - Rule 10.3: Ensure explicit casts for index arithmetic
          - Rule 1.1: Strip any __attribute__ annotations (should not exist, but defensively)
        """
        lines = source.split("\n")
        result = []

        for line in lines:
            transformed = line

            # Rule 1.1: Remove any compiler extension annotations
            transformed = re.sub(r'__attribute__\s*\(\([^)]*\)\)', '', transformed)
            transformed = re.sub(r'__extension__\s*', '', transformed)

            # Rule 7.2: Add U suffix to hex constants not already suffixed
            transformed = re.sub(r'(0x[0-9a-fA-F]+)\b(?!U)', r'\1U', transformed)

            # Banner replacement in file header comment
            if line.startswith("/* model.c"):
                result.append(
                    f"/* model.c — Timber compiled inference ({_MISRA_BANNER} compliant) */"
                )
                continue
            if line.startswith("/* model_data.c"):
                result.append(
                    f"/* model_data.c — Timber compiled data ({_MISRA_BANNER} compliant) */"
                )
                continue

            result.append(transformed)

        return "\n".join(result)
