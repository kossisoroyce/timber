---
sidebar_position: 6
title: MISRA-C Compliance
---

# MISRA-C Compliance

For safety-critical deployments (automotive ECUs, medical devices, avionics), Timber provides a MISRA-C:2012 compliant code emitter.

## Generating MISRA-C Code

```python
from timber.codegen.misra_c import MisraCEmitter, check_misra_compliance
from timber.frontends.auto_detect import parse_model

ir = parse_model("model.json")
emitter = MisraCEmitter(ir)
files = emitter.emit()

# Write files
for filename, content in files.items():
    with open(filename, "w") as f:
        f.write(content)
```

## Compliance Checking

```python
report = check_misra_compliance(files)
print(f"Violations: {report.violations}")
print(f"Warnings: {report.warnings}")
print(f"Compliant: {report.is_compliant}")

# Detailed rule results
for rule in report.rules:
    print(f"  Rule {rule.id}: {'PASS' if rule.passed else 'FAIL'} — {rule.description}")
```

## Rules Checked

| Rule | Description | Category |
|------|-------------|----------|
| 1.1 | No compiler extensions | Required |
| 10.1 | Unsigned integer suffix enforcement | Required |
| 14.4 | Boolean conditions | Required |
| 15.7 | Else after if-else-if | Required |
| 17.7 | Return value usage | Required |
| 20.7 | Macro expansion | Required |
| 21.1 | No redefinition of standard identifiers | Required |

## Transformations Applied

The MISRA-C emitter wraps the standard C99 emitter and applies:

1. **Unsigned suffix enforcement** — all unsigned literals get `U` suffix
2. **No compiler-specific extensions** — pure ISO C
3. **Standard identifier protection** — no redefinition of reserved names
4. **Compliance markers** — generated headers include MISRA compliance declaration

## Use Cases

- **Automotive** — ADAS systems, engine control, autonomous driving
- **Medical devices** — patient monitoring, diagnostic equipment
- **Avionics** — flight control systems, navigation
- **Industrial** — PLC controllers, safety-instrumented systems

## Integration with Certification

The generated code, combined with the [audit trail](/docs/guides/audit-trails), provides documentation suitable for:

- ISO 26262 (automotive)
- IEC 62304 (medical software)
- DO-178C (avionics)
