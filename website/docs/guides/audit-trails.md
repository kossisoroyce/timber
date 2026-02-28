---
sidebar_position: 9
title: Audit Trails
---

# Audit Trails

Every Timber compilation produces a deterministic JSON audit report for regulatory compliance.

## What's in the Report

```json
{
  "timber_version": "0.1.0",
  "timestamp": "2026-02-28T00:00:00Z",
  "input_hash": "sha256:abc123def456...",
  "model_summary": {
    "n_trees": 50,
    "n_features": 30,
    "n_outputs": 1,
    "objective": "binary:logistic",
    "framework": "xgboost"
  },
  "passes": [
    {
      "name": "dead_leaf_elimination",
      "changed": true,
      "nodes_before": 1550,
      "nodes_after": 1523,
      "duration_ms": 1.2
    },
    {
      "name": "constant_feature_detection",
      "changed": false,
      "duration_ms": 0.8
    }
  ],
  "output_files": {
    "model.c": "sha256:def456...",
    "model.h": "sha256:ghi789...",
    "model_data.c": "sha256:jkl012..."
  },
  "target": {
    "arch": "x86_64",
    "format": "c_source"
  }
}
```

## Key Fields

| Field | Purpose |
|-------|---------|
| `input_hash` | SHA-256 of the original model file — proves which artifact was compiled |
| `passes` | Complete log of every optimization pass with timing |
| `output_files` | SHA-256 of every generated file — proves the output is deterministic |
| `timber_version` | Exact compiler version for reproducibility |

## Accessing the Report

After `timber load`:

```bash
cat ~/.timber/models/my-model/audit_report.json | python -m json.tool
```

Or programmatically:

```python
import json
from timber.store import ModelStore

store = ModelStore()
model_dir = store.get_model_dir("my-model")
with open(model_dir / "audit_report.json") as f:
    report = json.load(f)
```

## Regulatory Use Cases

| Standard | Industry | How Timber Helps |
|----------|----------|-----------------|
| SOX | Finance | Proves model version and compilation are traceable |
| MiFID II | Finance | Complete audit trail for algorithmic trading models |
| FDA / IEC 62304 | Medical | Software change documentation for medical device models |
| ISO 26262 | Automotive | Traceability for safety-critical ML components |
| DO-178C | Avionics | Verifiable compilation pipeline for flight software |

## Determinism

The audit report is deterministic: compiling the same model with the same Timber version always produces the same output hashes. This means you can verify that a deployed artifact matches a known-good compilation.
