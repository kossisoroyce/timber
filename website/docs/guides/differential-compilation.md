---
sidebar_position: 7
title: Differential Compilation
---

# Differential Compilation

When models are retrained frequently, Timber can diff two model versions and identify exactly what changed.

## Computing a Diff

```python
from timber.optimizer.diff_compile import diff_models
from timber.frontends.auto_detect import parse_model

old_ir = parse_model("model_v1.json")
new_ir = parse_model("model_v2.json")

diff = diff_models(old_ir, new_ir)

print(f"Added trees:     {len(diff.added)}")
print(f"Removed trees:   {len(diff.removed)}")
print(f"Modified trees:  {len(diff.modified)}")
print(f"Unchanged trees: {len(diff.unchanged)}")
```

## How Tree Hashing Works

Each tree is assigned a content hash based on:

- Node structure (parent/child indices)
- Feature indices at each split
- Threshold values
- Leaf values
- Default-left flags

Two trees with identical structure and values will have the same hash, even if they appear at different positions in the ensemble.

## Use Cases

### Model Monitoring

```python
# Scheduled job: check how much the model changed after retraining
diff = diff_models(production_ir, retrained_ir)

change_ratio = len(diff.modified) / (len(diff.unchanged) + len(diff.modified))
if change_ratio > 0.5:
    alert("Model drift detected: >50% of trees changed")
```

### Incremental Deployment

In pipelines where models are retrained hourly, typically only 5–20% of trees change. Differential compilation identifies exactly which trees need recompilation.

### Audit & Compliance

The diff output provides a precise record of what changed between model versions — useful for change management in regulated environments.

## API Reference

```python
@dataclass
class ModelDiff:
    added: list[str]      # Hashes of new trees
    removed: list[str]    # Hashes of deleted trees
    modified: list[str]   # Trees with same position but different hash
    unchanged: list[str]  # Trees identical in both versions
```
