---
sidebar_position: 8
title: Ensemble Composition
---

# Ensemble Composition

Timber supports composing multiple models into voting or stacking ensembles as first-class IR stages.

## Voting Ensemble

Combine predictions from multiple models via weighted averaging:

```python
from timber.ir.ensemble_meta import VotingEnsembleStage
from timber.frontends.auto_detect import parse_model

model_a = parse_model("xgb_model.json")
model_b = parse_model("lgb_model.txt")

ensemble = VotingEnsembleStage(
    sub_models=[model_a, model_b],
    weights=[0.6, 0.4],
    voting="soft",  # "soft" = weighted average, "hard" = majority vote
)
```

## Stacking Ensemble

Use a meta-learner trained on base model outputs:

```python
from timber.ir.ensemble_meta import StackingEnsembleStage
from timber.frontends.auto_detect import parse_model

base_a = parse_model("xgb_model.json")
base_b = parse_model("lgb_model.txt")
meta = parse_model("meta_model.json")

ensemble = StackingEnsembleStage(
    base_models=[base_a, base_b],
    meta_model=meta,
    passthrough=True,  # Also pass original features to meta-learner
)
```

With `passthrough=True`, the meta-learner receives both the base model predictions and the original input features.

## Why Compose at the IR Level?

Because ensemble stages are first-class IR stages, the full optimization pipeline can operate on them. This means:

- Pipeline fusion can absorb scalers into sub-models
- Dead leaf elimination works on each sub-model
- The entire ensemble can be compiled to a single shared library
