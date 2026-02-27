"""Stacking and voting ensemble support — combines multiple compiled sub-models.

This module extends the Timber IR to support:
  - VotingEnsemble: majority vote / weighted average across sub-models
  - StackingEnsemble: meta-learner over sub-model predictions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from timber.ir.model import PipelineStage, TimberIR


@dataclass
class VotingEnsembleStage(PipelineStage):
    """Voting ensemble — aggregates predictions from multiple sub-models.

    For classification: majority vote or weighted soft vote.
    For regression: (weighted) average.
    """
    sub_models: list[TimberIR] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    voting: str = "soft"  # "hard" or "soft"

    def __post_init__(self):
        self.stage_type = "voting_ensemble"
        if not self.weights:
            self.weights = [1.0] * len(self.sub_models)


@dataclass
class StackingEnsembleStage(PipelineStage):
    """Stacking ensemble — meta-learner trained on sub-model outputs.

    The sub-models produce feature vectors fed to a final estimator.
    """
    sub_models: list[TimberIR] = field(default_factory=list)
    meta_model: Optional[TimberIR] = None
    passthrough: bool = False  # include original features alongside sub-model outputs

    def __post_init__(self):
        self.stage_type = "stacking_ensemble"


def build_voting_ensemble(
    sub_models: list[TimberIR],
    weights: Optional[list[float]] = None,
    voting: str = "soft",
    name: str = "voting_ensemble",
) -> TimberIR:
    """Construct a TimberIR wrapping a voting ensemble of sub-models."""
    if not sub_models:
        raise ValueError("At least one sub-model is required")

    stage = VotingEnsembleStage(
        stage_name=name,
        stage_type="voting_ensemble",
        sub_models=sub_models,
        weights=weights or [1.0] * len(sub_models),
        voting=voting,
    )

    # Use the first sub-model's schema as the ensemble schema
    schema = sub_models[0].schema

    from timber.ir.model import Metadata
    metadata = Metadata(
        source_framework="timber_voting",
        source_framework_version=[0, 1, 0],
        source_artifact_hash="",
        feature_names=[f.name for f in schema.input_fields],
        objective_name="voting_ensemble",
        training_params={"n_sub_models": len(sub_models), "voting": voting},
    )

    return TimberIR(pipeline=[stage], schema=schema, metadata=metadata)


def build_stacking_ensemble(
    sub_models: list[TimberIR],
    meta_model: TimberIR,
    passthrough: bool = False,
    name: str = "stacking_ensemble",
) -> TimberIR:
    """Construct a TimberIR wrapping a stacking ensemble."""
    if not sub_models:
        raise ValueError("At least one sub-model is required")

    stage = StackingEnsembleStage(
        stage_name=name,
        stage_type="stacking_ensemble",
        sub_models=sub_models,
        meta_model=meta_model,
        passthrough=passthrough,
    )

    schema = sub_models[0].schema

    from timber.ir.model import Metadata
    metadata = Metadata(
        source_framework="timber_stacking",
        source_framework_version=[0, 1, 0],
        source_artifact_hash="",
        feature_names=[f.name for f in schema.input_fields],
        objective_name="stacking_ensemble",
        training_params={"n_sub_models": len(sub_models), "passthrough": passthrough},
    )

    return TimberIR(pipeline=[stage], schema=schema, metadata=metadata)
