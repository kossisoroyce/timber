"""Timber IR — Core data model for the intermediate representation.

A Timber IR document consists of three top-level sections:
- Pipeline: ordered sequence of preprocessing transforms and models
- Schema: input/output field definitions with types and constraints
- Metadata: provenance, framework version, compilation hints
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FieldType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT32 = "int32"
    INT8 = "int8"
    BOOL = "bool"
    CATEGORICAL = "categorical"


class Objective(Enum):
    BINARY_CLASSIFICATION = "binary:logistic"
    MULTICLASS_CLASSIFICATION = "multi:softprob"
    REGRESSION = "reg:squarederror"
    REGRESSION_LOGISTIC = "reg:logistic"
    RANKING = "rank:pairwise"
    CUSTOM = "custom"


class PrecisionMode(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    MIXED = "mixed"


class EncoderType(Enum):
    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    TARGET = "target"


class ImputerStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class Field:
    name: str
    dtype: FieldType
    index: int
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[list[str]] = None


@dataclass
class Schema:
    input_fields: list[Field] = field(default_factory=list)
    output_fields: list[Field] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.input_fields)

    @property
    def n_outputs(self) -> int:
        return len(self.output_fields)


# ---------------------------------------------------------------------------
# Tree representation — flat array of nodes
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A single node in the flat tree array.

    Internal nodes have feature_index >= 0 and left_child/right_child set.
    Leaf nodes have is_leaf=True and leaf_value set.
    """
    node_id: int
    feature_index: int = -1
    threshold: float = 0.0
    left_child: int = -1      # offset into the flat node array
    right_child: int = -1     # offset into the flat node array
    is_leaf: bool = False
    leaf_value: float = 0.0
    leaf_distribution: Optional[list[float]] = None  # for multi-class
    depth: int = 0
    default_left: bool = True  # missing value direction

    def copy(self) -> TreeNode:
        return copy.deepcopy(self)


@dataclass
class Tree:
    """A single decision tree represented as a flat node array."""
    tree_id: int
    nodes: list[TreeNode] = field(default_factory=list)
    max_depth: int = 0
    n_leaves: int = 0
    n_internal: int = 0

    def recount(self) -> None:
        """Recompute leaf/internal counts from the node array."""
        self.n_leaves = sum(1 for n in self.nodes if n.is_leaf)
        self.n_internal = sum(1 for n in self.nodes if not n.is_leaf)
        if self.nodes:
            self.max_depth = max(n.depth for n in self.nodes)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    """Base class for all pipeline stages."""
    stage_name: str
    stage_type: str


@dataclass
class ScalerStage(PipelineStage):
    """Elementwise affine transform: out = (x - mean) / scale."""
    means: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    feature_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.stage_type = "scaler"


@dataclass
class EncoderStage(PipelineStage):
    """Categorical to numeric transform."""
    encoder_type: EncoderType = EncoderType.ORDINAL
    categories_per_feature: dict[int, list[str]] = field(default_factory=dict)
    feature_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.stage_type = "encoder"


@dataclass
class ImputerStage(PipelineStage):
    """Missing value fill."""
    strategy: ImputerStrategy = ImputerStrategy.MEAN
    fill_values: list[float] = field(default_factory=list)
    feature_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.stage_type = "imputer"


@dataclass
class TreeEnsembleStage(PipelineStage):
    """Gradient boosted or bagged tree ensemble."""
    trees: list[Tree] = field(default_factory=list)
    n_features: int = 0
    n_classes: int = 1
    objective: Objective = Objective.REGRESSION
    base_score: float = 0.5
    learning_rate: float = 0.1
    is_boosted: bool = True  # GBT vs bagged (RF)
    annotations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.stage_type = "tree_ensemble"

    @property
    def n_trees(self) -> int:
        return len(self.trees)

    @property
    def max_depth(self) -> int:
        if not self.trees:
            return 0
        return max(t.max_depth for t in self.trees)


@dataclass
class LinearStage(PipelineStage):
    """Dot product + bias with optional nonlinearity."""
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    activation: str = "none"  # none, sigmoid, softmax

    def __post_init__(self):
        self.stage_type = "linear"


@dataclass
class AggregatorStage(PipelineStage):
    """Combines multiple stage outputs (voting, stacking)."""
    method: str = "average"  # average, vote, stack
    source_stages: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.stage_type = "aggregator"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class Metadata:
    source_framework: str = ""
    source_framework_version: str = ""
    source_artifact_hash: str = ""
    feature_names: list[str] = field(default_factory=list)
    feature_importances: list[float] = field(default_factory=list)
    objective_name: str = ""
    training_params: dict[str, Any] = field(default_factory=dict)
    compilation_hints: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level IR
# ---------------------------------------------------------------------------

@dataclass
class TimberIR:
    """Top-level Timber IR document."""
    pipeline: list[PipelineStage] = field(default_factory=list)
    schema: Schema = field(default_factory=Schema)
    metadata: Metadata = field(default_factory=Metadata)

    def get_tree_ensemble(self) -> Optional[TreeEnsembleStage]:
        """Return the first TreeEnsembleStage in the pipeline, if any."""
        for stage in self.pipeline:
            if isinstance(stage, TreeEnsembleStage):
                return stage
        return None

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary of the IR."""
        info: dict[str, Any] = {
            "n_stages": len(self.pipeline),
            "stages": [s.stage_type for s in self.pipeline],
            "n_input_features": self.schema.n_features,
            "n_outputs": self.schema.n_outputs,
            "source_framework": self.metadata.source_framework,
        }
        ensemble = self.get_tree_ensemble()
        if ensemble:
            info["n_trees"] = ensemble.n_trees
            info["max_depth"] = ensemble.max_depth
            info["n_features"] = ensemble.n_features
            info["objective"] = ensemble.objective.value
            info["n_classes"] = ensemble.n_classes
            total_nodes = sum(len(t.nodes) for t in ensemble.trees)
            total_leaves = sum(t.n_leaves for t in ensemble.trees)
            info["total_nodes"] = total_nodes
            info["total_leaves"] = total_leaves
        return info

    def deep_copy(self) -> TimberIR:
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the IR to a JSON-compatible dict."""
        return _ir_to_dict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TimberIR:
        return _ir_from_dict(data)

    @staticmethod
    def from_json(text: str) -> TimberIR:
        return TimberIR.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _node_to_dict(n: TreeNode) -> dict[str, Any]:
    d: dict[str, Any] = {
        "node_id": n.node_id,
        "feature_index": n.feature_index,
        "threshold": n.threshold,
        "left_child": n.left_child,
        "right_child": n.right_child,
        "is_leaf": n.is_leaf,
        "leaf_value": n.leaf_value,
        "depth": n.depth,
        "default_left": n.default_left,
    }
    if n.leaf_distribution is not None:
        d["leaf_distribution"] = n.leaf_distribution
    return d


def _node_from_dict(d: dict[str, Any]) -> TreeNode:
    return TreeNode(
        node_id=d["node_id"],
        feature_index=d.get("feature_index", -1),
        threshold=d.get("threshold", 0.0),
        left_child=d.get("left_child", -1),
        right_child=d.get("right_child", -1),
        is_leaf=d.get("is_leaf", False),
        leaf_value=d.get("leaf_value", 0.0),
        leaf_distribution=d.get("leaf_distribution"),
        depth=d.get("depth", 0),
        default_left=d.get("default_left", True),
    )


def _tree_to_dict(t: Tree) -> dict[str, Any]:
    return {
        "tree_id": t.tree_id,
        "nodes": [_node_to_dict(n) for n in t.nodes],
        "max_depth": t.max_depth,
        "n_leaves": t.n_leaves,
        "n_internal": t.n_internal,
    }


def _tree_from_dict(d: dict[str, Any]) -> Tree:
    tree = Tree(
        tree_id=d["tree_id"],
        nodes=[_node_from_dict(n) for n in d["nodes"]],
        max_depth=d.get("max_depth", 0),
        n_leaves=d.get("n_leaves", 0),
        n_internal=d.get("n_internal", 0),
    )
    return tree


def _stage_to_dict(s: PipelineStage) -> dict[str, Any]:
    d: dict[str, Any] = {"stage_name": s.stage_name, "stage_type": s.stage_type}
    if isinstance(s, ScalerStage):
        d["means"] = s.means
        d["scales"] = s.scales
        d["feature_indices"] = s.feature_indices
    elif isinstance(s, EncoderStage):
        d["encoder_type"] = s.encoder_type.value
        d["categories_per_feature"] = {str(k): v for k, v in s.categories_per_feature.items()}
        d["feature_indices"] = s.feature_indices
    elif isinstance(s, ImputerStage):
        d["strategy"] = s.strategy.value
        d["fill_values"] = s.fill_values
        d["feature_indices"] = s.feature_indices
    elif isinstance(s, TreeEnsembleStage):
        d["trees"] = [_tree_to_dict(t) for t in s.trees]
        d["n_features"] = s.n_features
        d["n_classes"] = s.n_classes
        d["objective"] = s.objective.value
        d["base_score"] = s.base_score
        d["learning_rate"] = s.learning_rate
        d["is_boosted"] = s.is_boosted
    elif isinstance(s, LinearStage):
        d["weights"] = s.weights
        d["bias"] = s.bias
        d["activation"] = s.activation
    elif isinstance(s, AggregatorStage):
        d["method"] = s.method
        d["source_stages"] = s.source_stages
    return d


def _stage_from_dict(d: dict[str, Any]) -> PipelineStage:
    st = d["stage_type"]
    name = d["stage_name"]
    if st == "scaler":
        return ScalerStage(
            stage_name=name,
            stage_type=st,
            means=d.get("means", []),
            scales=d.get("scales", []),
            feature_indices=d.get("feature_indices", []),
        )
    if st == "encoder":
        return EncoderStage(
            stage_name=name,
            stage_type=st,
            encoder_type=EncoderType(d.get("encoder_type", "ordinal")),
            categories_per_feature={int(k): v for k, v in d.get("categories_per_feature", {}).items()},
            feature_indices=d.get("feature_indices", []),
        )
    if st == "imputer":
        return ImputerStage(
            stage_name=name,
            stage_type=st,
            strategy=ImputerStrategy(d.get("strategy", "mean")),
            fill_values=d.get("fill_values", []),
            feature_indices=d.get("feature_indices", []),
        )
    if st == "tree_ensemble":
        return TreeEnsembleStage(
            stage_name=name,
            stage_type=st,
            trees=[_tree_from_dict(t) for t in d.get("trees", [])],
            n_features=d.get("n_features", 0),
            n_classes=d.get("n_classes", 1),
            objective=Objective(d.get("objective", "reg:squarederror")),
            base_score=d.get("base_score", 0.5),
            learning_rate=d.get("learning_rate", 0.1),
            is_boosted=d.get("is_boosted", True),
        )
    if st == "linear":
        return LinearStage(
            stage_name=name,
            stage_type=st,
            weights=d.get("weights", []),
            bias=d.get("bias", 0.0),
            activation=d.get("activation", "none"),
        )
    if st == "aggregator":
        return AggregatorStage(
            stage_name=name,
            stage_type=st,
            method=d.get("method", "average"),
            source_stages=d.get("source_stages", []),
        )
    return PipelineStage(stage_name=name, stage_type=st)


def _schema_to_dict(s: Schema) -> dict[str, Any]:
    def _field(f: Field) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": f.name,
            "dtype": f.dtype.value,
            "index": f.index,
            "nullable": f.nullable,
        }
        if f.min_value is not None:
            d["min_value"] = f.min_value
        if f.max_value is not None:
            d["max_value"] = f.max_value
        if f.categories is not None:
            d["categories"] = f.categories
        return d

    return {
        "input_fields": [_field(f) for f in s.input_fields],
        "output_fields": [_field(f) for f in s.output_fields],
    }


def _schema_from_dict(d: dict[str, Any]) -> Schema:
    def _field(fd: dict[str, Any]) -> Field:
        return Field(
            name=fd["name"],
            dtype=FieldType(fd["dtype"]),
            index=fd["index"],
            nullable=fd.get("nullable", False),
            min_value=fd.get("min_value"),
            max_value=fd.get("max_value"),
            categories=fd.get("categories"),
        )

    return Schema(
        input_fields=[_field(f) for f in d.get("input_fields", [])],
        output_fields=[_field(f) for f in d.get("output_fields", [])],
    )


def _metadata_to_dict(m: Metadata) -> dict[str, Any]:
    return {
        "source_framework": m.source_framework,
        "source_framework_version": m.source_framework_version,
        "source_artifact_hash": m.source_artifact_hash,
        "feature_names": m.feature_names,
        "feature_importances": m.feature_importances,
        "objective_name": m.objective_name,
        "training_params": m.training_params,
        "compilation_hints": m.compilation_hints,
    }


def _metadata_from_dict(d: dict[str, Any]) -> Metadata:
    return Metadata(
        source_framework=d.get("source_framework", ""),
        source_framework_version=d.get("source_framework_version", ""),
        source_artifact_hash=d.get("source_artifact_hash", ""),
        feature_names=d.get("feature_names", []),
        feature_importances=d.get("feature_importances", []),
        objective_name=d.get("objective_name", ""),
        training_params=d.get("training_params", {}),
        compilation_hints=d.get("compilation_hints", {}),
    )


def _ir_to_dict(ir: TimberIR) -> dict[str, Any]:
    return {
        "timber_ir_version": "0.1",
        "pipeline": [_stage_to_dict(s) for s in ir.pipeline],
        "schema": _schema_to_dict(ir.schema),
        "metadata": _metadata_to_dict(ir.metadata),
    }


def _ir_from_dict(d: dict[str, Any]) -> TimberIR:
    return TimberIR(
        pipeline=[_stage_from_dict(s) for s in d.get("pipeline", [])],
        schema=_schema_from_dict(d.get("schema", {})),
        metadata=_metadata_from_dict(d.get("metadata", {})),
    )
