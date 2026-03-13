"""URDF parser — converts URDF robot descriptions to Timber KinematicsStage IR.

Supports joint types: revolute, prismatic, fixed, continuous.
Parses the kinematic chain from a base link to an end-effector link and
emits a KinematicsStage IR suitable for C99 forward-kinematics code generation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from timber.ir.model import (
    Field,
    FieldType,
    JointSpec,
    KinematicsStage,
    Metadata,
    Schema,
    TimberIR,
)

_ACTIVE_TYPES = ("revolute", "prismatic", "continuous")


def _parse_vec3(s: str) -> list[float]:
    """Parse a space-separated 3-float string; return [0,0,0] on empty."""
    parts = s.strip().split()
    if len(parts) == 3:
        return [float(p) for p in parts]
    return [0.0, 0.0, 0.0]


def _find_chain(
    joints: list[JointSpec],
    base: str,
    end: str,
) -> list[JointSpec]:
    """Return the ordered joint list from *base* to *end*.

    Walks the parent→child joint graph. Raises ValueError if no path exists.
    """
    parent_to_joint: dict[str, JointSpec] = {}
    for j in joints:
        parent_to_joint.setdefault(j.parent, j)

    chain: list[JointSpec] = []
    current = base
    visited: set[str] = set()

    while current in parent_to_joint:
        if current in visited:
            raise ValueError(f"Kinematic loop detected at link '{current}'")
        visited.add(current)
        joint = parent_to_joint[current]
        chain.append(joint)
        if joint.child == end:
            break
        current = joint.child

    return chain


class URDFParser:
    """Parse URDF XML and return a TimberIR containing a KinematicsStage."""

    def parse(
        self,
        path: str | Path,
        base_link: Optional[str] = None,
        end_effector: Optional[str] = None,
    ) -> TimberIR:
        """Parse a URDF file from disk."""
        tree = ET.parse(str(path))
        return self._build_ir(tree.getroot(), base_link, end_effector, source=str(path))

    def parse_string(
        self,
        xml_string: str,
        base_link: Optional[str] = None,
        end_effector: Optional[str] = None,
    ) -> TimberIR:
        """Parse a URDF XML string."""
        root = ET.fromstring(xml_string)
        return self._build_ir(root, base_link, end_effector, source="<string>")

    def _build_ir(
        self,
        root: ET.Element,
        base_link: Optional[str],
        end_effector: Optional[str],
        source: str,
    ) -> TimberIR:
        robot_name = root.get("name", "robot")

        links = [el.get("name", "") for el in root.findall("link")]
        if not links:
            raise ValueError("URDF has no <link> elements")

        all_joints: list[JointSpec] = []
        child_links: set[str] = set()

        for el in root.findall("joint"):
            jname = el.get("name", "")
            jtype = el.get("type", "fixed")

            origin_el = el.find("origin")
            if origin_el is not None:
                origin_xyz = _parse_vec3(origin_el.get("xyz", "0 0 0"))
                origin_rpy = _parse_vec3(origin_el.get("rpy", "0 0 0"))
            else:
                origin_xyz = [0.0, 0.0, 0.0]
                origin_rpy = [0.0, 0.0, 0.0]

            axis_el = el.find("axis")
            axis = _parse_vec3(axis_el.get("xyz", "0 0 1")) if axis_el is not None else [0.0, 0.0, 1.0]

            parent_el = el.find("parent")
            parent = parent_el.get("link", "") if parent_el is not None else ""

            child_el = el.find("child")
            child = child_el.get("link", "") if child_el is not None else ""

            limit_el = el.find("limit")
            if limit_el is not None:
                limit_lower = float(limit_el.get("lower", "-3.14159265"))
                limit_upper = float(limit_el.get("upper", "3.14159265"))
            else:
                limit_lower = -3.14159265
                limit_upper = 3.14159265

            all_joints.append(JointSpec(
                name=jname,
                joint_type=jtype,
                axis=axis,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                parent=parent,
                child=child,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
            ))
            child_links.add(child)

        if not all_joints:
            raise ValueError("URDF has no <joint> elements")

        if base_link is None:
            roots = [lnk for lnk in links if lnk not in child_links]
            base_link = roots[0] if roots else links[0]

        if end_effector is None:
            parent_links = {j.parent for j in all_joints}
            leaves = [lnk for lnk in links if lnk not in parent_links]
            end_effector = leaves[-1] if leaves else links[-1]

        chain = _find_chain(all_joints, base_link, end_effector)
        if not chain:
            raise ValueError(
                f"No joint chain found from '{base_link}' to '{end_effector}'"
            )

        stage = KinematicsStage(
            stage_name="kinematics",
            stage_type="kinematics",
            joints=chain,
            base_link=base_link,
            end_effector=end_effector,
        )

        n_dof = stage.n_dof
        schema = Schema(
            input_fields=[
                Field(name=f"q{i}", dtype=FieldType.FLOAT32, index=i)
                for i in range(n_dof)
            ],
            output_fields=[
                Field(name=f"t{i}", dtype=FieldType.FLOAT32, index=i)
                for i in range(16)
            ],
        )
        metadata = Metadata(
            source_framework="urdf",
            source_framework_version="1.0",
            feature_names=[f"q{i}" for i in range(n_dof)],
            objective_name="forward_kinematics",
            training_params={"robot_name": robot_name, "source": source},
        )

        return TimberIR(pipeline=[stage], schema=schema, metadata=metadata)
