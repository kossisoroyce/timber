"""Tests for the URDF -> KinematicsStage -> C99 forward-kinematics pipeline."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import unittest

import numpy as np

from timber.codegen.c99 import C99Emitter
from timber.frontends.urdf_parser import URDFParser
from timber.ir.model import (
    KinematicsStage,
    Metadata,
    Schema,
    TimberIR,
)

# ---------------------------------------------------------------------------
# Minimal URDF fixtures
# ---------------------------------------------------------------------------

_URDF_2R = """\
<robot name="two_link">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>

  <joint name="joint1" type="revolute">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5708" upper="1.5708" effort="100" velocity="1"/>
  </joint>

  <joint name="joint2" type="revolute">
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5708" upper="1.5708" effort="100" velocity="1"/>
  </joint>
</robot>
"""

_URDF_MIXED = """\
<robot name="mixed">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="tool"/>

  <joint name="j_fixed" type="fixed">
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="link1"/>
  </joint>

  <joint name="j_rev" type="revolute">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="10" velocity="1"/>
  </joint>

  <joint name="j_pris" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="link2"/>
    <child link="tool"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0.3" effort="10" velocity="0.1"/>
  </joint>
</robot>
"""

_URDF_ALL_FIXED = """\
<robot name="static">
  <link name="base_link"/>
  <link name="end"/>
  <joint name="j0" type="fixed">
    <origin xyz="1 2 3" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="end"/>
  </joint>
</robot>
"""


# ---------------------------------------------------------------------------
# Pure-Python reference FK (used for numerical comparison)
# ---------------------------------------------------------------------------

def _rpy_to_mat4(rpy, xyz):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, xyz[0]],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, xyz[1]],
        [-sp,   cp*sr,            cp*cr,            xyz[2]],
        [0,     0,                0,                1     ],
    ], dtype=np.float64)


def _rodrigues(axis, angle):
    ax, ay, az = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    R = np.array([
        [c+ax*ax*t,    ax*ay*t-az*s, ax*az*t+ay*s, 0],
        [ay*ax*t+az*s, c+ay*ay*t,    ay*az*t-ax*s, 0],
        [az*ax*t-ay*s, az*ay*t+ax*s, c+az*az*t,    0],
        [0,            0,            0,            1],
    ], dtype=np.float64)
    return R


def _prismatic(axis, q):
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = axis[0] * q
    T[1, 3] = axis[1] * q
    T[2, 3] = axis[2] * q
    return T


def reference_fk(stage: KinematicsStage, joint_angles: list[float]) -> np.ndarray:
    """Pure-Python reference FK: returns 4x4 transform as numpy array."""
    acc = np.eye(4, dtype=np.float64)
    qi = 0
    for j in stage.joints:
        T_origin = _rpy_to_mat4(j.origin_rpy, j.origin_xyz)
        acc = acc @ T_origin
        if j.joint_type in ("revolute", "continuous"):
            Tj = _rodrigues(j.axis, joint_angles[qi])
            acc = acc @ Tj
            qi += 1
        elif j.joint_type == "prismatic":
            Tj = _prismatic(j.axis, joint_angles[qi])
            acc = acc @ Tj
            qi += 1
    return acc


# ---------------------------------------------------------------------------
# C99 compile + run helper
# ---------------------------------------------------------------------------

def _compile_and_run_fk(stage: KinematicsStage, joint_angles: list[float]) -> list[float]:
    """Compile the emitted C99 to a shared lib, run FK, return 16-element transform."""
    ir = TimberIR(
        pipeline=[stage],
        schema=Schema(),
        metadata=Metadata(),
    )
    emitter = C99Emitter()
    output = emitter.emit(ir)

    with tempfile.TemporaryDirectory() as tmpdir:
        output.write(tmpdir)
        harness = os.path.join(tmpdir, "harness.c")
        exe = os.path.join(tmpdir, "harness")

        n_dof = stage.n_dof
        angle_inits = ", ".join(f"{v:.10f}f" for v in joint_angles)

        harness_src = f"""\
#include <stdio.h>
#include "model.h"

int main(void) {{
    TimberCtx *ctx = NULL;
    float inputs[{n_dof}] = {{{angle_inits}}};
    float outputs[16] = {{0}};
    int i;
    timber_init(&ctx);
    timber_infer_single(inputs, outputs, ctx);
    for (i = 0; i < 16; i++) printf("%.10f\\n", (double)outputs[i]);
    timber_free(ctx);
    return 0;
}}
"""
        with open(harness, "w") as f:
            f.write(harness_src)

        result = subprocess.run(
            ["gcc", "-std=c99", "-O2", "-Wall",
             harness, os.path.join(tmpdir, "model.c"),
             "-lm", "-o", exe, f"-I{tmpdir}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compile failed:\n{result.stderr}")

        run = subprocess.run([exe], capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"Run failed:\n{run.stderr}")

        return [float(ln.strip()) for ln in run.stdout.strip().splitlines()]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestURDFParser(unittest.TestCase):

    def setUp(self):
        self.parser = URDFParser()

    def test_two_revolute_n_dof(self):
        ir = self.parser.parse_string(_URDF_2R)
        stage = ir.pipeline[0]
        self.assertIsInstance(stage, KinematicsStage)
        self.assertEqual(stage.n_dof, 2)

    def test_two_revolute_n_joints(self):
        ir = self.parser.parse_string(_URDF_2R)
        stage = ir.pipeline[0]
        self.assertEqual(len(stage.joints), 2)

    def test_joint_names_preserved(self):
        ir = self.parser.parse_string(_URDF_2R)
        stage = ir.pipeline[0]
        self.assertEqual(stage.joints[0].name, "joint1")
        self.assertEqual(stage.joints[1].name, "joint2")

    def test_joint_origin_xyz(self):
        ir = self.parser.parse_string(_URDF_2R)
        stage = ir.pipeline[0]
        self.assertAlmostEqual(stage.joints[0].origin_xyz[2], 0.1)
        self.assertAlmostEqual(stage.joints[1].origin_xyz[0], 0.5)

    def test_joint_axis(self):
        ir = self.parser.parse_string(_URDF_2R)
        stage = ir.pipeline[0]
        self.assertEqual(stage.joints[0].axis, [0.0, 0.0, 1.0])

    def test_joint_limits(self):
        ir = self.parser.parse_string(_URDF_2R)
        stage = ir.pipeline[0]
        self.assertAlmostEqual(stage.joints[0].limit_lower, -1.5708, places=3)
        self.assertAlmostEqual(stage.joints[0].limit_upper,  1.5708, places=3)

    def test_fixed_joint_excluded_from_dof(self):
        ir = self.parser.parse_string(_URDF_MIXED)
        stage = ir.pipeline[0]
        self.assertEqual(stage.n_dof, 2)

    def test_mixed_joint_types_present(self):
        ir = self.parser.parse_string(_URDF_MIXED)
        stage = ir.pipeline[0]
        types = [j.joint_type for j in stage.joints]
        self.assertIn("fixed", types)
        self.assertIn("revolute", types)
        self.assertIn("prismatic", types)

    def test_all_fixed_zero_dof(self):
        ir = self.parser.parse_string(_URDF_ALL_FIXED)
        stage = ir.pipeline[0]
        self.assertEqual(stage.n_dof, 0)

    def test_schema_input_fields_equal_n_dof(self):
        ir = self.parser.parse_string(_URDF_2R)
        self.assertEqual(len(ir.schema.input_fields), 2)

    def test_schema_output_fields_16(self):
        ir = self.parser.parse_string(_URDF_2R)
        self.assertEqual(len(ir.schema.output_fields), 16)

    def test_metadata_source_framework(self):
        ir = self.parser.parse_string(_URDF_2R)
        self.assertEqual(ir.metadata.source_framework, "urdf")

    def test_base_link_auto_detected(self):
        ir = self.parser.parse_string(_URDF_2R)
        stage = ir.pipeline[0]
        self.assertEqual(stage.base_link, "base_link")

    def test_explicit_end_effector(self):
        ir = self.parser.parse_string(_URDF_2R, end_effector="link2")
        stage = ir.pipeline[0]
        self.assertEqual(stage.end_effector, "link2")

    def test_no_joints_raises(self):
        with self.assertRaises(ValueError):
            self.parser.parse_string("<robot name='x'><link name='a'/></robot>")

    def test_parse_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
            f.write(_URDF_2R)
            path = f.name
        try:
            ir = self.parser.parse(path)
            self.assertIsInstance(ir.pipeline[0], KinematicsStage)
        finally:
            os.unlink(path)


class TestKinematicsIR(unittest.TestCase):

    def _make_stage(self):
        parser = URDFParser()
        ir = parser.parse_string(_URDF_2R)
        return ir, ir.pipeline[0]

    def test_stage_type(self):
        _, stage = self._make_stage()
        self.assertEqual(stage.stage_type, "kinematics")

    def test_n_dof_property(self):
        _, stage = self._make_stage()
        self.assertEqual(stage.n_dof, 2)

    def test_json_roundtrip_stage_type(self):
        ir, _ = self._make_stage()
        ir2 = TimberIR.from_json(ir.to_json())
        self.assertEqual(ir2.pipeline[0].stage_type, "kinematics")

    def test_json_roundtrip_n_dof(self):
        ir, stage = self._make_stage()
        ir2 = TimberIR.from_json(ir.to_json())
        stage2 = ir2.pipeline[0]
        self.assertIsInstance(stage2, KinematicsStage)
        self.assertEqual(stage2.n_dof, stage.n_dof)

    def test_json_roundtrip_joint_names(self):
        ir, stage = self._make_stage()
        ir2 = TimberIR.from_json(ir.to_json())
        stage2 = ir2.pipeline[0]
        self.assertEqual(
            [j.name for j in stage2.joints],
            [j.name for j in stage.joints],
        )

    def test_json_roundtrip_origin_xyz(self):
        ir, stage = self._make_stage()
        ir2 = TimberIR.from_json(ir.to_json())
        stage2 = ir2.pipeline[0]
        for j_orig, j_new in zip(stage.joints, stage2.joints):
            self.assertEqual(j_orig.origin_xyz, j_new.origin_xyz)

    def test_json_roundtrip_axis(self):
        ir, stage = self._make_stage()
        ir2 = TimberIR.from_json(ir.to_json())
        stage2 = ir2.pipeline[0]
        for j_orig, j_new in zip(stage.joints, stage2.joints):
            self.assertEqual(j_orig.axis, j_new.axis)

    def test_json_is_valid_utf8(self):
        ir, _ = self._make_stage()
        self.assertIsInstance(ir.to_json().encode("utf-8"), bytes)

    def test_deep_copy_independence(self):
        ir, stage = self._make_stage()
        ir2 = ir.deep_copy()
        stage2 = ir2.pipeline[0]
        stage2.joints[0].name = "MODIFIED"
        self.assertNotEqual(stage.joints[0].name, "MODIFIED")


class TestC99KinematicsEmit(unittest.TestCase):

    def _emit(self, urdf_str):
        parser = URDFParser()
        ir = parser.parse_string(urdf_str)
        return C99Emitter().emit(ir)

    def test_header_has_include_guard(self):
        out = self._emit(_URDF_2R)
        self.assertIn("#ifndef TIMBER_MODEL_H", out.model_h)

    def test_header_has_n_dof(self):
        out = self._emit(_URDF_2R)
        self.assertIn("TIMBER_N_DOF", out.model_h)

    def test_header_n_dof_value(self):
        out = self._emit(_URDF_2R)
        self.assertIn("#define TIMBER_N_DOF 2", out.model_h)

    def test_header_n_outputs_16(self):
        out = self._emit(_URDF_2R)
        self.assertIn("#define TIMBER_N_OUTPUTS  16", out.model_h)

    def test_header_has_timber_fk(self):
        out = self._emit(_URDF_2R)
        self.assertIn("timber_fk", out.model_h)

    def test_model_c_has_rodrigues(self):
        out = self._emit(_URDF_2R)
        self.assertIn("rodrigues", out.model_c)

    def test_model_c_has_mat4_mul(self):
        out = self._emit(_URDF_2R)
        self.assertIn("mat4_mul", out.model_c)

    def test_model_c_has_timber_fk_def(self):
        out = self._emit(_URDF_2R)
        self.assertIn("int timber_fk(", out.model_c)

    def test_model_c_prismatic_mixed(self):
        out = self._emit(_URDF_MIXED)
        self.assertIn("prismatic", out.model_c)

    def test_data_has_joint_origins(self):
        out = self._emit(_URDF_2R)
        self.assertIn("JOINT_0_ORIGIN", out.model_data_c)
        self.assertIn("JOINT_1_ORIGIN", out.model_data_c)

    def test_data_has_axis_for_active_joints(self):
        out = self._emit(_URDF_2R)
        self.assertIn("JOINT_0_AXIS", out.model_data_c)
        self.assertIn("JOINT_1_AXIS", out.model_data_c)

    def test_fixed_joint_no_axis_array(self):
        out = self._emit(_URDF_ALL_FIXED)
        self.assertNotIn("JOINT_0_AXIS", out.model_data_c)

    def test_model_infer_single_delegates_to_fk(self):
        out = self._emit(_URDF_2R)
        self.assertIn("timber_fk(inputs, outputs, ctx)", out.model_c)

    @unittest.skipUnless(
        subprocess.run(["which", "gcc"], capture_output=True).returncode == 0,
        "gcc not available"
    )
    def test_c99_compiles_cleanly(self):
        parser = URDFParser()
        ir = parser.parse_string(_URDF_2R)
        out = C99Emitter().emit(ir)
        with tempfile.TemporaryDirectory() as tmpdir:
            out.write(tmpdir)
            result = subprocess.run(
                ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-c", os.path.join(tmpdir, "model.c"),
                 f"-I{tmpdir}",
                 "-o", os.path.join(tmpdir, "model.o")],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)


@unittest.skipUnless(
    subprocess.run(["which", "gcc"], capture_output=True).returncode == 0,
    "gcc not available"
)
class TestFKNumerical(unittest.TestCase):
    """Compile and run FK; compare against pure-Python reference."""

    def _stage_2r(self):
        return URDFParser().parse_string(_URDF_2R).pipeline[0]

    def _stage_mixed(self):
        return URDFParser().parse_string(_URDF_MIXED).pipeline[0]

    def _stage_fixed(self):
        return URDFParser().parse_string(_URDF_ALL_FIXED).pipeline[0]

    def _run(self, stage, angles):
        c_flat = _compile_and_run_fk(stage, angles)
        c_mat  = np.array(c_flat, dtype=np.float64).reshape(4, 4)
        py_mat = reference_fk(stage, angles)
        return c_mat, py_mat

    def test_2r_zero_angles_translation(self):
        stage = self._stage_2r()
        c, py = self._run(stage, [0.0, 0.0])
        np.testing.assert_allclose(c, py, atol=1e-5,
                                   err_msg="2R zero-angle FK mismatch")

    def test_2r_first_joint_90deg(self):
        stage = self._stage_2r()
        c, py = self._run(stage, [math.pi / 2, 0.0])
        np.testing.assert_allclose(c, py, atol=1e-5,
                                   err_msg="2R j1=90° FK mismatch")

    def test_2r_both_joints_45deg(self):
        stage = self._stage_2r()
        c, py = self._run(stage, [math.pi / 4, math.pi / 4])
        np.testing.assert_allclose(c, py, atol=1e-5,
                                   err_msg="2R j1=j2=45° FK mismatch")

    def test_2r_rotation_matrix_is_orthogonal(self):
        stage = self._stage_2r()
        c, _ = self._run(stage, [0.3, -0.7])
        R = c[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5,
                                   err_msg="Rotation sub-matrix not orthogonal")

    def test_2r_homogeneous_last_row(self):
        stage = self._stage_2r()
        c, _ = self._run(stage, [1.0, -1.0])
        np.testing.assert_allclose(c[3], [0, 0, 0, 1], atol=1e-6,
                                   err_msg="Last row of 4x4 must be [0,0,0,1]")

    def test_2r_zero_position(self):
        stage = self._stage_2r()
        c, py = self._run(stage, [0.0, 0.0])
        np.testing.assert_allclose(c[:3, 3], py[:3, 3], atol=1e-5,
                                   err_msg="End-effector position at zero angle mismatch")

    def test_mixed_zero_angles(self):
        stage = self._stage_mixed()
        c, py = self._run(stage, [0.0, 0.0])
        np.testing.assert_allclose(c, py, atol=1e-5,
                                   err_msg="Mixed zero-angle FK mismatch")

    def test_mixed_prismatic_extends(self):
        stage = self._stage_mixed()
        c0, _ = self._run(stage, [0.0, 0.0])
        c1, _ = self._run(stage, [0.0, 0.1])
        # Prismatic along Z: z-position should increase by 0.1
        self.assertAlmostEqual(c1[2, 3] - c0[2, 3], 0.1, places=4)

    def test_all_fixed_identity_rotation(self):
        stage = self._stage_fixed()
        c_flat = _compile_and_run_fk(stage, [])
        c = np.array(c_flat).reshape(4, 4)
        py = reference_fk(stage, [])
        np.testing.assert_allclose(c, py, atol=1e-6,
                                   err_msg="All-fixed FK mismatch")

    def test_all_fixed_translation_xyz(self):
        stage = self._stage_fixed()
        c_flat = _compile_and_run_fk(stage, [])
        c = np.array(c_flat).reshape(4, 4)
        np.testing.assert_allclose(c[:3, 3], [1, 2, 3], atol=1e-6,
                                   err_msg="All-fixed translation wrong")


if __name__ == "__main__":
    unittest.main()
