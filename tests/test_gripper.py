from __future__ import annotations

from pathlib import Path

import pytest
import torch

from srno.geometry.gripper import GripperAsset, preprocess_urdf


URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="synthetic_gripper">
  <link name="base"/>
  <link name="left_finger">
    <collision><geometry><box size="0.01 0.02 0.03"/></geometry></collision>
  </link>
  <link name="right_finger">
    <collision><geometry><box size="0.01 0.02 0.03"/></geometry></collision>
  </link>
  <joint name="left_joint" type="{joint_type}">
    <parent link="base"/><child link="left_finger"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="right_joint" type="{joint_type}">
    <parent link="base"/><child link="right_finger"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="-1 0 0"/>
    <limit lower="0" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
"""


def test_urdf_preprocessing_is_deterministic_and_affine(tmp_path: Path) -> None:
    urdf = tmp_path / "gripper.urdf"
    urdf.write_text(URDF_TEMPLATE.format(joint_type="prismatic"), encoding="utf-8")
    kwargs = dict(
        finger_links=("left_finger", "right_finger"),
        joint_map={"left_joint": (0.0, 0.5), "right_joint": (0.0, 0.5)},
        aperture_min=0.0,
        aperture_max=0.08,
        length_scale=0.08,
        samples_per_link=128,
        seed=5,
    )
    first = preprocess_urdf(urdf, **kwargs)
    second = preprocess_urdf(urdf, **kwargs)
    assert first.point_count == 256
    assert torch.equal(first.intercept, second.intercept)
    assert torch.equal(first.slope, second.slope)
    assert torch.allclose(first.points(0.04), 0.5 * (first.points(0.0) + first.points(0.08)))
    path = tmp_path / "asset.npz"
    first.save(path)
    restored = GripperAsset.load(path)
    assert restored.sha256() == first.sha256()


def test_urdf_preprocessing_rejects_non_affine_motion(tmp_path: Path) -> None:
    urdf = tmp_path / "gripper.urdf"
    urdf.write_text(URDF_TEMPLATE.format(joint_type="revolute"), encoding="utf-8")
    with pytest.raises(ValueError, match="not affine"):
        preprocess_urdf(
            urdf,
            finger_links=("left_finger", "right_finger"),
            joint_map={"left_joint": (0.0, 1.0), "right_joint": (0.0, 1.0)},
            aperture_min=0.0,
            aperture_max=0.8,
            length_scale=0.08,
            samples_per_link=16,
            affine_tolerance=1e-7,
        )


def test_schedule_asset_preserves_all_33_exact_states(tmp_path: Path) -> None:
    apertures = torch.linspace(0.01, 0.11, 33)
    points = torch.zeros(33, 256, 3)
    points[:, :128, 0] = 0.5 * apertures[:, None]
    points[:, 128:, 0] = -0.5 * apertures[:, None]
    points[:, :, 2] = torch.linspace(0.08, 0.17, 33)[:, None].square()
    slope = (points[-1] - points[0]) / (apertures[-1] - apertures[0])
    intercept = points[0] - apertures[0] * slope
    asset = GripperAsset(
        intercept,
        slope,
        torch.cat((torch.zeros(128), torch.ones(128))).long(),
        float(apertures[0]),
        float(apertures[-1]),
        float(apertures[-1]),
        "runtime-usd",
        apertures,
        points,
    )
    assert asset.point_count == 256
    assert asset.aperture_knots is not None
    assert asset.point_knots is not None
    assert asset.aperture_knots.shape == (33,)
    assert torch.equal(asset.points(asset.aperture_knots), asset.point_knots)
    path = tmp_path / "scheduled.npz"
    asset.save(path)
    restored = GripperAsset.load(path)
    assert restored.sha256() == asset.sha256()
    assert torch.equal(restored.points(restored.aperture_knots), restored.point_knots)


def test_joint_fk_asset_roundtrip_and_gradients(
    gripper: GripperAsset, tmp_path: Path
) -> None:
    assert gripper.supports_joint_fk
    commands = torch.tensor([0.08, 0.04, 0.0])
    joints = gripper.free_joint_configuration(commands).requires_grad_(True)
    points = gripper.points_from_joints(joints)
    assert points.shape == (3, 256, 3)
    assert torch.allclose(gripper.aperture_from_joints(joints), commands, atol=1e-6)
    points.square().sum().backward()
    assert joints.grad is not None and torch.isfinite(joints.grad).all()

    path = tmp_path / "joint-fk.npz"
    gripper.save(path)
    restored = GripperAsset.load(path)
    assert restored.supports_joint_fk
    assert restored.joint_names == gripper.joint_names
    assert restored.sha256() == gripper.sha256()
    assert torch.equal(
        restored.points_from_joints(joints.detach()), points.detach()
    )
