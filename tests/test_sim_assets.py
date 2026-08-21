from __future__ import annotations

import math

import pytest

from srno.geometry.gripper import preprocess_urdf
from srno.sim import SimulatorAssetCatalog


def test_vendored_validation_assets_are_complete_and_hashed() -> None:
    catalog = SimulatorAssetCatalog.load()

    assert len(catalog.object_ids) == 28
    assert len(set(catalog.object_ids)) == 28
    assert "ogurtsy-marinovannye-670-g-21054" not in catalog.object_ids
    assert catalog.approach_axis_local == (0.0, 0.0, 1.0)
    assert catalog.object_spawn_scale is None
    assert catalog.object_spawn_quaternion_wxyz == (0.0, 0.0, 0.0, 1.0)
    assert catalog.gripper.runtime_usd.name == "gripper_playground.usd"
    assert not catalog.gripper.affine_urdf_preprocessor_compatible

    catalog.validate_files(verify_hashes=True)


def test_object_records_use_only_catalog_local_paths() -> None:
    catalog = SimulatorAssetCatalog.load()

    for record in catalog.objects:
        assert record.spawn_scale is None
        assert record.usd_path.is_relative_to(catalog.asset_root)
        assert record.texture_directory.is_relative_to(catalog.asset_root)
        assert record.usd_path.parent == record.texture_directory.parent
        assert math.isclose(sum(q * q for q in record.spawn_quaternion_wxyz), 1.0)


def test_astribot_revolute_fingers_are_not_accepted_as_affine() -> None:
    catalog = SimulatorAssetCatalog.load()
    close = catalog.gripper.close_joint_position_rad

    with pytest.raises(ValueError, match="finger motion is not affine"):
        preprocess_urdf(
            catalog.gripper.source_urdf,
            finger_links=(
                "astribot_gripper_right_Link_L11",
                "astribot_gripper_right_Link_R11",
            ),
            joint_map={name: (0.0, position) for name, position in close.items()},
            aperture_min=0.0,
            aperture_max=1.0,
            length_scale=0.1,
            samples_per_link=32,
        )
