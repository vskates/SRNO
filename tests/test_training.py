from __future__ import annotations

from pathlib import Path

import pytest
import torch

from srno.data.dataset import LocalTransitionBatch, TrajectoryBatch
from srno.data.tools import build_active_index
from srno.geometry.gripper import GripperAsset
from srno.losses import state_error
from srno.model import SRNOModel
from srno.training.config import (
    ExperimentConfig,
    LoaderConfig,
    LossConfig,
    PathsConfig,
    TrainingConfig,
)
from srno.training.engine import (
    _local_iteration,
    _run_epoch,
    evaluate_checkpoint,
    train,
)
from srno.types import PoseState, SDFBatch


def test_legacy_aperture_config_loads_but_drive_error_is_rejected(
    tmp_path: Path,
) -> None:
    template = """
[paths]
manifest = "manifest.json"
active_index = "active.npz"
output_dir = "run"

[model]
global_conditioning = "{conditioning}"
"""
    aperture = tmp_path / "aperture.toml"
    aperture.write_text(
        template.format(conditioning="aperture"), encoding="utf-8"
    )
    loaded = ExperimentConfig.load(aperture)
    assert not hasattr(loaded.model, "global_conditioning")

    drive_error = tmp_path / "drive-error.toml"
    drive_error.write_text(
        template.format(conditioning="drive_error"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="drive_error conditioning was removed"):
        ExperimentConfig.load(drive_error)


def test_contact_offset_is_not_applied_to_geometric_feasibility(
    gripper: GripperAsset, tmp_path: Path
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=0.004,
        contact_offset_sum=0.00256,
    )
    state = PoseState(
        torch.eye(3)[None],
        torch.zeros(1, 3),
        gripper.free_joint_configuration(0.08)[None],
    )
    sdf = SDFBatch(
        torch.full((1, 8, 8, 8), 0.001),
        torch.full((1, 3), -0.04),
        torch.full((1, 3), 0.01),
        torch.zeros(1, dtype=torch.long),
        0.02,
    )
    batch = LocalTransitionBatch(
        sdf=sdf,
        current=state,
        target=state,
        next_command=torch.tensor([0.07]),
        target_aperture=torch.tensor([0.08]),
        object_ids=("object",),
        trajectory_index=torch.zeros(1, dtype=torch.long),
        step_index=torch.zeros(1, dtype=torch.long),
    )
    config = ExperimentConfig(
        paths=PathsConfig(
            tmp_path / "manifest.json",
            tmp_path / "active.npz",
            tmp_path / "run",
        ),
        loss=LossConfig(
            local_lambda_joints=0.0,
            local_lambda_feasibility=0.0,
        ),
        training=TrainingConfig(use_bfloat16=False),
        device="cpu",
    )

    assert model.query_gap(state, sdf).amin() < 0.0
    assert torch.all(model.query_geometric_gap(state, sdf) > 0.0)
    loss, metrics = _local_iteration(model, batch, config)
    assert metrics["feasibility"] == 0.0
    assert metrics["joints"] > 0.0
    assert loss == 0.0


def test_all_free_rollout_batch_skips_adamw_update(gripper: GripperAsset, tmp_path: Path) -> None:
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    schedule = torch.linspace(0.08, 0.0, 33)
    joint_schedule = gripper.free_joint_configuration(schedule)
    batch = TrajectoryBatch(
        sdf=SDFBatch(
            torch.full((1, 8, 8, 8), 0.02),
            torch.full((1, 3), -0.04),
            torch.full((1, 3), 0.01),
            torch.zeros(1, dtype=torch.long),
            0.02,
        ),
        states=PoseState(
            torch.eye(3).expand(1, 33, 3, 3).clone(),
            torch.zeros(1, 33, 3),
            joint_schedule[None].clone(),
        ),
        command_schedule=schedule,
        actual_aperture=schedule[None].clone(),
        object_ids=("free",),
        trajectory_index=torch.zeros(1, dtype=torch.long),
    )
    config = ExperimentConfig(
        paths=PathsConfig(tmp_path / "manifest.json", tmp_path / "active.npz", tmp_path / "run"),
        training=TrainingConfig(use_bfloat16=False),
        device="cpu",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = _run_epoch(
        model,
        [batch],
        config,
        torch.device("cpu"),
        mode="rollout",
        horizon=4,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert metrics["gradient_norm"] == 0.0
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_small_contact_problem_loss_decreases(gripper: GripperAsset) -> None:
    device = torch.device("cuda")
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005).to(device)
    state = PoseState(
        torch.eye(3, device=device)[None],
        torch.zeros(1, 3, device=device),
        gripper.free_joint_configuration(0.08)[None].to(device),
    )
    target_joint = gripper.free_joint_configuration(0.07)[None].to(device)
    target_joint = target_joint + 0.01 * torch.tensor(
        [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]], device=device
    )
    target = PoseState(
        torch.eye(3, device=device)[None],
        torch.tensor([[0.002, 0.0, 0.0]], device=device),
        target_joint,
    )
    sdf = SDFBatch(
        torch.full((1, 8, 8, 8), -1e-3, device=device),
        torch.full((1, 3), -0.04, device=device),
        torch.full((1, 3), 0.01, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
        0.02,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_step(state, 0.07, sdf)
        assert isinstance(prediction, PoseState)
        loss = state_error(
            prediction,
            target,
            length_scale=0.08,
            joint_scale=model.joint_travel_range,
        )[0].mean()
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_local_trainer_end_to_end(dataset_bundle: Path, tmp_path: Path) -> None:
    active_path = tmp_path / "active.npz"
    build_active_index(dataset_bundle, active_path)
    config = ExperimentConfig(
        paths=PathsConfig(dataset_bundle, active_path, tmp_path / "run"),
        loader=LoaderConfig(
            objects_per_batch=1,
            local_samples_per_object=1,
            rollout_trajectories_per_object=1,
            workers=0,
        ),
        training=TrainingConfig(
            local_epochs=1,
            rollout_epochs_per_horizon=1,
            rollout_horizons=(4, 8, 16, 32),
            early_stopping_patience=1,
            use_bfloat16=True,
        ),
        device="cuda",
    )
    checkpoint = train(config, stage="local")
    assert checkpoint.is_file()
    assert (tmp_path / "run" / "last-local.pt").is_file()
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    assert any((tmp_path / "run" / "tensorboard").glob("events.out.tfevents.*"))
    rollout_checkpoint = train(config, stage="rollout", resume=checkpoint)
    assert rollout_checkpoint.is_file()
    metrics = evaluate_checkpoint(config, rollout_checkpoint, split="test")
    assert metrics["terminal_dx"] >= 0
    assert metrics["max_penetration_m"] >= 0
