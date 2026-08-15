from __future__ import annotations

from pathlib import Path

import pytest
import torch

from srno.data.tools import build_active_index
from srno.geometry.gripper import GripperAsset
from srno.losses import state_error
from srno.model import SRNOModel
from srno.training.config import (
    ExperimentConfig,
    LoaderConfig,
    PathsConfig,
    TrainingConfig,
)
from srno.training.engine import evaluate_checkpoint, train
from srno.types import PoseState, SDFBatch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_small_contact_problem_loss_decreases(gripper: GripperAsset) -> None:
    device = torch.device("cuda")
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005).to(device)
    state = PoseState(
        torch.eye(3, device=device)[None],
        torch.zeros(1, 3, device=device),
        torch.full((1,), 0.08, device=device),
    )
    target = PoseState(
        torch.eye(3, device=device)[None],
        torch.tensor([[0.002, 0.0, 0.0]], device=device),
        torch.full((1,), 0.071, device=device),
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
        loss = state_error(prediction, target, length_scale=0.08)[0].mean()
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
