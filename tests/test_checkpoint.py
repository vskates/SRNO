from __future__ import annotations

from pathlib import Path

import torch

from srno.model import SRNOModel
from srno.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_and_scheduler(tmp_path: Path, gripper) -> None:
    torch.manual_seed(12)
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    objective = sum(parameter.square().sum() for parameter in model.parameters())
    objective.backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config={"test": True},
        manifest_sha256="manifest",
        gripper_sha256="gripper",
        stage="local",
        epoch=3,
        horizon=1,
        horizon_epoch=3,
        stale_epochs=0,
        best_metric=0.5,
    )
    restored = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda step: 0.9**step
    )
    checkpoint = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    assert checkpoint["epoch"] == 3
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)
    assert optimizer.state_dict()["param_groups"] == restored_optimizer.state_dict()["param_groups"]
