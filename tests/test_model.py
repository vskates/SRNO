from __future__ import annotations

import pytest
import torch

from srno.geometry.gripper import GripperAsset
from srno.geometry.se3 import apply_left_increment, so3_exp, so3_log_vector
from srno.model import (
    ContactDualPotentialCell,
    ContactImplicitResolventCell,
    ContactIntegralCell,
    ContactMonotoneResolventCell,
    ContactNormalConeCell,
    SRNOModel,
    solve_metric_contact_resolvent,
)
from srno.types import PoseState, SDFBatch


def _state(gripper: GripperAsset, batch: int = 2, aperture: float = 0.08) -> PoseState:
    joints = gripper.free_joint_configuration(torch.tensor(aperture))
    return PoseState(
        torch.eye(3).expand(batch, 3, 3).clone(),
        torch.zeros(batch, 3),
        joints.expand(batch, 6).clone(),
    )


def _sdf(value: float, batch: int = 2) -> SDFBatch:
    return SDFBatch(
        torch.full((1, 8, 8, 8), value),
        torch.full((1, 3), -0.04),
        torch.full((1, 3), 0.01),
        torch.zeros(batch, dtype=torch.long),
        0.02,
    )


def test_exact_free_bypass_does_not_call_cell(gripper: GripperAsset) -> None:
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    calls = 0

    def hook(*_: object) -> None:
        nonlocal calls
        calls += 1

    state = _state(gripper)
    handle = model.contact_cell.register_forward_hook(hook)
    result, aux = model.forward_step(state, 0.07, _sdf(0.02), return_aux=True)
    handle.remove()
    assert calls == 0
    assert not aux.active.any()
    assert torch.equal(result.rotation, state.rotation)
    assert torch.equal(result.position, state.position)
    assert torch.equal(
        result.joint_position,
        model.free_joint_configuration(0.07).expand(2, 6),
    )


def test_active_step_outputs_joint_residual_and_has_finite_backward(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    state = _state(gripper)
    commands = torch.tensor([0.07, 0.06])
    result, aux = model.forward_step(
        state, commands, _sdf(-1e-3), return_aux=True
    )
    assert aux.active.all()
    assert aux.joint_residual.shape == (2, 6)
    assert torch.equal(
        result.joint_position,
        model.free_joint_configuration(commands) + aux.joint_residual,
    )
    loss = (
        result.position.square().sum()
        + result.joint_position.sum()
        + result.rotation.sum()
    )
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_contact_offset_shifts_contact_gap_but_not_geometric_gap(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=0.004,
        contact_offset_sum=0.00256,
    )
    state = _state(gripper, aperture=0.06)
    sdf = _sdf(0.006)

    geometric = model.query_geometric_gap(state, sdf)
    effective = model.query_gap(state, sdf)
    _, aux = model.forward_step(state, 0.06, sdf, return_aux=True)

    assert geometric.amin() == pytest.approx(0.006)
    assert torch.allclose(effective, geometric - 0.00256)
    assert aux.active.all()


def test_integral_cell_is_permutation_invariant_after_pooling() -> None:
    torch.manual_seed(4)
    cell = ContactIntegralCell(16)
    final = cell.head[-1]
    assert isinstance(final, torch.nn.Linear)
    torch.nn.init.normal_(final.weight)
    gap = torch.randn(3, 12)
    points = torch.randn(12, 3)
    aperture = torch.full((3,), 0.5)
    twist, residual = cell(gap, points, aperture, aperture)
    permutation = torch.randperm(12)
    permuted_twist, permuted_residual = cell(
        gap[:, permutation], points[permutation], aperture, aperture
    )
    assert torch.allclose(twist, permuted_twist, atol=1e-6)
    assert torch.allclose(residual, permuted_residual, atol=1e-6)


def test_depth_one_is_strictly_backward_compatible() -> None:
    torch.manual_seed(41)
    implicit = ContactIntegralCell(16)
    explicit = ContactIntegralCell(16, operator_layers=1)
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    gap = torch.randn(2, 9)
    points = torch.randn(9, 3)
    aperture = torch.rand(2)
    implicit_output = implicit(gap, points, aperture, aperture)
    explicit_output = explicit(gap, points, aperture, aperture)
    assert all(
        torch.equal(left, right)
        for left, right in zip(implicit_output, explicit_output, strict=True)
    )


def test_residual_depth_reinjects_geometry_and_preserves_permutation() -> None:
    torch.manual_seed(42)
    depth = 4
    cell = ContactIntegralCell(16, operator_layers=depth)
    final = cell.head[-1]
    assert isinstance(final, torch.nn.Linear)
    torch.nn.init.normal_(final.weight)
    kernel_calls = [0 for _ in range(depth)]

    def count(index: int):
        def hook(_: torch.nn.Module, __: tuple[object, ...], ___: object) -> None:
            kernel_calls[index] += 1

        return hook

    handles = [cell.kernel.register_forward_hook(count(0))]
    handles.extend(
        layer.kernel.register_forward_hook(count(index + 1))
        for index, layer in enumerate(cell.residual_layers)
    )
    gap = torch.randn(2, 11)
    points = torch.randn(11, 3)
    aperture = torch.rand(2)
    output = cell(gap, points, aperture, aperture)
    for handle in handles:
        handle.remove()
    assert kernel_calls == [1, 1, 1, 1]

    permutation = torch.randperm(11)
    permuted = cell(
        gap[:, permutation], points[permutation], aperture, aperture
    )
    assert all(
        torch.allclose(left, right, atol=2e-6)
        for left, right in zip(output, permuted, strict=True)
    )


def test_operator_depth_parameter_count_and_active_backward(
    gripper: GripperAsset,
) -> None:
    models = [
        SRNOModel(
            gripper,
            sdf_scale=0.02,
            delta_gate=0.005,
            operator_layers=depth,
        )
        for depth in range(1, 5)
    ]
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert counts == [31_436, 44_300, 57_164, 70_028]
    final = models[-1].contact_cell.head[-1]
    assert isinstance(final, torch.nn.Linear)
    torch.nn.init.normal_(final.weight)
    result = models[-1].forward_step(
        _state(gripper, batch=1), 0.07, _sdf(-1e-3, batch=1)
    )
    assert isinstance(result, PoseState)
    loss = result.position.square().sum() + result.joint_position.square().sum()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in models[-1].parameters()
    )


def test_jq_integral_cell_is_permutation_invariant_after_pooling() -> None:
    torch.manual_seed(5)
    cell = ContactIntegralCell(16, input_dim=7)
    final = cell.head[-1]
    assert isinstance(final, torch.nn.Linear)
    torch.nn.init.normal_(final.weight)
    features = torch.randn(3, 12, 7)
    points = torch.randn(12, 3)
    aperture = torch.full((3,), 0.5)
    twist, residual = cell(features, points, aperture, aperture)
    permutation = torch.randperm(12)
    permuted_twist, permuted_residual = cell(
        features[:, permutation], points[permutation], aperture, aperture
    )
    assert torch.allclose(twist, permuted_twist, atol=1e-6)
    assert torch.allclose(residual, permuted_residual, atol=1e-6)


def test_normal_cone_cell_is_permutation_invariant_and_has_finite_backward() -> None:
    torch.manual_seed(51)
    cell = ContactNormalConeCell(16)
    features = torch.randn(3, 12, 13)
    jacobian = features[..., 1:].clone()
    points = torch.randn(12, 3)
    support = torch.rand(3, 12) > 0.25
    aperture = torch.full((3,), 0.5)
    output = cell(
        features,
        points,
        aperture,
        aperture,
        contact_jacobian=jacobian,
        contact_support=support,
    )
    permutation = torch.randperm(12)
    permuted = cell(
        features[:, permutation],
        points[permutation],
        aperture,
        aperture,
        contact_jacobian=jacobian[:, permutation],
        contact_support=support[:, permutation],
    )
    assert all(
        torch.allclose(left, right, atol=2e-6)
        for left, right in zip(output, permuted, strict=True)
    )
    sum(part.square().sum() for part in output).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in cell.parameters()
    )


def test_32_step_free_rollout_is_exact(gripper: GripperAsset) -> None:
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    schedule = torch.linspace(0.08, 0.0, 33)[1:]
    rollout = model.rollout(_state(gripper), schedule, _sdf(0.02))
    assert rollout.rotation.shape == (2, 33, 3, 3)
    assert rollout.joint_position.shape == (2, 33, 6)
    assert torch.equal(rollout.position, torch.zeros(2, 33, 3))
    expected = model.free_joint_configuration(schedule).expand(2, 32, 6)
    assert torch.allclose(rollout.joint_position[:, 1:], expected)
    assert torch.allclose(
        model.aperture_from_joints(rollout.joint_position[:, 1:]),
        schedule.expand(2, 32),
        atol=1e-6,
    )


def test_pose_delta_history_conditions_active_step_and_rollout(
    gripper: GripperAsset,
) -> None:
    torch.manual_seed(43)
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=1.0,
        pose_update="decoupled",
        history_conditioning="pose_delta",
    )
    final = model.contact_cell.head[-1]
    assert isinstance(final, torch.nn.Linear)
    torch.nn.init.normal_(final.weight)
    current = _state(gripper, batch=1)
    same_previous = current
    moved_previous = PoseState(
        so3_exp(torch.tensor([[0.02, -0.01, 0.03]])).transpose(-1, -2),
        current.position - torch.tensor([[0.001, -0.002, 0.003]]),
        current.joint_position,
    )
    stationary = model.forward_step(
        current,
        0.07,
        _sdf(-1e-3, batch=1),
        previous_state=same_previous,
    )
    moving = model.forward_step(
        current,
        0.07,
        _sdf(-1e-3, batch=1),
        previous_state=moved_previous,
    )
    assert isinstance(stationary, PoseState) and isinstance(moving, PoseState)
    assert not torch.allclose(stationary.position, moving.position)
    rollout = model.rollout(
        current,
        torch.tensor([0.07, 0.06]),
        _sdf(-1e-3, batch=1),
    )
    assert rollout.shape == (1, 3)

    with pytest.raises(ValueError, match="requires previous_state"):
        model.forward_step(current, 0.07, _sdf(-1e-3, batch=1))


def test_model_geometry_depends_on_actual_joints(gripper: GripperAsset) -> None:
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    free = model.free_joint_configuration(torch.tensor([0.08, 0.04]))
    contact_stalled = free.clone()
    contact_stalled[1, 0] = contact_stalled[0, 0]
    scheduled_points = model.gripper_points(free)
    actual_points = model.gripper_points(contact_stalled)
    assert not torch.equal(actual_points[1], scheduled_points[1])
    assert torch.equal(actual_points[0], scheduled_points[0])


def test_finite_backward_through_32_active_steps(gripper: GripperAsset) -> None:
    model = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    schedule = torch.linspace(0.08, 0.0, 33)[1:]
    rollout = model.rollout(_state(gripper, batch=1), schedule, _sdf(-1e-3, 1))
    loss = rollout.position.square().sum() + rollout.joint_position[:, -1].sum()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_gap_jq_finite_backward_through_32_active_steps(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=0.005,
        contact_features="gap_jq",
    )
    schedule = torch.linspace(0.08, 0.0, 33)[1:]
    sdf, _ = _linear_signed_distance_sdf()
    rollout = model.rollout(_state(gripper, batch=1), schedule, sdf)
    loss = rollout.position.square().sum() + rollout.joint_position[:, -1].sum()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_trial_current_gap_exposes_actual_joint_contact_geometry(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=1.0,
        contact_features="trial_current_gap",
    )
    state = _state(gripper)
    asymmetric_joints = state.joint_position.clone()
    asymmetric_joints[1, 0] += 0.08
    state = PoseState(state.rotation, state.position, asymmetric_joints)
    sdf, _ = _linear_signed_distance_sdf(batch=2)
    captured: list[torch.Tensor] = []

    def capture(_: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        features = inputs[0]
        assert isinstance(features, torch.Tensor)
        captured.append(features.detach().clone())

    handle = model.contact_cell.register_forward_pre_hook(capture)
    model.forward_step(state, 0.07, sdf)
    handle.remove()
    assert len(captured) == 1
    features = captured[0]
    assert features.shape == (2, 256, 2)
    # The trial command and object pose are identical across samples.
    assert torch.equal(features[0, :, 0], features[1, :, 0])
    # Only the actual joint state differs, and the new channel observes it.
    assert not torch.equal(features[0, :, 1], features[1, :, 1])


def test_decoupled_pose_update_predicts_direct_translation(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=1.0,
        pose_update="decoupled",
    )
    final = model.contact_cell.head[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.bias[:6] = torch.tensor([0.1, -0.2, 0.3, 0.0, 0.0, 0.1])
    state = _state(gripper, batch=1)
    state = PoseState(
        state.rotation,
        torch.tensor([[0.4, -0.1, 0.2]]),
        state.joint_position,
    )
    prediction = model.forward_step(state, 0.07, _sdf(-1e-3, batch=1))
    assert isinstance(prediction, PoseState)
    expected_position = state.position + model.length_scale * torch.tensor(
        [[0.1, -0.2, 0.3]]
    )
    assert torch.allclose(prediction.position, expected_position, atol=1e-7)
    assert torch.allclose(
        prediction.rotation,
        so3_exp(torch.tensor([[0.0, 0.0, 0.1]])) @ state.rotation,
        atol=1e-7,
    )


def test_default_gap_mode_strict_loads_legacy_state_dict_without_output_change(
    gripper: GripperAsset,
) -> None:
    torch.manual_seed(23)
    legacy = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005).eval()
    restored = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=0.005,
        contact_features="gap",
    ).eval()
    restored.load_state_dict(legacy.state_dict(), strict=True)
    state = _state(gripper)
    sdf = _sdf(-1e-3)
    with torch.no_grad():
        legacy_result = legacy.forward_step(state, 0.07, sdf)
        restored_result = restored.forward_step(state, 0.07, sdf)
    assert isinstance(legacy_result, PoseState)
    assert isinstance(restored_result, PoseState)
    assert torch.equal(restored_result.rotation, legacy_result.rotation)
    assert torch.equal(restored_result.position, legacy_result.position)
    assert torch.equal(restored_result.joint_position, legacy_result.joint_position)


def _linear_signed_distance_sdf(batch: int = 1) -> tuple[SDFBatch, torch.Tensor]:
    normal = torch.tensor([0.3, -0.4, 0.5])
    normal = normal / torch.linalg.vector_norm(normal)
    origin = torch.full((1, 3), -0.4)
    voxel = torch.full((1, 3), 0.05)
    z, y, x = torch.meshgrid(
        torch.arange(17.0), torch.arange(17.0), torch.arange(17.0), indexing="ij"
    )
    coordinates = torch.stack(
        (
            origin[0, 0] + x * voxel[0, 0],
            origin[0, 1] + y * voxel[0, 1],
            origin[0, 2] + z * voxel[0, 2],
        ),
        dim=-1,
    )
    values = torch.einsum("zyxc,c->zyx", coordinates, normal)[None]
    return (
        SDFBatch(
            values,
            origin,
            voxel,
            torch.zeros(batch, dtype=torch.long),
            0.02,
        ),
        normal,
    )


def test_jq_matches_all_six_left_spatial_finite_differences(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=0.005,
        contact_features="gap_jq",
    )
    rotation = so3_exp(torch.tensor([[0.2, -0.1, 0.15]]))
    position = torch.tensor([[0.015, -0.01, 0.02]])
    joint = model.free_joint_configuration(torch.tensor(0.05))[None]
    state = PoseState(rotation, position, joint)
    sdf, _ = _linear_signed_distance_sdf()
    points = model.gripper_points(joint)
    _, normalized_jacobian = model._contact_gap_and_pose_jacobian(state, points, sdf)
    raw_jacobian = torch.cat(
        (
            normalized_jacobian[..., :3],
            normalized_jacobian[..., 3:] * model.length_scale,
        ),
        dim=-1,
    )

    epsilon = 1e-4
    finite_difference = []
    for component in range(6):
        increment = torch.zeros(1, 6)
        increment[:, component] = epsilon
        plus_rotation, plus_position = apply_left_increment(
            state.rotation, state.position, increment
        )
        minus_rotation, minus_position = apply_left_increment(
            state.rotation, state.position, -increment
        )
        plus = model.query_geometric_gap(
            PoseState(plus_rotation, plus_position, joint), sdf
        )
        minus = model.query_geometric_gap(
            PoseState(minus_rotation, minus_position, joint), sdf
        )
        finite_difference.append((plus - minus) / (2.0 * epsilon))
    finite_difference_tensor = torch.stack(finite_difference, dim=-1)
    assert torch.allclose(
        normalized_jacobian[..., :3], raw_jacobian[..., :3], atol=1e-7
    )
    assert torch.allclose(finite_difference_tensor, raw_jacobian, atol=3e-4)


def test_normal_cone_full_jacobian_matches_all_product_finite_differences(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=0.005,
        contact_head="normal_cone",
        pose_update="decoupled",
    )
    rotation = so3_exp(torch.tensor([[0.2, -0.1, 0.15]]))
    position = torch.tensor([[0.015, -0.01, 0.02]])
    joint = model.free_joint_configuration(torch.tensor(0.05))[None]
    state = PoseState(rotation, position, joint)
    sdf, _ = _linear_signed_distance_sdf()
    _, _, jacobian = model._contact_gap_and_full_jacobian(state, sdf)

    epsilon = 1e-4
    finite_difference = []
    for component in range(12):
        increment = torch.zeros(1, 12)
        increment[:, component] = epsilon

        def displaced(value: torch.Tensor) -> PoseState:
            return PoseState(
                so3_exp(value[:, 3:6]) @ state.rotation,
                state.position + value[:, :3] * model.length_scale,
                state.joint_position + value[:, 6:] * model.joint_travel_range,
            )

        plus = model.query_geometric_gap(displaced(increment), sdf)
        minus = model.query_geometric_gap(displaced(-increment), sdf)
        finite_difference.append(
            (plus - minus) / (2.0 * epsilon * model.sdf_scale)
        )
    finite_difference_tensor = torch.stack(finite_difference, dim=-1)
    # Trilinear SDF finite differences are evaluated in float32; an epsilon
    # small enough for the SO(3) derivative leaves a few 1e-3 interpolation
    # round-off residuals after division by sdf_scale.
    absolute_error = (finite_difference_tensor - jacobian).abs()
    assert float(absolute_error.max()) < 1.5e-2, absolute_error.amax(dim=(0, 1))


def test_normal_cone_model_active_step_has_finite_backward(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=1.0,
        contact_head="normal_cone",
        pose_update="decoupled",
    )
    sdf, _ = _linear_signed_distance_sdf(batch=2)
    result = model.forward_step(_state(gripper), torch.tensor([0.07, 0.06]), sdf)
    assert isinstance(result, PoseState)
    loss = (
        result.position.square().sum()
        + result.rotation.square().sum()
        + result.joint_position.square().sum()
    )
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_metric_contact_resolvent_projects_onto_half_space() -> None:
    query = torch.zeros(1, 12)
    query[:, 0] = -1.0
    factor = torch.eye(12)[None]
    jacobian = torch.zeros(1, 1, 12)
    jacobian[:, :, 0] = 1.0
    solution, dual, slack = solve_metric_contact_resolvent(
        query,
        factor,
        jacobian,
        torch.zeros(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        iterations=8,
    )
    assert torch.allclose(solution[:, 0], torch.zeros(1), atol=1e-6)
    assert torch.allclose(solution[:, 1:], query[:, 1:], atol=1e-6)
    assert torch.allclose(dual, torch.ones_like(dual), atol=1e-6)
    assert torch.allclose(slack, torch.zeros_like(slack), atol=1e-6)


def test_implicit_resolvent_model_has_finite_backward(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=1.0,
        contact_head="implicit_resolvent",
        actuation_conditioning="drive_error",
        pose_update="decoupled",
    )
    assert isinstance(model.contact_cell, ContactImplicitResolventCell)
    sdf, _ = _linear_signed_distance_sdf(batch=2)
    result, aux = model.forward_step(
        _state(gripper),
        torch.tensor([0.07, 0.06]),
        sdf,
        return_aux=True,
    )
    assert isinstance(result, PoseState)
    assert aux.active.all()
    assert torch.allclose(
        result.joint_position,
        model.free_joint_configuration(torch.tensor([0.07, 0.06]))
        + aux.joint_residual,
        atol=1e-7,
    )
    loss = (
        result.position.square().sum()
        + result.rotation.square().sum()
        + result.joint_position.square().sum()
    )
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_implicit_resolvent_uses_previous_increment_as_single_qp_query(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=1.0,
        contact_head="implicit_resolvent",
        actuation_conditioning="drive_error",
        pose_update="decoupled",
        history_conditioning="pose_delta",
        resolvent_pose_query_factor=0.5,
    )
    current = _state(gripper, batch=1)
    current = PoseState(
        so3_exp(torch.tensor([[0.02, -0.01, 0.03]])),
        torch.tensor([[0.002, -0.001, 0.003]]),
        current.joint_position,
    )
    previous = PoseState(
        torch.eye(3)[None],
        torch.zeros(1, 3),
        current.joint_position,
    )
    # A constant SDF activates the branch but has zero normal/Jacobian, so the
    # projection leaves its pose query unchanged and isolates the query law.
    result = model.forward_step(
        current,
        0.07,
        _sdf(-1e-3, batch=1),
        previous_state=previous,
    )
    assert isinstance(result, PoseState)
    expected_position = current.position + 0.5 * (
        current.position - previous.position
    )
    incoming = current.rotation @ previous.rotation.transpose(-1, -2)
    expected_rotation = so3_exp(0.5 * so3_log_vector(incoming)) @ current.rotation
    assert torch.allclose(result.position, expected_position, atol=1e-7)
    assert torch.allclose(result.rotation, expected_rotation, atol=1e-6)


def test_dual_potential_is_zero_at_zero_force_and_firmly_nonexpansive() -> None:
    cell = ContactDualPotentialCell(hidden_dim=16)
    node = torch.zeros(2, 8, 13)
    points = torch.zeros(2, 8, 3)
    aperture = torch.full((2,), 0.5)
    drives = torch.tensor(
        [
            [0.02, -0.01, 0.03, 0.01, -0.02, 0.04],
            [-0.01, 0.03, 0.01, -0.02, 0.02, 0.01],
        ]
    )
    pose, joint = cell(
        node,
        points,
        aperture,
        drive_features=drives,
        pose_features=torch.zeros(2, 6),
        history_features=None,
    )
    zero_pose, zero_joint = cell(
        node[:1],
        points[:1],
        aperture[:1],
        drive_features=torch.zeros(1, 6),
        pose_features=torch.zeros(1, 6),
        history_features=None,
    )
    assert torch.equal(zero_pose, torch.zeros_like(zero_pose))
    assert torch.equal(zero_joint, torch.zeros_like(zero_joint))

    force_difference = torch.cat(
        (torch.zeros(1, 6), drives[0:1] - drives[1:2]), dim=-1
    )
    motion_difference = torch.cat(
        (pose[0:1] - pose[1:2], joint[0:1] - joint[1:2]), dim=-1
    )
    squared_norm = motion_difference.square().sum(dim=-1)
    dual_pairing = (motion_difference * force_difference).sum(dim=-1)
    assert torch.all(squared_norm <= dual_pairing + 1e-7)


def test_dual_potential_model_active_step_has_finite_backward(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=1.0,
        contact_head="dual_potential",
        actuation_conditioning="drive_error",
        pose_update="decoupled",
        history_conditioning="pose_delta",
    )
    assert isinstance(model.contact_cell, ContactDualPotentialCell)
    state = _state(gripper)
    result = model.forward_step(
        state,
        torch.tensor([0.07, 0.06]),
        _linear_signed_distance_sdf(batch=2)[0],
        previous_state=state,
    )
    assert isinstance(result, PoseState)
    loss = (
        result.position.square().sum()
        + result.rotation.square().sum()
        + result.joint_position.square().sum()
    )
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_monotone_resolvent_is_zero_at_zero_force_and_firmly_nonexpansive() -> None:
    cell = ContactMonotoneResolventCell(hidden_dim=16)
    final = cell.operator_head[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.bias.copy_(torch.linspace(-0.4, 0.4, final.bias.numel()))
    node = torch.zeros(2, 8, 13)
    points = torch.zeros(2, 8, 3)
    aperture = torch.full((2,), 0.5)
    drives = torch.tensor(
        [
            [0.02, -0.01, 0.03, 0.01, -0.02, 0.04],
            [-0.01, 0.03, 0.01, -0.02, 0.02, 0.01],
        ]
    )
    pose, joint = cell(
        node,
        points,
        aperture,
        drive_features=drives,
        pose_features=torch.zeros(2, 6),
        history_features=None,
    )
    zero_pose, zero_joint = cell(
        node[:1],
        points[:1],
        aperture[:1],
        drive_features=torch.zeros(1, 6),
        pose_features=torch.zeros(1, 6),
        history_features=None,
    )
    assert torch.equal(zero_pose, torch.zeros_like(zero_pose))
    assert torch.equal(zero_joint, torch.zeros_like(zero_joint))

    force_difference = torch.cat(
        (torch.zeros(1, 6), drives[0:1] - drives[1:2]), dim=-1
    )
    motion_difference = torch.cat(
        (pose[0:1] - pose[1:2], joint[0:1] - joint[1:2]), dim=-1
    )
    squared_norm = motion_difference.square().sum(dim=-1)
    dual_pairing = (motion_difference * force_difference).sum(dim=-1)
    assert torch.all(squared_norm <= dual_pairing + 1e-6)


def test_monotone_resolvent_model_active_step_has_finite_backward(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=1.0,
        contact_head="monotone_resolvent",
        actuation_conditioning="drive_error",
        pose_update="decoupled",
        history_conditioning="pose_delta",
    )
    assert isinstance(model.contact_cell, ContactMonotoneResolventCell)
    state = _state(gripper)
    result = model.forward_step(
        state,
        torch.tensor([0.07, 0.06]),
        _linear_signed_distance_sdf(batch=2)[0],
        previous_state=state,
    )
    assert isinstance(result, PoseState)
    loss = (
        result.position.square().sum()
        + result.rotation.square().sum()
        + result.joint_position.square().sum()
    )
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_friction_frame_normal_column_matches_full_jacobian(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=0.005,
        contact_head="friction_cone",
        pose_update="decoupled",
    )
    state = _state(gripper, batch=1, aperture=0.05)
    sdf, _ = _linear_signed_distance_sdf()
    _, points, jacobian = model._contact_gap_and_full_jacobian(state, sdf)
    directions = model._friction_cone_directions(state, points, jacobian)
    assert directions.shape == (1, 256, 12, 3)
    assert torch.allclose(directions[..., 0], jacobian, atol=2e-6)


def test_continuation_predictor_only_affects_active_contact_branch(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=1.0,
        contact_head="normal_cone",
        pose_update="decoupled",
        pose_predictor="continuation",
        continuation_factor=0.75,
    )
    assert isinstance(model.contact_cell, ContactNormalConeCell)
    pressure_final = model.contact_cell.pressure_head[-1]
    assert isinstance(pressure_final, torch.nn.Linear)
    with torch.no_grad():
        pressure_final.weight.zero_()
        pressure_final.bias.fill_(-30.0)
    current = _state(gripper, batch=1)
    current = PoseState(
        so3_exp(torch.tensor([[0.02, -0.01, 0.03]])),
        torch.tensor([[0.002, -0.001, 0.003]]),
        current.joint_position,
    )
    previous = PoseState(
        torch.eye(3)[None],
        torch.zeros(1, 3),
        current.joint_position,
    )
    active = model.forward_step(
        current,
        0.07,
        _sdf(-1e-3, batch=1),
        previous_state=previous,
    )
    assert isinstance(active, PoseState)
    incoming = current.rotation @ previous.rotation.transpose(-1, -2)
    expected_rotation = (
        so3_exp(0.75 * so3_log_vector(incoming)) @ current.rotation
    )
    expected_position = current.position + 0.75 * (
        current.position - previous.position
    )
    assert torch.allclose(active.rotation, expected_rotation, atol=1e-6)
    assert torch.allclose(active.position, expected_position, atol=1e-6)

    rollout = model.rollout(
        current,
        torch.tensor([0.07, 0.065]),
        _sdf(-1e-3, batch=1),
    )
    assert rollout.position.shape == (1, 3, 3)

    inactive_model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=0.005,
        contact_head="normal_cone",
        pose_update="decoupled",
        pose_predictor="continuation",
        continuation_factor=0.75,
    )
    inactive = inactive_model.forward_step(
        current,
        0.07,
        _sdf(0.02, batch=1),
        previous_state=previous,
    )
    assert isinstance(inactive, PoseState)
    assert torch.equal(inactive.rotation, current.rotation)
    assert torch.equal(inactive.position, current.position)


def test_gap_jq_zero_gradient_is_finite_and_preserves_parameter_delta(
    gripper: GripperAsset,
) -> None:
    baseline = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005)
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=0.005,
        contact_features="gap_jq",
    )
    assert sum(p.numel() for p in baseline.parameters()) == 31_436
    assert sum(p.numel() for p in model.parameters()) == 32_204
    result = model.forward_step(_state(gripper), 0.07, _sdf(-1e-3))
    assert isinstance(result, PoseState)
    loss = result.position.square().sum() + result.joint_position.sum()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_contact_outputs_agree(gripper: GripperAsset) -> None:
    torch.manual_seed(7)
    cpu = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005).eval()
    gpu = SRNOModel(gripper, sdf_scale=0.02, delta_gate=0.005).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    with torch.no_grad():
        cpu_result = cpu.forward_step(_state(gripper, 1), 0.07, _sdf(-1e-3, 1))
        gpu_result = gpu.forward_step(
            _state(gripper, 1).to("cuda"),
            0.07,
            _sdf(-1e-3, 1).to(device="cuda"),
        )
    assert isinstance(cpu_result, PoseState) and isinstance(gpu_result, PoseState)
    assert torch.allclose(cpu_result.position, gpu_result.position.cpu(), atol=2e-5)
    assert torch.allclose(
        cpu_result.joint_position, gpu_result.joint_position.cpu(), atol=2e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gap_jq_keeps_geometry_float32_under_bfloat16_autocast(
    gripper: GripperAsset,
) -> None:
    model = SRNOModel(
        gripper,
        sdf_scale=0.02,
        delta_gate=0.005,
        contact_features="gap_jq",
    ).cuda()
    state = _state(gripper, 1).to("cuda")
    sdf, _ = _linear_signed_distance_sdf()
    sdf = sdf.to(device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result, aux = model.forward_step(state, 0.07, sdf, return_aux=True)
    assert torch.isfinite(result.position).all()
    assert aux.trial_gap.dtype == torch.float32
