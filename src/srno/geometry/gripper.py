from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from srno.geometry.se3 import so3_exp


@dataclass(frozen=True)
class GripperAsset:
    """Collision samples, free schedule, and optional differentiable joint FK."""

    intercept: Tensor
    slope: Tensor
    link_index: Tensor
    aperture_min: float
    aperture_max: float
    length_scale: float
    source_sha256: str = ""
    aperture_knots: Tensor | None = None
    point_knots: Tensor | None = None
    joint_names: tuple[str, ...] = ()
    free_joint_knots: Tensor | None = None
    local_points: Tensor | None = None
    link_pivots: Tensor | None = None
    link_open_positions: Tensor | None = None
    link_open_rotations: Tensor | None = None
    link_axes: Tensor | None = None
    link_position_joint_coefficients: Tensor | None = None
    link_rotation_joint_coefficients: Tensor | None = None

    def __post_init__(self) -> None:
        if self.intercept.ndim != 2 or self.intercept.shape[-1] != 3:
            raise ValueError("intercept must have shape [points, 3]")
        if self.slope.shape != self.intercept.shape:
            raise ValueError("slope must have the same shape as intercept")
        if self.link_index.shape != self.intercept.shape[:1]:
            raise ValueError("link_index must have shape [points]")
        if self.aperture_min >= self.aperture_max:
            raise ValueError("aperture_min must be less than aperture_max")
        if self.length_scale <= 0:
            raise ValueError("length_scale must be positive")
        if self.point_knots is not None and self.aperture_knots is None:
            raise ValueError("point_knots require aperture_knots")
        if self.aperture_knots is not None:
            if self.aperture_knots.ndim != 1 or len(self.aperture_knots) < 2:
                raise ValueError("aperture_knots must have shape [states] with at least two states")
            if self.point_knots is not None and self.point_knots.shape != (
                len(self.aperture_knots),
                self.point_count,
                3,
            ):
                raise ValueError("point_knots must have shape [states, points, 3]")
            if not torch.all(self.aperture_knots[1:] > self.aperture_knots[:-1]):
                raise ValueError("aperture_knots must be strictly increasing")
            if not torch.isclose(
                self.aperture_knots[0],
                self.aperture_knots.new_tensor(self.aperture_min),
                atol=1e-7,
                rtol=0.0,
            ) or not torch.isclose(
                self.aperture_knots[-1],
                self.aperture_knots.new_tensor(self.aperture_max),
                atol=1e-7,
                rtol=0.0,
            ):
                raise ValueError("aperture knots must span the declared gripper limits")
        joint_fields = (
            self.free_joint_knots,
            self.local_points,
            self.link_pivots,
            self.link_open_positions,
            self.link_open_rotations,
            self.link_axes,
            self.link_position_joint_coefficients,
            self.link_rotation_joint_coefficients,
        )
        has_joint_fk = any(value is not None for value in joint_fields) or bool(
            self.joint_names
        )
        if has_joint_fk and not all(value is not None for value in joint_fields):
            raise ValueError("joint FK fields must be provided together")
        if has_joint_fk:
            assert self.free_joint_knots is not None
            assert self.local_points is not None
            assert self.link_pivots is not None
            assert self.link_open_positions is not None
            assert self.link_open_rotations is not None
            assert self.link_axes is not None
            assert self.link_position_joint_coefficients is not None
            assert self.link_rotation_joint_coefficients is not None
            if self.aperture_knots is None:
                raise ValueError("joint FK requires aperture_knots")
            joints = len(self.joint_names)
            links = int(self.link_pivots.shape[0])
            if joints != 6 or len(set(self.joint_names)) != 6:
                raise ValueError("SRNO-r requires six unique joint names")
            if self.free_joint_knots.shape != (len(self.aperture_knots), joints):
                raise ValueError("free_joint_knots must have shape [states, joints]")
            if self.local_points.shape != self.intercept.shape:
                raise ValueError("local_points must have shape [points, 3]")
            if self.link_pivots.shape != (links, 3):
                raise ValueError("link_pivots must have shape [links, 3]")
            if self.link_open_positions.shape != (links, 3):
                raise ValueError("link_open_positions must have shape [links, 3]")
            if self.link_open_rotations.shape != (links, 3, 3):
                raise ValueError("link_open_rotations must have shape [links, 3, 3]")
            if self.link_axes.shape != (links, 3):
                raise ValueError("link_axes must have shape [links, 3]")
            if self.link_position_joint_coefficients.shape != (links, joints):
                raise ValueError(
                    "link_position_joint_coefficients must have shape [links, joints]"
                )
            if self.link_rotation_joint_coefficients.shape != (links, joints):
                raise ValueError(
                    "link_rotation_joint_coefficients must have shape [links, joints]"
                )
            if self.link_index.numel() and (
                int(self.link_index.min()) < 0 or int(self.link_index.max()) >= links
            ):
                raise ValueError("link_index refers to an unknown FK link")
            axis_norm = torch.linalg.vector_norm(self.link_axes, dim=-1)
            if not torch.allclose(axis_norm, torch.ones_like(axis_norm), atol=1e-6):
                raise ValueError("link_axes must be unit vectors")

    @property
    def point_count(self) -> int:
        return self.intercept.shape[0]

    @property
    def supports_joint_fk(self) -> bool:
        return self.free_joint_knots is not None

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    @property
    def joint_travel_range(self) -> Tensor:
        if self.free_joint_knots is None:
            raise ValueError("gripper asset has no joint FK")
        return torch.abs(self.free_joint_knots[0] - self.free_joint_knots[-1])

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "GripperAsset":
        return GripperAsset(
            self.intercept.to(device=device, dtype=dtype),
            self.slope.to(device=device, dtype=dtype),
            self.link_index.to(device=device),
            self.aperture_min,
            self.aperture_max,
            self.length_scale,
            self.source_sha256,
            None
            if self.aperture_knots is None
            else self.aperture_knots.to(device=device, dtype=dtype),
            None
            if self.point_knots is None
            else self.point_knots.to(device=device, dtype=dtype),
            self.joint_names,
            None
            if self.free_joint_knots is None
            else self.free_joint_knots.to(device=device, dtype=dtype),
            None
            if self.local_points is None
            else self.local_points.to(device=device, dtype=dtype),
            None
            if self.link_pivots is None
            else self.link_pivots.to(device=device, dtype=dtype),
            None
            if self.link_open_positions is None
            else self.link_open_positions.to(device=device, dtype=dtype),
            None
            if self.link_open_rotations is None
            else self.link_open_rotations.to(device=device, dtype=dtype),
            None
            if self.link_axes is None
            else self.link_axes.to(device=device, dtype=dtype),
            None
            if self.link_position_joint_coefficients is None
            else self.link_position_joint_coefficients.to(device=device, dtype=dtype),
            None
            if self.link_rotation_joint_coefficients is None
            else self.link_rotation_joint_coefficients.to(device=device, dtype=dtype),
        )

    def _interpolate_aperture_values(
        self, aperture: Tensor | float, values: Tensor
    ) -> Tensor:
        if self.aperture_knots is None:
            raise ValueError("gripper asset has no aperture schedule")
        aperture_tensor = torch.as_tensor(
            aperture, dtype=self.intercept.dtype, device=self.intercept.device
        )
        if torch.any(aperture_tensor < self.aperture_min - 1e-7) or torch.any(
            aperture_tensor > self.aperture_max + 1e-7
        ):
            raise ValueError("aperture is outside gripper limits")
        knots = self.aperture_knots.to(device=aperture_tensor.device, dtype=aperture_tensor.dtype)
        scheduled = values.to(device=aperture_tensor.device, dtype=aperture_tensor.dtype)
        flat = aperture_tensor.reshape(-1).contiguous()
        upper = torch.searchsorted(knots, flat).clamp(1, len(knots) - 1)
        lower = upper - 1
        low_aperture = knots.index_select(0, lower)
        high_aperture = knots.index_select(0, upper)
        alpha = (flat - low_aperture) / (high_aperture - low_aperture)
        low_values = scheduled.index_select(0, lower)
        high_values = scheduled.index_select(0, upper)
        value_rank = scheduled.ndim - 1
        alpha_shape = (len(flat),) + (1,) * value_rank
        interpolated = low_values + alpha.reshape(alpha_shape) * (
            high_values - low_values
        )
        knot_distance = torch.abs(flat[:, None] - knots[None, :])
        exact_distance, exact_index = knot_distance.min(dim=-1)
        exact = exact_distance <= 4.0 * torch.finfo(flat.dtype).eps
        exact_values = scheduled.index_select(0, exact_index)
        exact_shape = (len(flat),) + (1,) * value_rank
        interpolated = torch.where(
            exact.reshape(exact_shape), exact_values, interpolated
        )
        return interpolated.reshape(aperture_tensor.shape + scheduled.shape[1:])

    def free_joint_configuration(self, aperture: Tensor | float) -> Tensor:
        """Lookup/interpolate the deterministic empty-gripper joint schedule."""

        if self.free_joint_knots is None:
            raise ValueError("gripper asset has no free joint schedule")
        return self._interpolate_aperture_values(aperture, self.free_joint_knots)

    def aperture_from_joints(self, joint_position: Tensor) -> Tensor:
        """Return the scalar aperture diagnostic A(r) used by the collector."""

        if self.free_joint_knots is None or self.aperture_knots is None:
            raise ValueError("gripper asset has no joint FK")
        joints = torch.as_tensor(
            joint_position, dtype=self.intercept.dtype, device=self.intercept.device
        )
        if joints.shape[-1:] != (self.joint_count,):
            raise ValueError("joint_position has the wrong final dimension")
        open_joint = self.free_joint_knots[-1]
        close_joint = self.free_joint_knots[0]
        joint_range = close_joint - open_joint
        denominator = joint_range.square().sum().clamp_min(1e-8)
        closure = (
            ((joints - open_joint) * joint_range).sum(dim=-1) / denominator
        ).clamp(0.0, 1.0)
        schedule_position = closure * float(len(self.aperture_knots) - 1)
        lower = torch.floor(schedule_position).long()
        upper = (lower + 1).clamp(max=len(self.aperture_knots) - 1)
        alpha = schedule_position - lower.to(schedule_position.dtype)
        open_to_closed = torch.flip(self.aperture_knots, dims=(0,))
        return (
            open_to_closed.index_select(0, lower.reshape(-1)).reshape(lower.shape)
            * (1.0 - alpha)
            + open_to_closed.index_select(0, upper.reshape(-1)).reshape(upper.shape)
            * alpha
        )

    def points_from_joints(self, joint_position: Tensor) -> Tensor:
        """Transform each local collision sample through differentiable FK."""

        if not self.supports_joint_fk:
            raise ValueError("gripper asset has no joint FK")
        assert self.local_points is not None
        assert self.link_pivots is not None
        assert self.link_open_positions is not None
        assert self.link_open_rotations is not None
        assert self.link_axes is not None
        assert self.link_position_joint_coefficients is not None
        assert self.link_rotation_joint_coefficients is not None
        joints = torch.as_tensor(
            joint_position, dtype=self.intercept.dtype, device=self.intercept.device
        )
        if joints.shape[-1:] != (self.joint_count,):
            raise ValueError("joint_position has the wrong final dimension")
        position_angle = torch.einsum(
            "...j,lj->...l", joints, self.link_position_joint_coefficients
        )
        rotation_angle = torch.einsum(
            "...j,lj->...l", joints, self.link_rotation_joint_coefficients
        )
        position_rotation = so3_exp(
            position_angle[..., None] * self.link_axes
        )
        link_rotation = (
            so3_exp(rotation_angle[..., None] * self.link_axes)
            @ self.link_open_rotations
        )
        open_offset = self.link_open_positions - self.link_pivots
        link_position = (
            torch.einsum("...lij,lj->...li", position_rotation, open_offset)
            + self.link_pivots
        )
        point_rotation = link_rotation[..., self.link_index, :, :]
        point_position = link_position[..., self.link_index, :]
        return (
            torch.einsum("...mij,mj->...mi", point_rotation, self.local_points)
            + point_position
        )

    def points(self, aperture: Tensor | float) -> Tensor:
        """Return empty-gripper points; runtime SRNO-r uses joint FK internally."""

        if self.supports_joint_fk:
            return self.points_from_joints(self.free_joint_configuration(aperture))
        aperture_tensor = torch.as_tensor(
            aperture, dtype=self.intercept.dtype, device=self.intercept.device
        )
        if self.aperture_knots is None or self.point_knots is None:
            return self.intercept + aperture_tensor[..., None, None] * self.slope
        return self._interpolate_aperture_values(aperture_tensor, self.point_knots)

    def pair_features(self, aperture: Tensor | float) -> Tensor:
        points = self.points(aperture) / self.length_scale
        target = points.unsqueeze(-2).expand(points.shape[:-2] + (self.point_count, self.point_count, 3))
        source = points.unsqueeze(-3).expand_as(target)
        return torch.cat((target, source), dim=-1)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        format_version = (
            3 if self.supports_joint_fk else 2 if self.aperture_knots is not None else 1
        )
        metadata = {
            "format_version": format_version,
            "aperture_min": self.aperture_min,
            "aperture_max": self.aperture_max,
            "length_scale": self.length_scale,
            "source_sha256": self.source_sha256,
            "joint_names": list(self.joint_names),
        }
        arrays: dict[str, np.ndarray] = {
            "intercept": self.intercept.detach().cpu().numpy().astype(np.float32),
            "slope": self.slope.detach().cpu().numpy().astype(np.float32),
            "link_index": self.link_index.detach().cpu().numpy().astype(np.int16),
            "metadata": np.asarray(json.dumps(metadata)),
        }
        if self.aperture_knots is not None:
            arrays["aperture_knots"] = (
                self.aperture_knots.detach().cpu().numpy().astype(np.float32)
            )
        if self.point_knots is not None:
            arrays["point_knots"] = self.point_knots.detach().cpu().numpy().astype(np.float32)
        if self.supports_joint_fk:
            joint_arrays = {
                "free_joint_knots": self.free_joint_knots,
                "local_points": self.local_points,
                "link_pivots": self.link_pivots,
                "link_open_positions": self.link_open_positions,
                "link_open_rotations": self.link_open_rotations,
                "link_axes": self.link_axes,
                "link_position_joint_coefficients": self.link_position_joint_coefficients,
                "link_rotation_joint_coefficients": self.link_rotation_joint_coefficients,
            }
            for name, tensor in joint_arrays.items():
                assert tensor is not None
                arrays[name] = tensor.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "GripperAsset":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            version = int(metadata.get("format_version", 0))
            if version not in (1, 2, 3):
                raise ValueError("unsupported gripper asset version")
            scheduled = version >= 2
            joint_fk = version == 3
            return cls(
                torch.from_numpy(archive["intercept"].copy()),
                torch.from_numpy(archive["slope"].copy()),
                torch.from_numpy(archive["link_index"].copy()).long(),
                float(metadata["aperture_min"]),
                float(metadata["aperture_max"]),
                float(metadata["length_scale"]),
                str(metadata.get("source_sha256", "")),
                torch.from_numpy(archive["aperture_knots"].copy()) if scheduled else None,
                (
                    torch.from_numpy(archive["point_knots"].copy())
                    if "point_knots" in archive
                    else None
                ),
                tuple(map(str, metadata.get("joint_names", ()))) if joint_fk else (),
                torch.from_numpy(archive["free_joint_knots"].copy()) if joint_fk else None,
                torch.from_numpy(archive["local_points"].copy()) if joint_fk else None,
                torch.from_numpy(archive["link_pivots"].copy()) if joint_fk else None,
                torch.from_numpy(archive["link_open_positions"].copy()) if joint_fk else None,
                torch.from_numpy(archive["link_open_rotations"].copy()) if joint_fk else None,
                torch.from_numpy(archive["link_axes"].copy()) if joint_fk else None,
                (
                    torch.from_numpy(
                        archive["link_position_joint_coefficients"].copy()
                    )
                    if joint_fk
                    else None
                ),
                (
                    torch.from_numpy(
                        archive["link_rotation_joint_coefficients"].copy()
                    )
                    if joint_fk
                    else None
                ),
            )

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.intercept, self.slope, self.link_index):
            digest.update(tensor.detach().cpu().numpy().tobytes())
        if self.aperture_knots is not None:
            digest.update(self.aperture_knots.detach().cpu().numpy().tobytes())
        if self.point_knots is not None:
            digest.update(self.point_knots.detach().cpu().numpy().tobytes())
        if self.supports_joint_fk:
            digest.update("\0".join(self.joint_names).encode("utf-8"))
            for tensor in (
                self.free_joint_knots,
                self.local_points,
                self.link_pivots,
                self.link_open_positions,
                self.link_open_rotations,
                self.link_axes,
                self.link_position_joint_coefficients,
                self.link_rotation_joint_coefficients,
            ):
                assert tensor is not None
                digest.update(tensor.detach().cpu().numpy().tobytes())
        digest.update(
            f"{self.aperture_min}:{self.aperture_max}:{self.length_scale}:{self.source_sha256}".encode()
        )
        return digest.hexdigest()


def _farthest_point_subset(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < count:
        raise ValueError(f"surface sampler returned only {len(points)} points, need {count}")
    chosen = np.empty(count, dtype=np.int64)
    chosen[0] = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))[0]
    minimum_distance = np.full(len(points), np.inf)
    for index in range(1, count):
        delta = points - points[chosen[index - 1]]
        minimum_distance = np.minimum(minimum_distance, np.einsum("ij,ij->i", delta, delta))
        chosen[index] = int(np.argmax(minimum_distance))
    return points[chosen]


def _sample_collision_links(
    robot: object,
    finger_links: tuple[str, str],
    *,
    samples_per_link: int,
    seed: int,
) -> list[np.ndarray]:
    import trimesh

    local_samples: list[np.ndarray] = []
    for link_offset, link_name in enumerate(finger_links):
        pieces = []
        for collision in robot.link_map[link_name].collisions:
            scene = robot._geometry2trimeshscene(
                collision.geometry,
                load_file=True,
                force_mesh=True,
                skip_materials=True,
            )
            if scene is None:
                continue
            mesh_piece = scene.to_geometry()
            mesh_piece.apply_transform(np.eye(4) if collision.origin is None else collision.origin)
            pieces.append(mesh_piece)
        if not pieces:
            raise ValueError(f"finger link {link_name!r} has no collision mesh")
        mesh = trimesh.util.concatenate(pieces)
        candidates, _ = trimesh.sample.sample_surface(
            mesh, samples_per_link * 8, seed=seed + link_offset
        )
        local_samples.append(_farthest_point_subset(np.asarray(candidates), samples_per_link))
    return local_samples


def _transform_link_samples(
    robot: object,
    finger_links: tuple[str, str],
    local_samples: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    transformed: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    for link_name, points in zip(finger_links, local_samples, strict=True):
        transform = np.asarray(robot.get_transform(link_name, robot.base_link, collision_geometry=True))
        homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
        transformed.append((transform @ homogeneous.T).T[:, :3])
        origins.append(transform[:3, 3])
    return np.concatenate(transformed, axis=0), np.stack(origins)


def preprocess_urdf(
    urdf_path: str | Path,
    *,
    finger_links: tuple[str, str],
    joint_map: Mapping[str, tuple[float, float]],
    aperture_min: float,
    aperture_max: float,
    length_scale: float,
    samples_per_link: int = 128,
    seed: int = 0,
    affine_tolerance: float = 1e-6,
) -> GripperAsset:
    """Create a runtime gripper asset from collision meshes in a URDF.

    ``joint_map`` maps a joint name to ``(offset, aperture_multiplier)``.
    """

    from yourdfpy import URDF
    path = Path(urdf_path).resolve()
    robot = URDF.load(
        str(path),
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_meshes=True,
        load_collision_meshes=True,
    )
    missing = [name for name in finger_links if name not in robot.link_map]
    if missing:
        raise ValueError(f"finger links not found in URDF: {missing}")

    local_samples = _sample_collision_links(
        robot,
        finger_links,
        samples_per_link=samples_per_link,
        seed=seed,
    )

    apertures = np.asarray(
        [aperture_min, 0.5 * (aperture_min + aperture_max), aperture_max], dtype=np.float64
    )
    points_by_aperture: list[np.ndarray] = []
    for aperture in apertures:
        robot.update_cfg(
            {name: offset + multiplier * float(aperture) for name, (offset, multiplier) in joint_map.items()}
        )
        transformed, _ = _transform_link_samples(robot, finger_links, local_samples)
        points_by_aperture.append(transformed)

    at_min, at_mid, at_max = points_by_aperture
    slope = (at_max - at_min) / (aperture_max - aperture_min)
    intercept = at_min - aperture_min * slope
    reconstructed_mid = intercept + apertures[1] * slope
    maximum_error = float(np.linalg.norm(at_mid - reconstructed_mid, axis=-1).max())
    if maximum_error > affine_tolerance:
        raise ValueError(
            f"finger motion is not affine in aperture: max error {maximum_error:.3e} m "
            f"> tolerance {affine_tolerance:.3e} m"
        )

    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    link_index = np.repeat(np.arange(2, dtype=np.int64), samples_per_link)
    return GripperAsset(
        torch.from_numpy(intercept.astype(np.float32)),
        torch.from_numpy(slope.astype(np.float32)),
        torch.from_numpy(link_index),
        float(aperture_min),
        float(aperture_max),
        float(length_scale),
        source_hash,
    )
