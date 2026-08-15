from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class GripperAsset:
    """Affine parallel-jaw surface kinematics ``x(a) = intercept + slope * a``."""

    intercept: Tensor
    slope: Tensor
    link_index: Tensor
    aperture_min: float
    aperture_max: float
    length_scale: float
    source_sha256: str = ""
    aperture_knots: Tensor | None = None
    point_knots: Tensor | None = None

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
        if (self.aperture_knots is None) != (self.point_knots is None):
            raise ValueError("aperture_knots and point_knots must be provided together")
        if self.aperture_knots is not None and self.point_knots is not None:
            if self.aperture_knots.ndim != 1 or len(self.aperture_knots) < 2:
                raise ValueError("aperture_knots must have shape [states] with at least two states")
            if self.point_knots.shape != (len(self.aperture_knots), self.point_count, 3):
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

    @property
    def point_count(self) -> int:
        return self.intercept.shape[0]

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
        )

    def points(self, aperture: Tensor | float) -> Tensor:
        aperture_tensor = torch.as_tensor(
            aperture, dtype=self.intercept.dtype, device=self.intercept.device
        )
        if torch.any(aperture_tensor < self.aperture_min - 1e-7) or torch.any(
            aperture_tensor > self.aperture_max + 1e-7
        ):
            raise ValueError("aperture is outside gripper limits")
        if self.aperture_knots is None or self.point_knots is None:
            return self.intercept + aperture_tensor[..., None, None] * self.slope

        knots = self.aperture_knots.to(device=aperture_tensor.device, dtype=aperture_tensor.dtype)
        points = self.point_knots.to(device=aperture_tensor.device, dtype=aperture_tensor.dtype)
        flat = aperture_tensor.reshape(-1)
        upper = torch.searchsorted(knots, flat).clamp(1, len(knots) - 1)
        lower = upper - 1
        low_aperture = knots.index_select(0, lower)
        high_aperture = knots.index_select(0, upper)
        alpha = (flat - low_aperture) / (high_aperture - low_aperture)
        low_points = points.index_select(0, lower)
        high_points = points.index_select(0, upper)
        interpolated = low_points + alpha[:, None, None] * (high_points - low_points)
        # Command schedules query the authored knots exactly.  Select their stored
        # geometry verbatim instead of introducing a rounding error through lerp.
        knot_distance = torch.abs(flat[:, None] - knots[None, :])
        exact_distance, exact_index = knot_distance.min(dim=-1)
        exact = exact_distance <= 4.0 * torch.finfo(flat.dtype).eps
        exact_points = points.index_select(0, exact_index)
        interpolated = torch.where(exact[:, None, None], exact_points, interpolated)
        return interpolated.reshape(aperture_tensor.shape + (self.point_count, 3))

    def pair_features(self, aperture: Tensor | float) -> Tensor:
        points = self.points(aperture) / self.length_scale
        target = points.unsqueeze(-2).expand(points.shape[:-2] + (self.point_count, self.point_count, 3))
        source = points.unsqueeze(-3).expand_as(target)
        return torch.cat((target, source), dim=-1)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        metadata = {
            "format_version": 2 if self.aperture_knots is not None else 1,
            "aperture_min": self.aperture_min,
            "aperture_max": self.aperture_max,
            "length_scale": self.length_scale,
            "source_sha256": self.source_sha256,
        }
        arrays: dict[str, np.ndarray] = {
            "intercept": self.intercept.detach().cpu().numpy().astype(np.float32),
            "slope": self.slope.detach().cpu().numpy().astype(np.float32),
            "link_index": self.link_index.detach().cpu().numpy().astype(np.int16),
            "metadata": np.asarray(json.dumps(metadata)),
        }
        if self.aperture_knots is not None and self.point_knots is not None:
            arrays["aperture_knots"] = (
                self.aperture_knots.detach().cpu().numpy().astype(np.float32)
            )
            arrays["point_knots"] = self.point_knots.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "GripperAsset":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format_version") not in (1, 2):
                raise ValueError("unsupported gripper asset version")
            scheduled = metadata["format_version"] == 2
            return cls(
                torch.from_numpy(archive["intercept"].copy()),
                torch.from_numpy(archive["slope"].copy()),
                torch.from_numpy(archive["link_index"].copy()).long(),
                float(metadata["aperture_min"]),
                float(metadata["aperture_max"]),
                float(metadata["length_scale"]),
                str(metadata.get("source_sha256", "")),
                torch.from_numpy(archive["aperture_knots"].copy()) if scheduled else None,
                torch.from_numpy(archive["point_knots"].copy()) if scheduled else None,
            )

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.intercept, self.slope, self.link_index):
            digest.update(tensor.detach().cpu().numpy().tobytes())
        if self.aperture_knots is not None and self.point_knots is not None:
            digest.update(self.aperture_knots.detach().cpu().numpy().tobytes())
            digest.update(self.point_knots.detach().cpu().numpy().tobytes())
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


def preprocess_scheduled_urdf(
    urdf_path: str | Path,
    *,
    finger_links: tuple[str, str],
    close_joint_positions: Mapping[str, float],
    command_fractions: np.ndarray | None = None,
    samples_per_link: int = 128,
    seed: int = 0,
) -> GripperAsset:
    """Sample exact gripper geometry along a coupled revolute closure schedule.

    Aperture is the separation of the two finger-link origins projected onto the
    gripper-local X axis. This is metric and matches the quantity measured by the
    simulator collector from the same runtime links.
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
    if not close_joint_positions:
        raise ValueError("close_joint_positions must not be empty")
    fractions = (
        np.linspace(0.0, 1.0, 33, dtype=np.float64)
        if command_fractions is None
        else np.asarray(command_fractions, dtype=np.float64)
    )
    if fractions.ndim != 1 or len(fractions) < 2:
        raise ValueError("command_fractions must be a one-dimensional schedule")
    if not np.isclose(fractions[0], 0.0) or not np.isclose(fractions[-1], 1.0):
        raise ValueError("command_fractions must start at zero and end at one")
    if not np.all(np.diff(fractions) > 0.0):
        raise ValueError("command_fractions must be strictly increasing")

    local_samples = _sample_collision_links(
        robot,
        finger_links,
        samples_per_link=samples_per_link,
        seed=seed,
    )
    points_by_fraction: list[np.ndarray] = []
    apertures: list[float] = []
    for fraction in fractions:
        robot.update_cfg(
            {
                name: float(fraction) * float(target)
                for name, target in close_joint_positions.items()
            }
        )
        transformed, origins = _transform_link_samples(robot, finger_links, local_samples)
        points_by_fraction.append(transformed)
        apertures.append(float(abs(origins[0, 0] - origins[1, 0])))

    aperture_array = np.asarray(apertures, dtype=np.float64)
    if not np.all(np.diff(aperture_array) < 0.0):
        raise ValueError("finger-origin aperture is not strictly decreasing along closure")
    point_array = np.stack(points_by_fraction)
    # GripperAsset interpolation knots are ascending; closure is open-to-closed.
    ascending_aperture = aperture_array[::-1].copy()
    ascending_points = point_array[::-1].copy()
    slope = (ascending_points[-1] - ascending_points[0]) / (
        ascending_aperture[-1] - ascending_aperture[0]
    )
    intercept = ascending_points[0] - ascending_aperture[0] * slope
    link_index = np.repeat(np.arange(2, dtype=np.int64), samples_per_link)
    return GripperAsset(
        torch.from_numpy(intercept.astype(np.float32)),
        torch.from_numpy(slope.astype(np.float32)),
        torch.from_numpy(link_index),
        float(ascending_aperture[0]),
        float(ascending_aperture[-1]),
        float(ascending_aperture[-1]),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        torch.from_numpy(ascending_aperture.astype(np.float32)),
        torch.from_numpy(ascending_points.astype(np.float32)),
    )
