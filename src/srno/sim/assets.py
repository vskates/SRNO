"""Resolve the vendored validation assets without importing ``vv_assets``.

The catalog is intentionally JSON and the resolver only uses the Python standard
library. Isaac Lab can consume the returned USD paths, while model/data tooling can
inspect the exact spawn transform without importing the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from xml.etree import ElementTree


CATALOG_SCHEMA_VERSION = 1


class AssetCatalogError(ValueError):
    """Raised when an asset catalog is malformed or inconsistent."""


@dataclass(frozen=True)
class GripperAssetRecord:
    """Local paths and hashes for the validation gripper snapshot."""

    runtime_usd: Path
    runtime_usd_sha256: str
    source_urdf: Path
    source_urdf_sha256: str
    affine_urdf_preprocessor_compatible: bool
    contact_links: tuple[str, ...]
    open_joint_position_rad: Mapping[str, float]
    close_joint_position_rad: Mapping[str, float]


@dataclass(frozen=True)
class ObjectAssetRecord:
    """One object in the frozen validation object set."""

    object_id: str
    usd_path: Path
    texture_directory: Path
    bbox_size_m: tuple[float, float, float]
    mass_kg: float
    sha256: str
    spawn_scale: tuple[float, float, float] | None
    spawn_quaternion_wxyz: tuple[float, float, float, float]
    pose_seed_path: Path


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssetCatalogError(f"{label} must be a JSON object")
    return value


def _finite_tuple(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise AssetCatalogError(f"{label} must contain exactly {length} numbers")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise AssetCatalogError(f"{label} must contain only finite numbers")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SimulatorAssetCatalog:
    """Immutable, library-independent view of the vendored simulator assets."""

    def __init__(self, catalog_path: Path, payload: Mapping[str, Any]) -> None:
        self.catalog_path = catalog_path.resolve()
        self.asset_root = self.catalog_path.parent
        self.payload = payload

        version = payload.get("schema_version")
        if version != CATALOG_SCHEMA_VERSION:
            raise AssetCatalogError(
                f"Unsupported asset catalog schema {version!r}; "
                f"expected {CATALOG_SCHEMA_VERSION}"
            )
        if payload.get("units") != "SI":
            raise AssetCatalogError("asset catalog units must be 'SI'")

        spawn = _require_mapping(payload.get("validation_spawn"), "validation_spawn")
        if spawn.get("quaternion_order") != "wxyz":
            raise AssetCatalogError("validation spawn quaternions must use WXYZ order")
        raw_object_scale = spawn.get("object_scale")
        self.object_spawn_scale = (
            None
            if raw_object_scale is None
            else _finite_tuple(raw_object_scale, 3, "object_scale")
        )
        if self.object_spawn_scale is not None and any(
            component <= 0.0 for component in self.object_spawn_scale
        ):
            raise AssetCatalogError("object_scale components must be positive")
        self.object_spawn_quaternion_wxyz = _finite_tuple(
            spawn.get("object_quaternion_wxyz"), 4, "object_quaternion_wxyz"
        )
        quaternion_norm = math.sqrt(
            sum(component * component for component in self.object_spawn_quaternion_wxyz)
        )
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise AssetCatalogError("object spawn quaternion must be normalized")
        self.approach_axis_local = _finite_tuple(
            spawn.get("approach_axis_local"), 3, "approach_axis_local"
        )
        if not math.isclose(
            math.sqrt(sum(component * component for component in self.approach_axis_local)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise AssetCatalogError("approach_axis_local must be a unit vector")

        pose_seeds = _require_mapping(payload.get("pose_seeds"), "pose_seeds")
        self.pose_seed_directory = self._resolve(
            pose_seeds.get("directory"), "pose_seeds.directory"
        )
        self.pose_seed_filename_pattern = str(pose_seeds.get("filename_pattern", ""))
        if self.pose_seed_filename_pattern != "{object_id}.npz":
            raise AssetCatalogError("pose seed filename pattern must be '{object_id}.npz'")
        if int(pose_seeds.get("format_version", -1)) != 1:
            raise AssetCatalogError("unsupported pose seed format version")

        gripper = _require_mapping(payload.get("gripper"), "gripper")
        self.gripper = GripperAssetRecord(
            runtime_usd=self._resolve(gripper.get("runtime_usd"), "gripper.runtime_usd"),
            runtime_usd_sha256=self._digest(gripper, "runtime_usd_sha256"),
            source_urdf=self._resolve(gripper.get("source_urdf"), "gripper.source_urdf"),
            source_urdf_sha256=self._digest(gripper, "source_urdf_sha256"),
            affine_urdf_preprocessor_compatible=bool(
                gripper.get("affine_urdf_preprocessor_compatible")
            ),
            contact_links=tuple(str(link) for link in gripper.get("contact_links", [])),
            open_joint_position_rad=MappingProxyType(
                self._joint_positions(
                    gripper.get("open_joint_position_rad"), "open_joint_position_rad"
                )
            ),
            close_joint_position_rad=MappingProxyType(
                self._joint_positions(
                    gripper.get("close_joint_position_rad"), "close_joint_position_rad"
                )
            ),
        )

        raw_objects = payload.get("objects")
        if not isinstance(raw_objects, list) or not raw_objects:
            raise AssetCatalogError("objects must be a non-empty JSON array")
        records: dict[str, ObjectAssetRecord] = {}
        for index, raw_record in enumerate(raw_objects):
            record = _require_mapping(raw_record, f"objects[{index}]")
            object_id = str(record.get("id", "")).strip()
            if not object_id:
                raise AssetCatalogError(f"objects[{index}].id must not be empty")
            if object_id in records:
                raise AssetCatalogError(f"duplicate object id: {object_id}")
            bbox = _finite_tuple(record.get("bbox_size_m"), 3, f"{object_id}.bbox_size_m")
            mass = float(record.get("mass_kg", math.nan))
            if any(component <= 0.0 for component in bbox) or not math.isfinite(mass) or mass <= 0.0:
                raise AssetCatalogError(f"{object_id} must have positive finite dimensions and mass")
            records[object_id] = ObjectAssetRecord(
                object_id=object_id,
                usd_path=self._resolve(record.get("usd"), f"{object_id}.usd"),
                texture_directory=self._resolve(record.get("textures"), f"{object_id}.textures"),
                bbox_size_m=bbox,
                mass_kg=mass,
                sha256=self._digest(record, "sha256", prefix=object_id),
                spawn_scale=self.object_spawn_scale,
                spawn_quaternion_wxyz=self.object_spawn_quaternion_wxyz,
                pose_seed_path=self.pose_seed_directory / f"{object_id}.npz",
            )
        self._objects = records

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SimulatorAssetCatalog":
        """Load the repository catalog or an explicitly supplied catalog."""

        catalog_path = cls.default_path() if path is None else Path(path)
        try:
            with catalog_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as exc:
            raise AssetCatalogError(f"Invalid JSON in {catalog_path}: {exc}") from exc
        return cls(catalog_path, _require_mapping(payload, "catalog root"))

    @staticmethod
    def default_path() -> Path:
        """Return ``assets/catalog.json`` for an editable/source checkout."""

        return Path(__file__).resolve().parents[3] / "assets" / "catalog.json"

    @property
    def object_ids(self) -> tuple[str, ...]:
        """Configured object IDs, preserving validation-config order."""

        return tuple(self._objects)

    @property
    def objects(self) -> tuple[ObjectAssetRecord, ...]:
        return tuple(self._objects.values())

    def object(self, object_id: str) -> ObjectAssetRecord:
        try:
            return self._objects[object_id]
        except KeyError as exc:
            raise KeyError(f"Unknown simulator object {object_id!r}") from exc

    def validate_files(self, *, verify_hashes: bool = False) -> None:
        """Check local files, source-URDF meshes, and optionally USD hashes."""

        files_and_hashes = (
            (self.gripper.runtime_usd, self.gripper.runtime_usd_sha256),
            (self.gripper.source_urdf, self.gripper.source_urdf_sha256),
            *((record.usd_path, record.sha256) for record in self.objects),
        )
        for path, expected_digest in files_and_hashes:
            if not path.is_file():
                raise FileNotFoundError(f"Missing simulator asset: {path}")
            if verify_hashes and _sha256(path) != expected_digest:
                raise AssetCatalogError(f"SHA-256 mismatch for {path}")

        for record in self.objects:
            if not record.texture_directory.is_dir():
                raise FileNotFoundError(
                    f"Missing texture directory for {record.object_id}: {record.texture_directory}"
                )
            if not any(path.is_file() for path in record.texture_directory.iterdir()):
                raise AssetCatalogError(
                    f"Texture directory for {record.object_id} is empty: "
                    f"{record.texture_directory}"
                )
            if not record.pose_seed_path.is_file():
                raise FileNotFoundError(
                    f"Missing pose seeds for {record.object_id}: {record.pose_seed_path}"
                )

        try:
            root = ElementTree.parse(self.gripper.source_urdf).getroot()
        except ElementTree.ParseError as exc:
            raise AssetCatalogError(f"Invalid source URDF: {exc}") from exc
        for mesh in root.iter("mesh"):
            filename = mesh.get("filename")
            if not filename:
                raise AssetCatalogError("URDF mesh element is missing filename")
            mesh_path = self._resolve_from(self.gripper.source_urdf.parent, filename, "URDF mesh")
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing URDF mesh: {mesh_path}")

    def _resolve(self, value: Any, label: str) -> Path:
        return self._resolve_from(self.asset_root, value, label)

    @staticmethod
    def _resolve_from(root: Path, value: Any, label: str) -> Path:
        if not isinstance(value, str) or not value:
            raise AssetCatalogError(f"{label} must be a non-empty relative path")
        relative = Path(value)
        if relative.is_absolute():
            raise AssetCatalogError(f"{label} must be relative, got {value!r}")
        root = root.resolve()
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise AssetCatalogError(f"{label} escapes its asset directory: {value!r}")
        return resolved

    @staticmethod
    def _digest(record: Mapping[str, Any], key: str, *, prefix: str = "gripper") -> str:
        digest = record.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise AssetCatalogError(f"{prefix}.{key} must be a lowercase SHA-256 digest")
        return digest

    @staticmethod
    def _joint_positions(value: Any, label: str) -> Mapping[str, float]:
        raw = _require_mapping(value, f"gripper.{label}")
        parsed = {str(name): float(position) for name, position in raw.items()}
        if not parsed or not all(math.isfinite(position) for position in parsed.values()):
            raise AssetCatalogError(f"gripper.{label} must contain finite joint positions")
        return parsed
