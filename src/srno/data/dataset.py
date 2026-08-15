from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from srno.data.index import ActiveIndex
from srno.data.schema import DatasetManifest
from srno.geometry.se3 import quaternion_xyzw_to_matrix
from srno.types import PoseState, SDFBatch


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    sdf: Tensor
    origin: Tensor
    voxel_size: Tensor
    position: Tensor
    quaternion_xyzw: Tensor
    aperture: Tensor
    active_pairs: Tensor | None
    diagnostics: dict[str, Tensor]
    sample_indices: Tensor | None = None


@dataclass(frozen=True)
class ObjectSampleKey:
    object_index: int
    sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class LocalTransitionBatch:
    sdf: SDFBatch
    current: PoseState
    target: PoseState
    next_command: Tensor
    object_ids: tuple[str, ...]
    trajectory_index: Tensor
    step_index: Tensor

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "LocalTransitionBatch":
        return LocalTransitionBatch(
            self.sdf.to(device=device, non_blocking=non_blocking),
            self.current.to(device=device, non_blocking=non_blocking),
            self.target.to(device=device, non_blocking=non_blocking),
            self.next_command.to(device=device, non_blocking=non_blocking),
            self.object_ids,
            self.trajectory_index.to(device=device, non_blocking=non_blocking),
            self.step_index.to(device=device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "LocalTransitionBatch":
        return self.to("cpu")._pin()

    def _pin(self) -> "LocalTransitionBatch":
        return LocalTransitionBatch(
            _pin_sdf(self.sdf),
            _pin_state(self.current),
            _pin_state(self.target),
            self.next_command.pin_memory(),
            self.object_ids,
            self.trajectory_index.pin_memory(),
            self.step_index.pin_memory(),
        )


@dataclass(frozen=True)
class TrajectoryBatch:
    sdf: SDFBatch
    states: PoseState
    command_schedule: Tensor
    object_ids: tuple[str, ...]
    trajectory_index: Tensor

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "TrajectoryBatch":
        return TrajectoryBatch(
            self.sdf.to(device=device, non_blocking=non_blocking),
            self.states.to(device=device, non_blocking=non_blocking),
            self.command_schedule.to(device=device, non_blocking=non_blocking),
            self.object_ids,
            self.trajectory_index.to(device=device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "TrajectoryBatch":
        return TrajectoryBatch(
            _pin_sdf(self.sdf),
            _pin_state(self.states),
            self.command_schedule.pin_memory(),
            self.object_ids,
            self.trajectory_index.pin_memory(),
        )


def _pin_state(state: PoseState) -> PoseState:
    return PoseState(state.rotation.pin_memory(), state.position.pin_memory(), state.aperture.pin_memory())


def _pin_sdf(sdf: SDFBatch) -> SDFBatch:
    return SDFBatch(
        sdf.values.pin_memory(),
        sdf.origin.pin_memory(),
        sdf.voxel_size.pin_memory(),
        sdf.sample_to_object.pin_memory(),
        sdf.outside_value,
    )


class H5ObjectDataset(Dataset[ObjectRecord]):
    """Lazy object-level reader. HDF5 handles are process-local."""

    def __init__(
        self,
        manifest: DatasetManifest | str | Path,
        *,
        split: Literal["train", "val", "test"],
        active_index: ActiveIndex | str | Path | None = None,
        active_only: bool = False,
        hdf5_cache_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.manifest = (
            manifest if isinstance(manifest, DatasetManifest) else DatasetManifest.load(manifest)
        )
        self.locations = self.manifest.object_locations()
        self.active_index = (
            ActiveIndex.load(active_index)
            if isinstance(active_index, (str, Path))
            else active_index
        )
        if self.active_index is not None:
            if self.active_index.manifest_sha256 != self.manifest.sha256():
                raise ValueError("active index was built for a different manifest")
            if not np.isclose(self.active_index.delta_gate_m, self.manifest.delta_gate_m):
                raise ValueError("active index gate does not match manifest")
        object_ids = list(self.manifest.splits[split])
        if active_only:
            if self.active_index is None:
                raise ValueError("active_only requires an active index")
            object_ids = [
                object_id for object_id in object_ids if len(self.active_index.pairs_for(object_id))
            ]
        self.object_ids = tuple(object_ids)
        self.hdf5_cache_bytes = int(hdf5_cache_bytes)
        self._handles: dict[Path, h5py.File] = {}
        self._pid = os.getpid()

    def __len__(self) -> int:
        return len(self.object_ids)

    def _handle(self, path: Path) -> h5py.File:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r", swmr=True, rdcc_nbytes=self.hdf5_cache_bytes)
            self._handles[path] = handle
        return handle

    def __getitem__(self, index: int | ObjectSampleKey) -> ObjectRecord:
        sample_indices = None
        if isinstance(index, ObjectSampleKey):
            sample_indices = torch.tensor(index.sample_indices, dtype=torch.long)
            index = index.object_index
        object_id = self.object_ids[index]
        path, group_name = self.locations[object_id]
        group = self._handle(path)[group_name]
        stored_id = str(group.attrs["object_id"])
        if stored_id != object_id:
            raise ValueError(f"manifest/HDF5 object mismatch: {object_id!r} != {stored_id!r}")
        diagnostics: dict[str, Tensor] = {}
        if "diagnostics" in group:
            diagnostics = {
                key: torch.from_numpy(dataset[...])
                for key, dataset in group["diagnostics"].items()
            }
        pairs = None
        if self.active_index is not None:
            pairs = torch.from_numpy(self.active_index.pairs_for(object_id))
        return ObjectRecord(
            object_id,
            torch.from_numpy(group["sdf"][...]),
            torch.from_numpy(np.asarray(group.attrs["grid_origin"], dtype=np.float32)),
            torch.from_numpy(np.asarray(group.attrs["voxel_size"], dtype=np.float32)),
            torch.from_numpy(group["position"][...]),
            torch.from_numpy(group["quaternion_xyzw"][...]),
            torch.from_numpy(group["actual_aperture"][...]),
            pairs,
            diagnostics,
            sample_indices,
        )

    def sample_counts(self, mode: Literal["local", "rollout"]) -> tuple[int, ...]:
        if mode == "local":
            if self.active_index is None:
                raise ValueError("local sample counts require an active index")
            return tuple(len(self.active_index.pairs_for(object_id)) for object_id in self.object_ids)
        counts: list[int] = []
        for object_id in self.object_ids:
            path, group_name = self.locations[object_id]
            with h5py.File(path, "r", swmr=True) as handle:
                counts.append(int(handle[group_name]["position"].shape[0]))
        return tuple(counts)

    def close(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        if hasattr(self, "_handles"):
            self._handles.clear()

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_pid"] = os.getpid()
        return state


class ObjectBatchCollator:
    def __init__(
        self,
        manifest: DatasetManifest,
        *,
        mode: Literal["local", "rollout"],
        samples_per_object: int = 0,
        seed: int = 0,
        resample: bool = True,
    ) -> None:
        self.manifest = manifest
        self.mode = mode
        self.samples_per_object = samples_per_object
        self.seed = seed
        self.resample = resample
        self._calls = 0

    def _rng(self, records: list[ObjectRecord]) -> np.random.Generator:
        call = self._calls if self.resample else 0
        payload = f"{self.seed}:{call}:" + ":".join(record.object_id for record in records)
        if self.resample:
            self._calls += 1
        value = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")
        return np.random.default_rng(value)

    def __call__(self, records: list[ObjectRecord]) -> LocalTransitionBatch | TrajectoryBatch:
        if not records:
            raise ValueError("cannot collate an empty object batch")
        rng = self._rng(records)
        sdf = torch.stack([record.sdf for record in records])
        origin = torch.stack([record.origin for record in records])
        voxel = torch.stack([record.voxel_size for record in records])
        schedule = torch.tensor(self.manifest.commanded_aperture_m, dtype=torch.float32)

        positions: list[Tensor] = []
        quaternions: list[Tensor] = []
        apertures: list[Tensor] = []
        mapping: list[int] = []
        trajectory_indices: list[int] = []
        step_indices: list[int] = []
        for object_index, record in enumerate(records):
            if self.mode == "local":
                if record.active_pairs is None or not len(record.active_pairs):
                    continue
                choices = (
                    record.sample_indices.numpy()
                    if record.sample_indices is not None
                    else np.arange(len(record.active_pairs))
                    if self.samples_per_object == 0
                    else rng.integers(
                        0, len(record.active_pairs), size=self.samples_per_object
                    )
                )
                pairs = record.active_pairs[torch.from_numpy(choices)]
                trajectory = pairs[:, 0].long()
                step = pairs[:, 1].long()
                positions.append(record.position[trajectory, step])
                quaternions.append(record.quaternion_xyzw[trajectory, step])
                apertures.append(record.aperture[trajectory, step])
                positions.append(record.position[trajectory, step + 1])
                quaternions.append(record.quaternion_xyzw[trajectory, step + 1])
                apertures.append(record.aperture[trajectory, step + 1])
                mapping.extend([object_index] * len(trajectory))
                trajectory_indices.extend(trajectory.tolist())
                step_indices.extend(step.tolist())
            else:
                trajectory_count = record.position.shape[0]
                choices = (
                    record.sample_indices.numpy()
                    if record.sample_indices is not None
                    else np.arange(trajectory_count)
                    if self.samples_per_object <= 0
                    else rng.integers(0, trajectory_count, size=self.samples_per_object)
                )
                trajectory = torch.from_numpy(choices).long()
                positions.append(record.position.index_select(0, trajectory))
                quaternions.append(record.quaternion_xyzw.index_select(0, trajectory))
                apertures.append(record.aperture.index_select(0, trajectory))
                mapping.extend([object_index] * len(trajectory))
                trajectory_indices.extend(trajectory.tolist())

        sdf_batch = SDFBatch(
            sdf,
            origin,
            voxel,
            torch.tensor(mapping, dtype=torch.long),
            self.manifest.sdf_scale_m,
        )
        if self.mode == "local":
            if not mapping:
                raise ValueError("object batch contains no active transitions")
            # Current and target records were appended in alternating pairs per object.
            current_positions = torch.cat(positions[0::2], dim=0)
            target_positions = torch.cat(positions[1::2], dim=0)
            current_quaternions = torch.cat(quaternions[0::2], dim=0)
            target_quaternions = torch.cat(quaternions[1::2], dim=0)
            current_apertures = torch.cat(apertures[0::2], dim=0)
            target_apertures = torch.cat(apertures[1::2], dim=0)
            steps = torch.tensor(step_indices, dtype=torch.long)
            return LocalTransitionBatch(
                sdf_batch,
                PoseState(
                    quaternion_xyzw_to_matrix(current_quaternions),
                    current_positions,
                    current_apertures,
                ),
                PoseState(
                    quaternion_xyzw_to_matrix(target_quaternions),
                    target_positions,
                    target_apertures,
                ),
                schedule.index_select(0, steps + 1),
                tuple(record.object_id for record in records),
                torch.tensor(trajectory_indices, dtype=torch.long),
                steps,
            )
        all_positions = torch.cat(positions, dim=0)
        all_quaternions = torch.cat(quaternions, dim=0)
        all_apertures = torch.cat(apertures, dim=0)
        return TrajectoryBatch(
            sdf_batch,
            PoseState(
                quaternion_xyzw_to_matrix(all_quaternions),
                all_positions,
                all_apertures,
            ),
            schedule,
            tuple(record.object_id for record in records),
            torch.tensor(trajectory_indices, dtype=torch.long),
        )


class CompleteCoverageBatchSampler(Sampler[list[ObjectSampleKey]]):
    """Cover every sample once per epoch while keeping object grids unique in a batch."""

    def __init__(
        self,
        dataset: H5ObjectDataset,
        *,
        mode: Literal["local", "rollout"],
        objects_per_batch: int,
        samples_per_object: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        if objects_per_batch <= 0 or samples_per_object < 0:
            raise ValueError("batch sizes must be non-negative and objects_per_batch positive")
        self.sample_counts = dataset.sample_counts(mode)
        if any(count <= 0 for count in self.sample_counts):
            raise ValueError("complete-coverage sampler received an empty object")
        self.objects_per_batch = objects_per_batch
        self.samples_per_object = samples_per_object
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _chunk_count(self, sample_count: int) -> int:
        if self.samples_per_object == 0:
            return 1
        return math.ceil(sample_count / self.samples_per_object)

    def __len__(self) -> int:
        remaining = [self._chunk_count(count) for count in self.sample_counts]
        batches = 0
        while any(remaining):
            active = sum(value > 0 for value in remaining)
            batches += math.ceil(active / self.objects_per_batch)
            remaining = [max(0, value - 1) for value in remaining]
        return batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        queues: list[list[tuple[int, ...]]] = []
        for sample_count in self.sample_counts:
            indices = rng.permutation(sample_count) if self.shuffle else np.arange(sample_count)
            chunk_count = self._chunk_count(sample_count)
            chunks = [tuple(map(int, chunk)) for chunk in np.array_split(indices, chunk_count)]
            queues.append(list(reversed(chunks)))

        while any(queues):
            active = [object_index for object_index, queue in enumerate(queues) if queue]
            if self.shuffle:
                rng.shuffle(active)
            for start in range(0, len(active), self.objects_per_batch):
                object_indices = active[start : start + self.objects_per_batch]
                yield [
                    ObjectSampleKey(object_index, queues[object_index].pop())
                    for object_index in object_indices
                ]


def make_dataloader(
    dataset: H5ObjectDataset,
    *,
    mode: Literal["local", "rollout"],
    objects_per_batch: int = 1,
    samples_per_object: int = 0,
    workers: int = 4,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader[LocalTransitionBatch | TrajectoryBatch]:
    generator = torch.Generator().manual_seed(seed)
    batch_sampler = CompleteCoverageBatchSampler(
        dataset,
        mode=mode,
        objects_per_batch=objects_per_batch,
        samples_per_object=samples_per_object,
        seed=seed,
        shuffle=shuffle,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        collate_fn=ObjectBatchCollator(
            dataset.manifest,
            mode=mode,
            samples_per_object=0,
            seed=seed,
            resample=False,
        ),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        generator=generator,
    )
