# SRNO contact-resolvent v1

This repository implements a shared neural resolvent for quasistatic finite-actuation
parallel-jaw closure. The learned component only corrects trial configurations near
contact; free motion is an exact deterministic bypass.

## Environment

The reference environment is the existing `isaaclab` conda environment (Python 3.11,
PyTorch 2.7 with CUDA 12.8):

```bash
conda run -n isaaclab python -m pip install -e . --no-deps
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n isaaclab pytest
```

The model, dataset, and geometry runtime has no dependency on Isaac Sim. The optional
`srno sim collect` entry point uses the existing Isaac Lab installation and exports
directly through the same generic HDF5 writer used by tests and external collectors.

Plugin autoload is disabled in the test command because the host ROS installation
registers Python 3.12 pytest plugins globally, while the Isaac environment uses Python
3.11. This does not affect SRNO runtime code.

## Frozen validation assets

The repository vendors the exact gripper and 28 product assets selected by the active
`graspvalidation` configuration under `assets/`. This removes the runtime dependency on
`vv_assets`:

```python
from srno.sim import SimulatorAssetCatalog

assets = SimulatorAssetCatalog.load()
assets.validate_files(verify_hashes=True)
object_usd = assets.object("voda-mineralnaya-psyzh-1-l-27215").usd_path
gripper_usd = assets.gripper.runtime_usd
```

The equivalent command-line check is `srno assets validate --verify-hashes`.

See `assets/README.md` for the verified validation-frame conventions and the distinction
between the active `gripper_playground.usd` and the generation-only source URDF. Large
binary assets are tracked with Git LFS.

## Simulator dataset collection

`configs/simulator.toml` is intentionally small. Object membership and every local USD
path come only from `assets/catalog.json`; the simulator config controls execution,
quasistatic settling, SDF resolution, and dataset scales:

```bash
srno sim collect --config configs/simulator.toml
```

For a bounded smoke run, use `--max-objects 1 --trajectories-per-object 1`. Individual
objects may be selected with repeated `--object ID` arguments. Collection is resumable at
one HDF5 shard per object unless `--overwrite` is supplied.

The scene reuses validation-gen's exact gripper USD, contact material, object USDs, spawn
rotations, and grasp-pose seeds. It deliberately uses zero gravity and no shelf: each
object and open gripper are placed directly at zero velocity, and the object is never fixed
along Z or any other coordinate. Omitting the validation approach is intentional: without
its supporting shelf, approach contact would push a free object away before closure begins.
Configured viscous damping lets the body dissipate contact-induced velocity, while an
implicit six-joint PD servo follows the exact Astribot mimic schedule at low velocity. The
initial state and each of the 32 aperture states are recorded only once object and joint
residual velocities satisfy the configured settling rule.

Dense SDFs are built one object at a time in isolated subprocesses, before Isaac Sim is
launched, so mesh-query memory is returned to the OS. Every simulator entry point is
wrapped by a separate-process RAM watchdog, so native Kit calls cannot prevent it from
running. By default it samples every 0.25 seconds and forcibly terminates the collector
if either total host RAM use or collector process-tree RSS exceeds 12 GiB; both values
are configurable in `configs/simulator.toml`.

## Coordinate and data contract

All distances are SI metres. `q=(R,p)` is the object-to-gripper transform:
`x_gripper = R @ x_object + p`. Quaternions on disk use XYZW order. SDF values are
positive outside the object; `grid_origin` is the XYZ coordinate of the centre of voxel
`[z=0,y=0,x=0]`, and `voxel_size` is XYZ spacing.

A bundle consists of one `manifest.json`, one immutable gripper `.npz`, and object-centric
HDF5 shards. A minimal exporter uses:

```python
from srno.data import H5DatasetWriter, PhysicsMetadata

physics = PhysicsMetadata(
    static_friction=2.4, dynamic_friction=2.0,
    friction_combine_mode="min", restitution=0.0,
    restitution_combine_mode="min", strong_friction_enabled=True,
    contact_model="rigid", contact_stiffness=0.0, contact_damping=0.0,
    friction_model="patch", contact_generation="PCM", solver_type="TGS",
    solver_position_iterations=64, solver_velocity_iterations=16,
    simulator="Isaac Sim / PhysX", simulator_version="5.1.0.0",
)

with H5DatasetWriter("data/shard-000.h5", physics=physics) as writer:
    writer.add_object(
        "object-0001",
        sdf=sdf_zyx,                    # [96, 96, 96]
        grid_origin=origin_xyz,         # [3]
        voxel_size=voxel_size_xyz,      # [3]
        position=position_m,            # [T, 33, 3]
        quaternion_xyzw=quaternion,     # [T, 33, 4]
        joint_position=joint_rad,       # [T, 33, 6], actual PhysX joints
        joint_names=joint_names,        # runtime articulation order
        actual_aperture=aperture_m,     # [T, 33], derived A(joint_position)
        diagnostics={                   # optional, never used as ML targets
            "contact_count": contact_count,       # [T, 32]
            "actuator_effort": actuator_effort,   # [T, 32]
            "max_penetration": penetration_m,     # [T, 32]
            "linear_velocity": linear_velocity,   # [T, 32, 3]
            "angular_velocity": angular_velocity, # [T, 32, 3]
            "settling_substeps": settling_steps,  # [T, 32]
        },
    )
```

The Isaac collector must record a state only after the requested aperture increment has
settled, with gravity, external wrench, and environment contacts disabled. One object SDF
is stored once no matter how many initial gripper configurations are simulated.

Construct `DatasetManifest` with `DatasetManifest.create(...)`; shard object IDs and the
object-wise train/val/test split must match exactly. The validator additionally enforces
`max(voxel_size) < delta_gate / 2`, normalized quaternions, the six-joint ordering from
the gripper asset, and consistency of the diagnostic `actual_aperture = A(r)`.
Schema v2 also requires the manifest and every shard to contain exactly the same physics
fingerprint; datasets without a verified friction/contact/solver contract are rejected.

## Gripper preprocessing

For grippers whose finger motion is affine, runtime geometry is the exact map
`x_G(y,a)=intercept+slope*a`. Build it from the two collision links of a prismatic URDF:

```bash
srno gripper preprocess \
  --urdf gripper.urdf \
  --finger-link left_finger --finger-link right_finger \
  --joint-map left_joint=0,0.5 --joint-map right_joint=0,0.5 \
  --aperture-min 0.0 --aperture-max 0.08 --length-scale 0.08 \
  --output data/gripper.npz
```

Each joint map is `joint_position = OFFSET + MULTIPLIER * aperture`. URDF joint axes carry
their own signs. Preprocessing samples 128 collision-surface points per finger and rejects
non-affine finger motion. The vendored Astribot runtime path instead samples its actual
USD collision hulls and stores a six-joint differentiable FK plus the 33 empty-gripper
joint configurations. Input gaps, contact gating, and feasibility therefore all use
`points_from_joints(r)`; scalar aperture is only an evaluation diagnostic.

## Validation and training

```bash
srno dataset validate data/manifest.json
srno dataset calibrate-gate data/manifest.json --device cuda
srno dataset build-active-index data/manifest.json \
  --output data/active-index.npz --device cuda

srno train --config configs/srno-r-joint-diagnostic.toml --stage local
srno train --config configs/srno-r-joint-diagnostic.toml --stage rollout \
  --resume runs/srno-r-joint-diagnostic/best-local.pt
srno evaluate --config configs/srno-r-joint-diagnostic.toml \
  --checkpoint runs/srno-r-joint-diagnostic/best-rollout.pt --split test
```

Calibration requires `contact_count` diagnostics. It recommends the smallest gate
threshold attaining 99.5% simulator-contact recall while also respecting voxel
resolution, and reports the 99.5%-coverage geometric `admissible_gap_m` from settled
states and actual joints. Regenerate trajectories and rerun this command after every
material change; the existing cooked-SDF caches and gripper FK do not need rebuilding.
The local stage reads only indexed active transitions. Rollout training is
autoregressive with horizons 4, 8, 16, and 32; no free-space, stall, force, or grasp-quality
auxiliary losses are present.

The learned head predicts a spatial object correction and six normalized contact-joint
residuals. State loss normalizes every joint error by that joint's free travel range.
The 2.56 mm PhysX contact envelope is subtracted only for contact gating; geometric
feasibility always queries the raw cooked-collider SDF.

Loader sizes are minibatch chunk sizes, not dataset subsampling limits. In every training
epoch the complete-coverage sampler visits each active train transition or trajectory
exactly once, without replacement. The default rollout minibatch contains up to four
objects and eight trajectories per object, while each object's SDF appears only once in
that minibatch. Validation and test evaluation always consume every trajectory.
