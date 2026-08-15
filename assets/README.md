# Frozen simulator assets

`catalog.json` is the source of truth for the simulator-facing asset set. It contains
the 29 object IDs from the active `graspvalidation/config.yaml`, the exact spawn
transform used by validation, SI bounding-box dimensions, authored masses, and SHA-256
digests of every dynamics-relevant USD file.

The object `.usdc` files and their relative texture dependencies were copied from the
`vv_assets/0.1.0` runtime cache. They are the already scaled, Z-up assets that Isaac Lab
spawns with scale `(1, 1, 1)`. No OBJ/STL duplicate is kept for objects: geometry and
collision data are already in each USDC, and the duplicate exported OBJ in
`graspvalidation/object_dataset` has an additional coordinate conversion used by grasp
generation rather than by physical spawning.

`grippers/astribot/gripper_playground.usd` is the actual validation articulation. It is
self-contained apart from Isaac's built-in `OmniPBR.mdl`. The neighboring `source`
directory preserves `gripper.urdf` and its STL dependencies for provenance and geometry
inspection; validation does not load that URDF.

The SRNO collector deliberately does not vendor or spawn validation's shelf. Objects are
free rigid bodies in zero gravity, with zero initial velocity and no coordinate fixation.
The validation assets are reused for geometry, dynamics, grasp-pose seeds, and the exact
six-joint gripper schedule. The collector uses a low-speed implicit PD realization of that
schedule because validation's lift-oriented explicit mimic controller does not converge to
quasistatic equilibrium in the free-body scene.

Two current-pipeline details are intentionally recorded rather than silently corrected:

- Validation's approach is local `+Z` of the commanded gripper base/grasp pose and
  pregrasp lies along local `-Z`; the base-to-articulation-root correction is numerically
  identity. This remains provenance metadata, although SRNO places the seed pose directly
  because the shelf-free object must not be displaced by an approach phase.
- The selected objects receive `(0, 0, 0, 1)` from a helper named as XYZW, but the value
  is passed to Isaac Lab as WXYZ. The actual current spawn is therefore a 180-degree
  rotation about Z. `catalog.json` stores this actual WXYZ value.

The Astribot fingers use six revolute joints. Their surface motion is not affine in a
scalar aperture, so the v1 affine URDF preprocessor must reject this source model. The
collector therefore stores all 33 exact schedule-indexed surface states; `SRNOModel`
selects these authored knots without an affine approximation. Runtime aperture is inferred
from the measured six-joint state on this same schedule, rather than from raw USD link-origin
separation (the runtime USD and source URDF define those origins differently).

Binary geometry and textures are configured for Git LFS in the repository root.
