// Runtime probe for the exact Isaac Sim 5.1 / PhysX 5.6.1 ABI used by SRNO.
//
// It exposes read-only PxMaterial properties that omni.physx does not expose
// through Python.  The Python wrapper rejects every other simulator version.

#include <cstddef>
#include <cstdint>

#include <carb/Framework.h>
#include <PxMaterial.h>

CARB_FRAMEWORK_GLOBALS("srno.physx-material-probe")

namespace
{
using GenericFunction = void(CARB_ABI*)();

struct MinimalIPhysx
{
    CARB_PLUGIN_INTERFACE("omni::physx::IPhysx", 5, 0)

    GenericFunction getObjectId;
    GenericFunction getPhysXPtr;
    void*(CARB_ABI* getPhysXPtrFast)(std::size_t objectId);
};

physx::PxMaterial* getMaterial(std::size_t objectId)
{
    static carb::Framework* framework = carb::acquireFramework("srno.physx-material-probe");
    if (!framework)
        return nullptr;
    static MinimalIPhysx* physxInterface = framework->tryAcquireInterface<MinimalIPhysx>();
    if (!physxInterface || !physxInterface->getPhysXPtrFast)
        return nullptr;
    return static_cast<physx::PxMaterial*>(physxInterface->getPhysXPtrFast(objectId));
}
} // namespace

extern "C" int srno_read_physx_material(
    std::size_t objectId,
    float* staticFriction,
    float* dynamicFriction,
    std::int32_t* frictionCombineMode,
    float* restitution,
    float* damping,
    std::uint16_t* flags)
{
    if (!staticFriction || !dynamicFriction || !frictionCombineMode || !restitution || !damping || !flags)
        return -1;
    physx::PxMaterial* material = getMaterial(objectId);
    if (!material)
        return -2;

    *staticFriction = material->getStaticFriction();
    *dynamicFriction = material->getDynamicFriction();
    *frictionCombineMode = static_cast<std::int32_t>(material->getFrictionCombineMode());
    *restitution = material->getRestitution();
    *damping = material->getDamping();
    *flags = static_cast<std::uint16_t>(material->getFlags());
    return 1;
}
