// Diagnostic-only bridge to a PhysX material flag that Omniverse 107.3 does
// not expose through its Python or USD material APIs.
//
// This intentionally mirrors only the first three entries of IPhysx 5.0.  The
// ABI was checked against the IPhysx.h shipped for the exact Isaac Sim PhysX
// version used by SRNO (107.3 / PhysX 5.6.1).

#include <cstddef>

#include <carb/Framework.h>
#include <PxMaterial.h>

CARB_FRAMEWORK_GLOBALS("srno.contact-memory-diagnostic")

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
} // namespace

extern "C" int srno_set_disable_strong_friction(std::size_t objectId, int disabled)
{
    carb::Framework* framework = carb::acquireFramework("srno.contact-memory-diagnostic");
    if (!framework)
        return -1;

    MinimalIPhysx* physxInterface = framework->tryAcquireInterface<MinimalIPhysx>();
    if (!physxInterface || !physxInterface->getPhysXPtrFast)
        return -2;

    void* pointer = physxInterface->getPhysXPtrFast(objectId);
    if (!pointer)
        return -3;

    auto* material = static_cast<physx::PxMaterial*>(pointer);
    const bool value = disabled != 0;
    material->setFlag(physx::PxMaterialFlag::eDISABLE_STRONG_FRICTION, value);
    const bool observed = material->getFlags().isSet(physx::PxMaterialFlag::eDISABLE_STRONG_FRICTION);
    return observed == value ? 1 : -4;
}
