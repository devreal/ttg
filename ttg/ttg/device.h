#ifndef TTG_DEVICE_H
#define TTG_DEVICE_H

#include "ttg/fwd.h"
#include "ttg/execution.h"

namespace ttg {
    namespace device {
        using DeviceAllocator = TTG_IMPL_NS::device::DeviceAllocator;
        std::size_t nb_devices() { return TTG_IMPL_NS::device::nb_devices(); }
    }
}

#endif /* TTG_DEVICE_H */