#pragma once

#include <array>

namespace solar_system
{
    namespace barycentric
    {
        template <typename Real>
        struct coord
        {
            std::array<Real, 3> r;
        };
    }
}