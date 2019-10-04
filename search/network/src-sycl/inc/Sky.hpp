#pragma once

#include <cstdint>
#include <array>

namespace sky
{
    namespace integer
    {
        struct coord
        {
            std::int32_t pm, mm, nn;
        };

        struct range
        {
            std::array<std::int32_t, 2> pmr, mr, nr;
        };
    }

    namespace linear
    {
        template <typename Real>
        struct coord
        {
            Real alpha1, alpha2;

            template <typename T>
            coord(integer::coord in, std::array<std::array<T, 4>, 4> fisher);
        };

        template <typename Real>
        struct grid
        {
            Real oms;   // Dimensionless angular frequency (fpo)
        };
    }

    namespace celestial
    {
        template <typename Real>
        struct coord
        {
            Real alpha, delta;
        };
    }
}
