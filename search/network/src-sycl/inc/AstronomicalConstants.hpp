#pragma once

#include <MathConstants.hpp>

namespace astro
{
    template<class T> constexpr T omega_r = static_cast<T>(7.2921151467064e-5);
    template<class T> constexpr T sidereal_day = static_cast<T>(static_cast<T>(2)*math::pi<T> / omega_r<T>);
}