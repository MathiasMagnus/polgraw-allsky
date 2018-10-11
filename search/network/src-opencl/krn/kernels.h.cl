#ifndef __KERNELS_HCL__
#define __KERNELS_HCL__

#include <floats.h.cl>


/// <summary>The purpose of this function was undocumented.</summary>
///
__kernel void computeB(__global complex_t* y,
                       __global complex_t* B,
                       int N);

/// <summary>Multiplies the tridiagonal matrix specified by <c>{dl, d, du}</c> with dense vector <c>x</c>.</summary>
///
__kernel void tridiagMul(__global real_t* dl,
                         __global real_t* d,
                         __global real_t* du,
                         __global complex_t* x,
                         __global complex_t* y);

/// <summary>The purpose of this function was undocumented.</summary>
///
__kernel void interpolate(__global real_t* new_x,
                          __global complex_t* new_y,
                          __global complex_t* z,
                          __global complex_t* y,
                          int N,
                          int new_N);

#endif // __KERNELS_HCL__
