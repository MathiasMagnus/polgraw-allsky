// Polgraw includes
#include <floats.h.cl>       // real_t, complex_t
#include <kernels.h.cl>      // function declarations

//#pragma OPENCL EXTENSION cl_amd_printf : enable




/// <summary>The purpose of this function was undocumented.</summary>
///
__kernel void computeB(__global complex_t* y,
                       __global complex_t* B,
                       int N)
{
    size_t idx = get_global_id(0);

    if (idx < N - 1)
    {
		B[idx] = 6 * (y[idx + 2] - 2 * y[idx + 1] + y[idx]);
    }
}

/// <summary>Multiplies the tridiagonal matrix specified by <c>{dl, d, du}</c> with dense vector <c>x</c>.</summary>
///
__kernel void tridiagMul(__global real_t* dl,
                         __global real_t* d,
                         __global real_t* du,
                         __global complex_t* x,
                         __global complex_t* y)
{
    size_t gid = get_global_id(0);
    size_t gsi = get_global_size(0);

	// Select 3 contributing values from x
	complex_t x1 = (gid == 0 ? (real_t)0 : x[gid - 1]),
	          x2 = x[gid],
			  x3 = (gid == gsi - 1 ? (real_t)0 : x[gid + 1]);


    y[gid] = dl[gid] * x1 +
              d[gid] * x2 +
             du[gid] * x3;
}

/// <summary>The purpose of this function was undocumented.</summary>
///
__kernel void interpolate(__global real_t* new_x,
                          __global complex_t* new_y,
                          __global complex_t* z,
                          __global complex_t* y,
                          int N,
                          int new_N)
{
    size_t idx = get_global_id(0);
    real_t alpha = 1. / 6.;
    complex_t result;

    if (idx < new_N)
    {
        real_t x = new_x[idx];
    
        //get index of interval
        int i = floor(x);

        real_t dist1 = x - i;
        real_t dist2 = i + 1 - x;

        new_y[idx] = dist1*(z[i + 1]*alpha*(dist1*dist1 - 1) + y[i + 1]) + dist2*(z[i]*alpha*(dist2*dist2 - 1) + y[i]);
    }
}
