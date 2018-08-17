#ifndef __SPLINE_Z_H__
#define __SPLINE_Z_H__

// Polgraw includes
#include <struct.h>     // OpenCL_handles

// OpenCL includes
#include <CL/cl.h>      // cl_mem, cl_int


/// <summary>Initialize the spline matrices.</summary>
/// <remarks>PCI Should replace it with kernels that initialize on the device.</remarks>
///
void init_spline_matrices(OpenCL_handles* cl_handles, 
                          cl_mem* cu_d,  // buffer of complex_devt
                          cl_mem* cu_dl, // buffer of complex_devt
                          cl_mem* cu_du, // buffer of complex_devt
                          cl_mem* cu_B,  // buffer of complex_devt
                          cl_int N);

/// <summary>Spline interpolation to xDatma, xDatmb arrays.</summary>
/// <note>Algorithm taken from Wikipedia: https://en.wikipedia.org/w/index.php?title=Spline_interpolation&oldid=816067429 </note>
/// <todo>Look for OpenCL accelerated tridiagonal solver</todo>
/// <todo>Merge interpolating ifo.sig.xDatma and ifo.sig.xDatmb in one invocation (not necessarily one kernel launch)</todo>
///
void gpu_interp(cl_mem cu_y,                // buffer of complex_t
                cl_int N,
                cl_mem cu_new_x,            // buffer of real_t
                cl_mem cu_new_y,            // buffer of complex_t
                cl_int new_N,
                cl_mem cu_d,                // buffer of complex_t
                cl_mem cu_dl,               // buffer of complex_t
                cl_mem cu_du,               // buffer of complex_t
                cl_mem cu_B,                // buffer of complex_t
                OpenCL_handles* cl_handles);// handles to OpenCL resources

/// <summary>Spline interpolation to xDatma, xDatmb arrays.</summary>
/// <note>Algorithm borrowed from gwsearch-cpu, until GPU version is implemented.</note>
///
void spline_interpolate_cpu(const cl_int idet,
                            const cl_int id,
                            const size_t arr_len,
	                        const size_t N,
	                        const int interpftpad,
	                        const real_t sig2,
	                        const cl_mem xa_d,
	                        const cl_mem xb_d,
	                        const cl_mem shftf_d,
	                        cl_mem xDatma_d,
	                        cl_mem xDatmb_d,
	                        BLAS_handles* blas_handles,
	                        OpenCL_handles* cl_handles,
	                        const cl_uint num_events_in_wait_list,
	                        const cl_event* event_wait_list,
	                        cl_event* spline_map_events,
	                        cl_event* spline_unmap_events,
	                        cl_event* spline_blas_events);

/// <summary>Scales vectors with a constant.</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
/// <remarks>Storage for the events must be provided by the caller.</remarks>
///
void blas_scale(const cl_int idet,
                const cl_int id,
                const size_t n,
                const real_t a,
                cl_mem xa_d,
                cl_mem xb_d,
                BLAS_handles* blas_handles,
                OpenCL_handles* cl_handles,
                const cl_uint num_events_in_wait_list,
                const cl_event* event_wait_list,
                cl_event* blas_exec);

/// <summary>The purpose of this function was undocumented.</summary>
///
void computeB_gpu(cl_mem y,
                  cl_mem B,
                  cl_int N,
                  OpenCL_handles* cl_handles);

/// <summary>Multiplies the tridiagonal matrix specified by <c>{dl, d, du}</c> with dense vector <c>x</c>.</summary>
///
void tridiagMul_gpu(cl_mem dl,
                    cl_mem d,
                    cl_mem du,
                    cl_mem x,
                    cl_int length,
                    OpenCL_handles* cl_handles);

/// <summary>The purpose of this function was undocumented.</summary>
///
void interpolate_gpu(cl_mem new_x,
                     cl_mem new_y,
                     cl_mem z,
                     cl_mem y,
                     cl_int N,
                     cl_int new_N,
                     OpenCL_handles* cl_handles);

#endif // __SPLINE_Z_H__
