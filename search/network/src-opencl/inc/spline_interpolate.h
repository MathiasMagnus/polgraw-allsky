#pragma once

// Polgraw includes
#include <struct.h> // OpenCL_handles
#include <floats.h> // fft_complex, shift_real, xDatm_complex

// OpenCL includes
#include <CL/cl.h>  // cl_event, cl_int, cl_uint, cl_mem

/// <summary>Spline interpolation to xDatma, xDatmb arrays.</summary>
/// <note>Algorithm borrowed from gwsearch-cpu, until GPU version is implemented.</note>
///
void spline_interpolate(const cl_int idet,
                        const cl_int id,
                        const size_t arr_len,
                        const size_t N,
                        const int interpftpad,
                        const double sig2,
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

void splintpad(fft_complex* ya,
               shift_real* shftf,
               int N,
               int interpftpad,
               xDatm_complex* out);

void spline(const fft_complex* y,
            int n,
            spline_complex* y2,
            spline_complex* u);

xDatm_complex splint(fft_complex *ya,
                     fft_complex *y2a,
                     int n,
                     spline_real x);
