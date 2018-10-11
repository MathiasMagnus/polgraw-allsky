#pragma once

// Polgraw includes
#include <struct.h> // OpenCL_handles
#include <floats.h> // fft_complex, shift_real, xDatm_complex

// OpenCL includes
#include <CL/cl.h>  // cl_event, cl_int, cl_uint, cl_mem


/// <summary>Calculates the inner product of both <c>x</c> and <c>y</c>.</summary>
///
void blas_dot(const cl_int idet,
              const cl_int id,
              const cl_uint n,
              const cl_mem aa_d,
              const cl_mem bb_d,
              cl_mem aadot_d,
              cl_mem bbdot_d,
              BLAS_handles* blas_handles,
              OpenCL_handles* cl_handles,
              const cl_uint num_events_in_wait_list,
              const cl_event* event_wait_list,
              cl_event* blas_exec);
