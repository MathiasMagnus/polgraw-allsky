#pragma once

// Polgraw includes
#include <struct.h> // OpenCL_handles
#include <floats.h> // fft_complex, shift_real, xDatm_complex

// OpenCL includes
#include <CL/cl.h>  // cl_event, cl_int, cl_uint, cl_mem


/// <summary>The purpose of this function was undocumented.</summary>
/// <todo>Merge phase_mod_1 and phase_mod_2 via zeroing out result arrays initially.</todo>
///
cl_event phase_mod_1(const cl_int idet,
                     const cl_int id,
                     const cl_int N,
                     const double het1,
                     const signal_params_t sgnlt1,
                     const cl_mem xar,
                     const cl_mem xbr,
                     const cl_mem shft,
                     cl_mem xa,
                     cl_mem xb,
                     OpenCL_handles* cl_handles,
                     const cl_uint num_events_in_wait_list,
                     const cl_event* event_wait_list);

/// <summary>The purpose of this function was undocumented.</summary>
/// <todo>Merge phase_mod_1 and phase_mod_2 via zeroing out result arrays initially.</todo>
///
cl_event phase_mod_2(const cl_int idet,
                     const cl_int id,
                     const cl_int N,
                     const double het1,
                     const signal_params_t sgnlt1,
                     const cl_mem xar,
                     const cl_mem xbr,
                     const cl_mem shft,
                     cl_mem xa,
                     cl_mem xb,
                     OpenCL_handles* cl_handles,
                     const cl_uint num_events_in_wait_list,
                     const cl_event* event_wait_list);
