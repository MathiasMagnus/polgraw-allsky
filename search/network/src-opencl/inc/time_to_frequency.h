#pragma once

// Polgraw includes
#include <struct.h> // OpenCL_handles

// OpenCL includes
#include <CL/cl.h>  // cl_event, cl_int, cl_uint, cl_mem

/// <summary>Zero pad from offset until the end of the buffer.</summary>
///
void zero_pad(const cl_int idet,
              const cl_int id,
              const Search_settings *sett,
              cl_mem xa_d,
              cl_mem xb_d,
              OpenCL_handles* cl_handles,
              const cl_uint num_events_in_wait_list,
              const cl_event* event_wait_list,
              cl_event* zero_pad_events);

/// <summary>Transform data from time-domain to frequency domain for F-statistics.</summary>
///
void time_to_frequency(const cl_int idet,
                       const cl_int id,
                       const Search_settings *sett,
                       const FFT_plans* plans,
                       cl_mem xa_d,
                       cl_mem xb_d,
                       OpenCL_handles* cl_handles,
                       const cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* fw2_fft_events);
