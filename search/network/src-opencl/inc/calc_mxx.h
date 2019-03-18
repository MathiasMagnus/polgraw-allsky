#pragma once

// Polgraw includes
#include <struct.h> // OpenCL_handles
#include <floats.h> // fft_complex, shift_real, xDatm_complex

// OpenCL includes
#include <CL/cl.h>  // cl_event, cl_int, cl_uint, cl_mem


/// <summary>Sum up results from all detectors.</summary>
/// <todo>Factor out the plain axpy call to separate function</todo>
/// <todo>Create an alternate implementation not doing a series of
///       scalar axpy on device, but mapping/unmapping and summing on host.</todo>
///
void calc_mxx(const cl_uint nifo,
              const cl_int id,
              const cl_mem* aadots_d,
              const cl_mem* bbdots_d,
              const Detector_settings* ifo,
              cl_mem maa_d,
              cl_mem mbb_d,
              OpenCL_handles* cl_handles,
              const cl_uint num_events_in_wait_list,
              cl_event** event_wait_list,
              cl_event* mxx_fill_events,
              cl_event* axpy_events);
