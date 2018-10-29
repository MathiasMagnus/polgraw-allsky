#pragma once

// Polgraw includes
#include <struct.h> // OpenCL_handles

// OpenCL includes
#include <CL/cl.h>  // cl_event, cl_int, cl_uint, cl_mem

/// <summary>Interpolates in Fourier-space.</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
/// <remarks>Storage for the events must be provided by the caller.</remarks>
/// <todo>Create persistent storage for FFT temporary.</todo>
/// <todo>Remove manual component copying. Fix API call based version.</todo>
///
void fft_interpolate(const cl_int idet,
                     const cl_int id,
                     const cl_int nfft,
                     const cl_int Ninterp,
                     const cl_int nyqst,
                     const FFT_plans* plans,
                     cl_mem xa_d,
                     cl_mem xb_d,
                     OpenCL_handles* cl_handles,
                     const cl_uint num_events_in_wait_list,
                     const cl_event* event_wait_list,
                     cl_event* fw_fft_events,
                     cl_event* fft_interpolate_resample_copy_events,
                     cl_event* fft_interpolate_resample_fill_events,
                     cl_event* inv_fft_events);

/// <summary>Shifts frequencies and remove those over Nyquist.</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
///
void resample_postfft(const cl_int idet,
                      const cl_int id,
                      const cl_int nfft,
                      const cl_int Ninterp,
                      const cl_int nyqst,
                      cl_mem xa_d,
                      cl_mem xb_d,
                      OpenCL_handles* cl_handles,
                      const cl_uint num_events_in_wait_list,
                      const cl_event* event_wait_list,
                      cl_event* copy_events,
                      cl_event* fill_events);
