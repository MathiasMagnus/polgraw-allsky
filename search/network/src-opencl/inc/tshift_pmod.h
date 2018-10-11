#pragma once

// Polgraw includes
#include <struct.h> // OpenCL_handles

// OpenCL includes
#include <CL/cl.h>  // cl_event, cl_int, cl_uint, cl_mem

/// <summary>The purpose of this function was undocumented.</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
/// <remarks>Becomes a blocking call when <c>TESTING</c> is enabled</remarks>
///
cl_event tshift_pmod(const cl_int idet,
                     const cl_int id,
                     const cl_int N,
                     const cl_int nfft,
                     const cl_int interpftpad,
                     const double shft1,
                     const double het0,
                     const double oms,
                     const cl_double3 ns,
                     const cl_mem xDat_d,
                     const cl_mem aa_d,
                     const cl_mem bb_d,
                     const cl_mem DetSSB_d,
                     cl_mem xa_d,
                     cl_mem xb_d,
                     cl_mem shft_d,
                     cl_mem shftf_d,
                     cl_mem tshift_d,
                     const OpenCL_handles* cl_handles,
                     const cl_uint num_events_in_wait_list,
                     const cl_event* event_wait_list);
