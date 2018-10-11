#pragma once

// Polgraw includes
#include <struct.h> // OpenCL_handles

// OpenCL includes
#include <CL/cl.h>  // cl_event, cl_int, cl_uint, cl_mem

/// <summary>Compute F-statistics.</summary>
/// 
cl_event compute_Fstat(const cl_int idet,
                       const cl_int id,
                       const cl_int nmin,
                       const cl_int nmax,
                       const cl_mem xa_d,
                       const cl_mem xb_d,
                       const cl_mem maa_d,
                       const cl_mem mbb_d,
                       cl_mem F_d,
                       OpenCL_handles* cl_handles,
                       const cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list);

/// <summary>Normalize F-statistics.</summary>
///
cl_event normalize_FStat_wg_reduce(const cl_int idet,
                                   const cl_int id,
                                   const cl_int nmin,
                                   const cl_int nmax,
                                   const cl_uint nav,
                                   cl_mem F_d,
                                   OpenCL_handles* cl_handles,
                                   const cl_uint num_events_in_wait_list,
                                   const cl_event* event_wait_list);
