#pragma once

// Polgraw includes
#include <struct.h> // OpenCL_handles

// OpenCL includes
#include <CL/cl.h>  // cl_event, cl_int, cl_uint, cl_mem

/// <summary>Calculate the amplitude modulation functions aa and bb of a given detector (in signal sub-structs: ifo->sig.aa, ifo->.sig.bb).</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
/// <remarks>Becomes a blocking call when <c>TESTING</c> is enabled</remarks>
///
cl_event modvir(const cl_int idet,
                const cl_int id,
                const cl_int Np,
                const double sinal,
                const double cosal,
                const double sindel,
                const double cosdel,
                const double cphir,
                const double sphir,
                const double omr,
                const cl_mem ifo_amod_d,
                cl_mem aa_d,
                cl_mem bb_d,
                const OpenCL_handles* cl_handles,
                const cl_uint num_events_in_wait_list,
                const cl_event* event_wait_list);
