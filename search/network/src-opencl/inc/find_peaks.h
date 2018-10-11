#pragma once

// Polgraw includes
#include <struct.h>        // OpenCL_handles

// OpenCL includes
#include <CL/cl.h>         // cl_event, cl_int, cl_uint, cl_mem
#include <signal_params.h> // signal_params_t

/// <summary>Looks for peaks above treshold <c>trl</c> and persists them.</summary>
///
void find_peaks(const cl_int idet,
                const cl_int id,
                const cl_int nmin,
                const cl_int nmax,
                const double trl,
                const signal_params_t sgnl_freq,
                const Search_settings *sett,
                const cl_mem F_d,
                Search_results* results,
                signal_params_t* sgnlt,
                OpenCL_handles* cl_handles,
                const cl_uint num_events_in_wait_list,
                const cl_event* event_wait_list,
                cl_event* peak_map_event,
                cl_event* peak_unmap_event);
