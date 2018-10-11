// C behavioral defines
//
// MSVC: macro to include constants, such as M_PI (include before math.h)
#define _USE_MATH_DEFINES
// OpenCL behavioral defines
//
// 1.2+ OpenCL headers: tells the headers not to bitch about clCreateCommandQueue being renamed to clCreateCommandQueueWithProperties
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
//
// Select API to use
#define CL_TARGET_OPENCL_VERSION 120

#include <find_peaks.h>

// Polgraw includes
#include <floats.h>     // tshift_pmod_real, tshift_pmod_real3
#include <CL/util.h>    // checkErr

// Standard C includes
#include <math.h>       // sqrt, M_PI
#include <stdbool.h>    // _Bool, true, false

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
                cl_event* peak_unmap_event)
{
  cl_int CL_err = CL_SUCCESS;

  fstat_real* F = clEnqueueMapBuffer(cl_handles->read_queues[id][0],
                                     F_d,
                                     CL_TRUE,
                                     CL_MAP_READ,
                                     0,
                                     (nmax - nmin) * sizeof(fstat_real),
                                     num_events_in_wait_list,
                                     event_wait_list,
                                     peak_map_event,
                                     &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(F_d)");
  double Fc;
  for (int i = nmin; i < nmax; i++)
  {
    if ((Fc = F[i]) > trl) // if F-stat exceeds trl (critical value)
    {                      // Find local maximum for neighboring signals
      int ii = i;

      while (++i < nmax && F[i] > trl)
      {
        if (F[i] >= Fc)
        {
          ii = i;
          Fc = F[i];
        } // if F[i] 
      } // while i 

      sgnlt[frequency] = (signal_params_t)(2.*M_PI*ii / ((double)sett->fftpad*sett->nfft) + sgnl_freq);
      sgnlt[signal_to_noise] = (signal_params_t)sqrt(2.*(Fc - sett->nd));

      // Checking if signal is within a known instrumental line
      _Bool veto = false;
      for (int k = 0; k < sett->numlines_band; k++)
        if (sgnlt[frequency] >= sett->lines[k][0] && sgnlt[frequency] <= sett->lines[k][1]) {
          veto = true;
          break;
        }

      // If not vetoed, add new parameters to output array
      if (!veto)
      {
        results->sgnlc++; // increase found number
        results->sgnlv = (double*)realloc(results->sgnlv, NPAR*(results->sgnlc) * sizeof(double));

        for (int j = 0; j < NPAR; ++j) // save new parameters
          results->sgnlv[NPAR*(results->sgnlc - 1) + j] = (double)sgnlt[j];

#ifdef VERBOSE
        printf("\nSignal %d: %d %d %d %d %d \tsnr=%.2f\n",
               *sgnlc, pm, mm, nn, ss, ii, sgnlt[4]);
#endif 
      }
    }
  }

  CL_err = clEnqueueUnmapMemObject(cl_handles->read_queues[id][0],
                                   F_d,
                                   F,
                                   0,
                                   NULL,
                                   peak_unmap_event);
  checkErr(CL_err, "clEnqueueUnMapMemObject(F_d)");

  CL_err = clWaitForEvents(1, peak_unmap_event);
  checkErr(CL_err, "clWaitForEvents(peak_unmap_event)");
}
