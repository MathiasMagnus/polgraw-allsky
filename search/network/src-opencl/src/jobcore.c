// C behavioral defines
//
// MSVC: macro to include constants, such as M_PI (include before math.h)
#define _USE_MATH_DEFINES
// ISO: request safe versions of functions
#define __STDC_WANT_LIB_EXT1__ 1
// GCC: hope this macro is not actually needed
//#define _GNU_SOURCE

// OpenCL behavioral defines
//
// 1.2+ OpenCL headers: tells the headers not to bitch about clCreateCommandQueue being renamed to clCreateCommandQueueWithProperties
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
//
// Select API to use
#define CL_TARGET_OPENCL_VERSION 120

// Polgraw includes
#include <CL/util.h>        // checkErr
#include <signal_params.h>
#include <sky_positions.h>
#include <modvir.h>
#include <tshift_pmod.h>
#include <fft_interpolate.h>
#include <spline_interpolate.h>
#include <blas_dot.h>
#include <calc_mxx.h>
#include <phase_mod.h>
#include <time_to_frequency.h>
#include <fstat.h>
#include <find_peaks.h>
#include <struct.h>
#include <jobcore.h>
#include <auxi.h>
#include <settings.h>
#include <timer.h>
//#include <spline_z.h>
#include <floats.h>

// clFFT includes
#include <clFFT.h>

// OpenCL includes
#include <CL/cl.h>

// OpenMP includes
#include <omp.h>

// Posix includes
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>             // _chsize_s
#include <direct.h>
#include <dirent.h>
#include <getopt.h>
#else
#include <unistd.h>         // ftruncate
#include <dirent.h>
#include <getopt.h>
#endif // WIN32

// Standard C includes
#include <math.h>
#include <stdio.h>          // fopen/fclose, fprintf
#include <malloc.h>
#include <complex.h>
#include <string.h>         // memcpy_s
#include <errno.h>          // errno_t
#include <stdlib.h>         // EXIT_FAILURE
#include <stdbool.h>        // TRUE
#include <assert.h>         // assert


/// <summary>Main searching function.</summary>
/// <remarks>This function loops over hemispheres, sky positions and spin-downs.</remarks>
///
void search(Detector_settings* ifo,
            Search_settings* sett,
            Command_line_opts* opts,
            Search_range* s_range,
            OpenCL_handles* cl_handles,
            BLAS_handles* blas_handles,
            FFT_plans* plans,
            FFT_arrays* fft_arr,
            Aux_arrays* aux,
            int* Fnum)
{
//#ifdef _WIN32
//  int low_state;
//#endif // WIN32
//  FILE* state = NULL;
//
//  if (opts->checkp_flag)
//  {
//#ifdef _WIN32
//    _sopen_s(&low_state, opts->qname,
//             _O_RDWR | _O_CREAT,   // Allowed operations
//             _SH_DENYNO,           // Allowed sharing
//             _S_IREAD | _S_IWRITE);// Permission settings
//
//    state = _fdopen(low_state, "w");
//#else
//    state = fopen(opts->qname, "w");
//#endif // WIN32
//  }

  Search_results*** results = init_results(s_range);

  // Loop over hemispheres //
  for (int pm = s_range->pst; pm <= s_range->pmr[1]; ++pm)
  {
    // Two main loops over sky positions //
    for (int mm = s_range->mst; mm <= s_range->mr[1]; ++mm)
    {
      int nn;
      #pragma omp parallel for
      for (nn = s_range->nst; nn <= s_range->nr[1]; ++nn)
      {
//        if (opts->checkp_flag)
//        {
//#ifdef _WIN32
//          if (_chsize(low_state, 0))
//          {
//            printf("Failed to resize file");
//            exit(EXIT_FAILURE);
//          }
//#else
//          if (ftruncate(fileno(state), 0))
//          {
//            printf("Failed to resize file");
//            exit(EXIT_FAILURE);
//          }
//#endif // _WIN32
//          fprintf(state, "%d %d %d %d %d\n", pm, mm, nn, s_range->sst, *Fnum);
//          fseek(state, 0, SEEK_SET);
//        }

        // Loop over spindowns is inside job_core() //
        results[pm - s_range->pmr[0]]
               [mm - s_range->mr[0]]
               [nn - s_range->nr[0]] = job_core(pm,            // hemisphere
                                                mm,            // grid 'sky position'
                                                nn,            // other grid 'sky position'
                                                omp_get_thread_num(),
                                                ifo,           // detector settings
                                                sett,          // search settings
                                                opts,          // cmd opts
                                                s_range,       // range for searching
                                                plans,         // fftw plans 
                                                fft_arr,       // arrays for fftw
                                                aux,           // auxiliary arrays
                                                Fnum,          // Candidate signal number
                                                cl_handles,    // handles to OpenCL resources
                                                blas_handles); // handle for scaling

        // Get back to regular spin-down range
        //
        // NOTE: without checkpoint support, writing this shared variable breaks concurrency
        //
        // s_range->sst = s_range->spndr[0];

      } // for nn
      s_range->nst = s_range->nr[0];
    } // for mm
    s_range->mst = s_range->mr[0];
  } // for pm

  save_and_free_results(opts, s_range, results);

//  if (opts->checkp_flag)
//    fclose(state);

}

Search_results job_core(const int pm,                  // hemisphere
                        const int mm,                  // grid 'sky position'
                        const int nn,                  // other grid 'sky position'
                        const int id,                  // device id
                        Detector_settings* ifo,        // detector settings
                        Search_settings *sett,         // search settings
                        Command_line_opts *opts,       // cmd opts
                        Search_range *s_range,         // range for searching
                        FFT_plans *plans,              // plans for fftw
                        FFT_arrays *fft_arr,           // arrays for fftw
                        Aux_arrays *aux,               // auxiliary arrays
                        int *FNum,                     // candidate signal number
                        OpenCL_handles* cl_handles,    // handles to OpenCL resources
                        BLAS_handles* blas_handles)    // handle for scaling
{
  Search_results results = { 0, NULL };

  // Allocate storage for events to synchronize pipeline
  cl_event *modvir_events = (cl_event*)malloc(sett->nifo * sizeof(cl_event)),
	       *tshift_pmod_events = (cl_event*)malloc(sett->nifo * sizeof(cl_event)),
	       **fft_interpolate_fw_fft_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       **fft_interpolate_resample_copy_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
           **fft_interpolate_resample_fill_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       **fft_interpolate_inv_fft_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       **spline_map_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       **spline_unmap_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       **spline_blas_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       **blas_dot_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
           *mxx_fill_events = (cl_event*)malloc(2 * sizeof(cl_event)),
	       *axpy_events = (cl_event*)malloc(sett->nifo * 2 * sizeof(cl_event)),
           *phase_mod_events = (cl_event*)malloc(sett->nifo * sizeof(cl_event)),
           *zero_pad_events = (cl_event*)malloc(2 * sizeof(cl_event)),
	       *fw2_fft_events = (cl_event*)malloc(2 * sizeof(cl_event));
  for (int n = 0; n < sett->nifo; ++n)
  {
    fft_interpolate_fw_fft_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
    fft_interpolate_resample_copy_events[n] = (cl_event*)malloc(2 * sizeof(cl_event)),
    fft_interpolate_resample_fill_events[n] = (cl_event*)malloc(2 * sizeof(cl_event)),
    fft_interpolate_inv_fft_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
	spline_map_events[n] = (cl_event*)malloc(5 * sizeof(cl_event));
	spline_unmap_events[n] = (cl_event*)malloc(5 * sizeof(cl_event));
	spline_blas_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
	blas_dot_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
  }

  signal_params_t sgnlt[NPAR], sgnl_freq;
  double het0, ft, sinalt, cosalt, sindelt, cosdelt;

  sky_positions(pm, mm, nn,                                   // input
	            sett->M, sett->oms, sett->sepsm, sett->cepsm, // input
	            sgnlt, &het0, &ft,                            // output
	            &sinalt, &cosalt, &sindelt, &cosdelt);        // output

  // Loop for each detector
  for (int n = 0; n<sett->nifo; ++n)
  {
    modvir_events[n] =
      modvir(n, id, sett->N,                            // input
             sinalt, cosalt, sindelt, cosdelt,          // input
             ifo[n].sig.cphir, ifo[n].sig.sphir,        // input
             sett->omr, aux->ifo_amod_d,                // input
             ifo[n].sig.aa_d[id], ifo[n].sig.bb_d[id],  // output
             cl_handles, 0, NULL);                      // sync

    // Calculate detector positions with respect to baricenter
    cl_double3 nSource = { cosalt * cosdelt,
                           sinalt * cosdelt,
                           sindelt };
    double shft1 = nSource.s[0] * ifo[n].sig.DetSSB[0].s[0] +
                   nSource.s[1] * ifo[n].sig.DetSSB[0].s[1] +
                   nSource.s[2] * ifo[n].sig.DetSSB[0].s[2];

    tshift_pmod_events[n] =
      tshift_pmod(n, id, sett->N, sett->nfft, sett->interpftpad,               // input
                  shft1, het0, sett->oms, nSource,                             // input
                  ifo[n].sig.xDat_d, ifo[n].sig.aa_d[id], ifo[n].sig.bb_d[id], // input
                  ifo[n].sig.DetSSB_d,                                         // input
                  fft_arr->xa_d[id][n], fft_arr->xb_d[id][n],                  // output
                  ifo[n].sig.shft_d[id], ifo[n].sig.shftf_d[id],               // output
                  aux->tshift_d[id][n],                                        // output
                  cl_handles, 1, &modvir_events[n]);                           // sync

	fft_interpolate(n, id, sett->nfft, sett->Ninterp, sett->nyqst, plans,    // input
		            fft_arr->xa_d[id][n], fft_arr->xb_d[id][n],              // input / output
		            cl_handles, 1, &tshift_pmod_events[n],                   // sync
                    fft_interpolate_fw_fft_events[n],                        // sync
                    fft_interpolate_resample_copy_events[n],                 // sync
                    fft_interpolate_resample_fill_events[n],                 // sync
                    fft_interpolate_inv_fft_events[n]);                      // sync

	spline_interpolate(n, id, fft_arr->arr_len, sett->N, sett->interpftpad, ifo[n].sig.sig2, // input
		               fft_arr->xa_d[id][n], fft_arr->xb_d[id][n], ifo[n].sig.shftf_d[id],   // input
		               ifo[n].sig.xDatma_d[id], ifo[n].sig.xDatmb_d[id],                     // output
		               blas_handles, cl_handles, 2, fft_interpolate_inv_fft_events[n],       // sync
		               spline_map_events[n], spline_unmap_events[n], spline_blas_events[n]); // sync

	blas_dot(n, id, sett->N, ifo[n].sig.aa_d[id], ifo[n].sig.bb_d[id], // input
		     aux->aadots_d[id][n], aux->bbdots_d[id][n],               // output
		     blas_handles, cl_handles, 1, &modvir_events[n],           // sync
		     blas_dot_events[n]);                                      // sync
  } // end of detector loop

  calc_mxx(sett->nifo, id,                            // input
           aux->aadots_d[id], aux->bbdots_d[id], ifo, // input
           aux->maa_d[id], aux->mbb_d[id],            // output
           cl_handles, sett->nifo, blas_dot_events,   // sync
	       mxx_fill_events, axpy_events);             // sync
    
  int smin, smax; // if spindown parameter is taken into account, smin != smax
  spindown_range(mm, nn, sett->Smin, sett->Smax, sett->M, // input
                 s_range, opts,                           // input
                 &smin, &smax);                           // output

  printf("\n>>%zu\t%d\t%d\t[%d..%d]\n", results.sgnlc, mm, nn, smin, smax);

  // Spindown loop
  for (int ss = smin; ss <= smax; ++ss)
  {
    sgnlt[spindown] = ss*sett->M[5] + nn*sett->M[9] + mm*sett->M[13];

	double het1 = fmod(ss*sett->M[4], sett->M[0]);

    if (het1<0) het1 += sett->M[0];
	
    sgnl_freq = het0 + het1; // are we reusing memory here? What does 'sgnl0' mean?

	// spline_interpolate_cpu is the last operation we should wait on for args to be ready
	phase_mod_events[0] = 
      phase_mod_1(0, id, sett->N, het1, sgnlt[spindown],                                   // input
                  ifo[0].sig.xDatma_d[id], ifo[0].sig.xDatmb_d[id], ifo[0].sig.shft_d[id], // input
                  fft_arr->xa_d[id][0], fft_arr->xb_d[id][0],                              // output
                  cl_handles, 5, spline_unmap_events[0]);                                  // sync
    
    for (int n = 1; n<sett->nifo; ++n)
    {
      // xY_d[id][0] is intentional, we're summing into the first xY_d arrays from xDatmY_d
      phase_mod_events[n] =
        phase_mod_2(n, id, sett->N, het1, sgnlt[spindown],                                   // input
                    ifo[n].sig.xDatma_d[id], ifo[n].sig.xDatmb_d[id], ifo[n].sig.shft_d[id], // input
                    fft_arr->xa_d[id][0], fft_arr->xb_d[id][0],                              // output
                    cl_handles, 1, phase_mod_events + (n - 1));                              // sync
    }

	zero_pad(0, id, sett,                                      // input
             fft_arr->xa_d[id][0], fft_arr->xb_d[id][0],       // input / output
             cl_handles, sett->nifo, phase_mod_events,         // sync
             zero_pad_events);                                 // sync

	time_to_frequency(0, id, sett, plans,                         // input
                      fft_arr->xa_d[id][0], fft_arr->xb_d[id][0], // input / output
                      cl_handles, 2, zero_pad_events,             // sync
                      fw2_fft_events);                            // sync

    (*FNum)++; // TODO: revisit this variable, needs atomic at least

    cl_event compute_Fstat_event =
	  compute_Fstat(0, id, sett->nmin, sett->nmax,
                    fft_arr->xa_d[id][0], fft_arr->xb_d[id][0], aux->maa_d[id], aux->mbb_d[id],
                    aux->F_d[id],
                    cl_handles, 2, fw2_fft_events);

	cl_event normalize_Fstat_event =
		normalize_FStat_wg_reduce(0, id, sett->nmin, sett->nmax, NAVFSTAT, // input
                                  aux->F_d[id],                            // input / output
                                  cl_handles, 1, &compute_Fstat_event);    // sync

    cl_event peak_map_event, peak_unmap_event;
    find_peaks(0, id, sett->nmin, sett->nmax, opts->trl, sgnl_freq, sett, aux->F_d[id], // input
               &results, sgnlt,                                                     // output
               cl_handles, 1, &normalize_Fstat_event,                               // sync
               &peak_map_event, &peak_unmap_event);                                 // sync

    clReleaseEvent(peak_map_event); clReleaseEvent(peak_unmap_event);
  } // for ss 

#ifndef VERBOSE
    printf("Number of signals found: %zu\n", results.sgnlc);
#endif

  return results;

} // jobcore



void spindown_range(const int mm,                  // grid 'sky position'
                    const int nn,                  // other grid 'sky position'
                    const double Smin,
                    const double Smax,
                    const double* M,               // M matrix from grid point to linear coord
                    const Search_range* s_range,
                    const Command_line_opts *opts,
	                int* smin,
	                int* smax)
{
	// Check if the signal is added to the data 
	// or the range file is given:  
	// if not, proceed with the wide range of spindowns 
	// if yes, use smin = s_range->sst, smax = s_range->spndr[1]
	*smin = s_range->sst, *smax = s_range->spndr[1];
	if (!strcmp(opts->addsig, "") && !strcmp(opts->range, "")) {

		// Spindown range defined using Smin and Smax (settings.c)  
		*smin = (int)trunc((Smin - nn * M[9] - mm * M[13]) / M[5]);  // Cast is intentional and safe (silences warning).
		*smax = (int)trunc(-(nn * M[9] + mm * M[13] + Smax) / M[5]); // Cast is intentional and safe (silences warning).
	}

	// No-spindown calculations
	if (opts->s0_flag) *smin = *smax;
}


void save_and_free_results(const Command_line_opts* opts,
                           const Search_range* s_range,
                           Search_results*** results)
{
  // Loop over hemispheres //
  for (int pm = s_range->pmr[0]; pm <= s_range->pmr[1]; ++pm)
  {
    char outname[512];
    sprintf(outname, "%s/triggers_%03d_%03d%s_%d.bin",
        opts->prefix,
        opts->ident,
        opts->band,
        opts->label,
        pm);

    Search_results result = combine_results(s_range, results[pm - s_range->pmr[0]]);

    // if any signals found (Fstat>Fc)
    if (result.sgnlc)
    {
      FILE* fc = fopen(outname, "w");
      if (fc == NULL) perror("Failed to open output file.");

      size_t count = fwrite((void *)(result.sgnlv),
                            sizeof(double),
                            result.sgnlc*NPAR,
                            fc);
      if (count < result.sgnlc*NPAR) perror("Failed to write output file.");

      int close = fclose(fc);
      if (close == EOF) perror("Failed to close output file.");
    } // if sgnlc

    free(result.sgnlv);
  }

  // Free resources
  for (int pm = s_range->pmr[0]; pm <= s_range->pmr[1]; ++pm)
  {
    for (int mm = s_range->mr[0]; mm <= s_range->mr[1]; ++mm)
    {
      for (int nn = s_range->nr[0]; nn <= s_range->nr[1]; ++nn)
      {
        free(results[pm - s_range->pmr[0]]
                    [mm - s_range->mr[0]]
                    [nn - s_range->nr[0]].sgnlv);
      }
      free(results[pm - s_range->pmr[0]]
                  [mm - s_range->mr[0]]);
    }
    free(results[pm - s_range->pmr[0]]);
  }
  free(results);
}

Search_results combine_results(const Search_range* s_range,
                               const Search_results** results)
{
  Search_results result = { 0, NULL };

  // Two main loops over sky positions //
  for (int mm = s_range->mr[0]; mm <= s_range->mr[1]; ++mm)
  {
    for (int nn = s_range->nr[0]; nn <= s_range->nr[1]; ++nn)
    {
        const Search_results* select = &results[mm - s_range->mr[0]]
                                               [nn - s_range->nr[0]];

        // Add new parameters to output array
        size_t old_sgnlc = result.sgnlc;
        result.sgnlc += select->sgnlc;

        result.sgnlv = (double*)realloc(result.sgnlv,
                                         (result.sgnlc) * NPAR * sizeof(double));
        memcpy(result.sgnlv + (old_sgnlc * NPAR),
               select->sgnlv,
               select->sgnlc * NPAR * sizeof(double));
    } // for nn

  } // for mm

  return result;
}

Search_results*** init_results(const Search_range* s_range)
{
  // NOTE 1: Search ranges are inclusive (no empty range possible), hence +1 is needed.
  Search_results*** results = (Search_results***)malloc((s_range->pmr[1] - s_range->pmr[0] + 1) * sizeof(Search_results**));

  for (int pm = s_range->pmr[0]; pm <= s_range->pmr[1]; ++pm)
  {
    // NOTE 2: When using coordinates to index into the arrays, subtract the base coordinate value.
    results[pm -  s_range->pmr[0]] = (Search_results**)malloc((s_range->mr[1] - s_range->mr[0] + 1) * sizeof(Search_results*));
    
    for (int mm = s_range->mr[0]; mm <= s_range->mr[1]; ++mm)
    {
      results[pm - s_range->pmr[0]]
             [mm - s_range->mr[0]] = (Search_results*)malloc((s_range->nr[1] - s_range->nr[0] + 1) * sizeof(Search_results));
    }
  }

  for (int pm = s_range->pst; pm >= s_range->pmr[0]; --pm)
  {
    for (int mm = s_range->mst; mm >= s_range->mr[0]; --mm)
    {
      for (int nn = s_range->nst; nn > s_range->nr[0]; --nn)
      {
        Search_results* select =  &results[pm - s_range->pmr[0]]
                                          [mm - s_range->mr[0]]
                                          [nn - s_range->nr[0]];
        select->sgnlc = 0;
        select->sgnlv = NULL;
      }
    }
  }

  return results;
}
