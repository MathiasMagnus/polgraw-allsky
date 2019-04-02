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
      #pragma omp parallel for schedule(dynamic)
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
        //printf("id %d trying %d %d %d\n", omp_get_thread_num(), pm, mm, nn);

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
  Search_results results;
  results.sgnlc = 0;
  results.sgnlv = NULL;
  results.prof = init_profiling_info();

  // Allocate storage for events to synchronize pipeline
  Pipeline pl = init_pipeline(sett->nifo);

  signal_params_t sgnlt[NPAR], sgnl_freq;
  double het0, sinalt, cosalt, sindelt, cosdelt;

  struct timespec pre_spindown_start, pre_spindown_end, spindown_end;

  if (timespec_get(&pre_spindown_start, TIME_UTC) != TIME_UTC) { checkErr(TIME_UTC, "timespec_get(&pre_spindown_start, TIME_UTC)"); }

  // If the sky coordinate is not valid, shortcircuit the search
  if (!sky_positions(pm, mm, nn,                                   // input
                     sett->M, sett->oms, sett->sepsm, sett->cepsm, // input
                     sgnlt, &het0,                                 // output
                     &sinalt, &cosalt, &sindelt, &cosdelt))        // output
  {
    pre_spindown_end = spindown_end = pre_spindown_start;

    // Free OpenCL event object storage
    free_pipeline(sett->nifo, &pl);

    return results;
  }

  // Loop for each detector
  for (int n = 0; n<sett->nifo; ++n)
  {
    pl.modvir_events[n] =
      modvir(n, id, sett->N,                            // input
             sinalt, cosalt, sindelt, cosdelt,          // input
             ifo[n].sig.cphir, ifo[n].sig.sphir,        // input
             sett->omr, aux->ifo_amod_d[id],            // input
             ifo[n].sig.aa_d[id], ifo[n].sig.bb_d[id],  // output
             cl_handles, 0, NULL);                      // sync

    // Calculate detector positions with respect to baricenter
    cl_double3 nSource;
    nSource.s0 = cosalt * cosdelt;
    nSource.s1 = sinalt * cosdelt;
    nSource.s2 = sindelt;

    double shft1 = nSource.s[0] * ifo[n].sig.DetSSB[0].s[0] +
                   nSource.s[1] * ifo[n].sig.DetSSB[0].s[1] +
                   nSource.s[2] * ifo[n].sig.DetSSB[0].s[2];

    pl.tshift_pmod_events[n] =
      tshift_pmod(n, id, sett->N, sett->nfft, sett->interpftpad,                   // input
                  shft1, het0, sett->oms, nSource,                                 // input
                  ifo[n].sig.xDat_d[id], ifo[n].sig.aa_d[id], ifo[n].sig.bb_d[id], // input
                  ifo[n].sig.DetSSB_d[id],                                         // input
                  fft_arr->xa_d[id][n], fft_arr->xb_d[id][n],                      // output
                  ifo[n].sig.shft_d[id], ifo[n].sig.shftf_d[id],                   // output
                  aux->tshift_d[id][n],                                            // output
                  cl_handles, 1, &pl.modvir_events[n]);                            // sync

    fft_interpolate(n, id, sett->nfft, sett->Ninterp, sett->nyqst, plans,    // input
                    fft_arr->xa_d[id][n], fft_arr->xb_d[id][n],              // input / output
                    cl_handles, 1, &pl.tshift_pmod_events[n],                // sync
                    pl.fft_interpolate_fw_fft_events[n],                     // sync
                    pl.fft_interpolate_resample_copy_events[n],              // sync
                    pl.fft_interpolate_resample_fill_events[n],              // sync
                    pl.fft_interpolate_inv_fft_events[n]);                   // sync

    spline_interpolate(n, id, fft_arr->arr_len, sett->N, sett->interpftpad, ifo[n].sig.sig2,          // input
                       fft_arr->xa_d[id][n], fft_arr->xb_d[id][n], ifo[n].sig.shftf_d[id],            // input
                       ifo[n].sig.xDatma_d[id], ifo[n].sig.xDatmb_d[id],                              // output
                       blas_handles, cl_handles, 2, pl.fft_interpolate_inv_fft_events[n],             // sync
                       pl.spline_map_events[n], pl.spline_unmap_events[n], pl.spline_blas_events[n]); // sync

    blas_dot(n, id, sett->N, ifo[n].sig.aa_d[id], ifo[n].sig.bb_d[id], // input
             aux->aadots_d[id][n], aux->bbdots_d[id][n],               // output
             blas_handles, cl_handles, 1, &pl.modvir_events[n],        // sync
             pl.blas_dot_events[n]);                                   // sync

  } // end of detector loop

  calc_mxx(sett->nifo, id,                            // input
           aux->aadots_d[id], aux->bbdots_d[id], ifo, // input
           aux->maa_d[id], aux->mbb_d[id],            // output
           cl_handles, sett->nifo, pl.blas_dot_events,// sync
           pl.mxx_fill_events, pl.axpy_events);       // sync
    
  int smin, smax; // if spindown parameter is taken into account, smin != smax
  spindown_range(mm, nn, sett->Smin, sett->Smax, sett->M, // input
                 s_range, opts,                           // input
                 &smin, &smax);                           // output

  if (timespec_get(&pre_spindown_end, TIME_UTC) != TIME_UTC) { checkErr(TIME_UTC, "timespec_get(&pre_spindown_end, TIME_UTC)"); }
#ifndef VERBOSE
  printf(">>%d:\t%d\t%d\t%d\t[%d..%d]\n", id, pm, mm, nn, smin, smax);
#endif
  // Spindown loop
  for (int ss = smin; ss <= smax; ++ss)
  {
    sgnlt[spindown] = ss*sett->M[5] + nn*sett->M[9] + mm*sett->M[13];

    double het1 = fmod(ss*sett->M[4], sett->M[0]);

    if (het1<0) het1 += sett->M[0];

    sgnl_freq = het0 + het1; // are we reusing memory here? What does 'sgnl0' mean?

    // spline_interpolate_cpu is the last operation we should wait on for args to be ready
    pl.phase_mod_events[0] =
      phase_mod_1(0, id, sett->N, het1, sgnlt[spindown],                                   // input
                  ifo[0].sig.xDatma_d[id], ifo[0].sig.xDatmb_d[id], ifo[0].sig.shft_d[id], // input
                  fft_arr->xa_d[id][0], fft_arr->xb_d[id][0],                              // output
                  cl_handles, 5, pl.spline_unmap_events[0]);                               // sync
    
    for (int n = 1; n<sett->nifo; ++n)
    {
      // xY_d[id][0] is intentional, we're summing into the first xY_d arrays from xDatmY_d
        pl.phase_mod_events[n] =
        phase_mod_2(n, id, sett->N, het1, sgnlt[spindown],                                   // input
                    ifo[n].sig.xDatma_d[id], ifo[n].sig.xDatmb_d[id], ifo[n].sig.shft_d[id], // input
                    fft_arr->xa_d[id][0], fft_arr->xb_d[id][0],                              // output
                    cl_handles, 1, &pl.phase_mod_events[n - 1]);                             // sync
    }

  zero_pad(0, id, sett,                                      // input
           fft_arr->xa_d[id][0], fft_arr->xb_d[id][0],       // input / output
           cl_handles, sett->nifo, pl.phase_mod_events,      // sync
           pl.zero_pad_events);                              // sync

  time_to_frequency(0, id, sett, plans,                         // input
                    fft_arr->xa_d[id][0], fft_arr->xb_d[id][0], // input / output
                    cl_handles, 2, pl.zero_pad_events,          // sync
                    pl.fw2_fft_events);                         // sync

    (*FNum)++; // TODO: revisit this variable, needs atomic at least

    pl.compute_Fstat_event =
      compute_Fstat(0, id, sett->nmin, sett->nmax,
                    fft_arr->xa_d[id][0], fft_arr->xb_d[id][0], aux->maa_d[id], aux->mbb_d[id],
                    aux->F_d[id],
                    cl_handles, 2, pl.fw2_fft_events);

    pl.normalize_Fstat_event =
      normalize_FStat_wg_reduce(0, id, sett->nmin, sett->nmax, NAVFSTAT, // input
                                aux->F_d[id],                            // input / output
                                cl_handles, 1, &pl.compute_Fstat_event); // sync

    find_peaks(/*0,*/ id, sett->nmin, sett->nmax, opts->trl,    // input
               sgnl_freq, sett, aux->F_d[id],               // input
               &results, sgnlt,                             // output
               cl_handles, 1, &pl.normalize_Fstat_event,    // sync
               &pl.peak_map_event, &pl.peak_unmap_event);   // sync

    extract_spindown_profiling_info(sett->nifo,
                                    pl,
                                    &results.prof);

    release_spindown_events(sett->nifo, &pl);

  } // for ss

  if (timespec_get(&spindown_end, TIME_UTC) != TIME_UTC) { checkErr(TIME_UTC, "timespec_get(&spindown_end, TIME_UTC)"); }

  // Extract profiling info
  extract_non_spindown_profiling_info(sett->nifo,
                                      &pre_spindown_start,
                                      &pre_spindown_end,
                                      &spindown_end,
                                      pl, &results.prof);

#ifdef VERBOSE
    printf("Number of signals found: %zu\n", results.sgnlc);
#endif

  // Release OpenCL event objects and free their storage
    release_pipeline(sett->nifo, &pl);
    free_pipeline(sett->nifo, &pl);

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
  if (!strcmp(opts->addsig, "") && !strcmp(opts->range, ""))
  {
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
    char outname[2048];
    sprintf(outname, "%s/ocl_triggers_%03d_%03d%s_%d.bin",
        opts->prefix,
        opts->ident,
        opts->band,
        opts->label,
        pm);

    Search_results result = combine_results(s_range, (const Search_results**)results[pm - s_range->pmr[0]]);

    print_profiling_info(result.prof);

    // if any signals found (Fstat>Fc)
    if (result.sgnlc)
    {
      FILE* fc = fopen(outname, "w");
      if (fc == NULL) perror("Failed to open output file.");

      // Original
      //size_t count = fwrite((void *)(result.sgnlv),
      //                      sizeof(double),
      //                      result.sgnlc*NPAR,
      //                      fc);
      //if (count < result.sgnlc*NPAR) perror("Failed to write output file.");

      // Gnuplot friendly
      for (size_t i = 0; i < result.sgnlc; ++i)
      {
#ifdef _MSC_VER
        int out_count = fprintf_s(fc,
                                  "%e\t%e\t%e\t%e\t%e\n",
                                  result.sgnlv[i*NPAR + 0],
                                  result.sgnlv[i*NPAR + 1],
                                  result.sgnlv[i*NPAR + 2],
                                  result.sgnlv[i*NPAR + 3],
                                  result.sgnlv[i*NPAR + 4]);

        if (out_count < 0) checkErr(out_count, "fprintf_s");
#else
        fprintf(fc,
                "%e\t%e\t%e\t%e\t%e\n",
                result.sgnlv[i*NPAR + 0],
                result.sgnlv[i*NPAR + 1],
                result.sgnlv[i*NPAR + 2],
                result.sgnlv[i*NPAR + 3],
                result.sgnlv[i*NPAR + 4]);
#endif
      }

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
  Search_results result;
  result.sgnlc = 0;
  result.sgnlv = NULL;
  result.prof = init_profiling_info();

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

      result.prof.modvir_exec += select->prof.modvir_exec;
      result.prof.tshift_pmod_exec += select->prof.tshift_pmod_exec;
      result.prof.fft_interpolate_fw_fft_exec += select->prof.fft_interpolate_fw_fft_exec;
      result.prof.fft_interpolate_resample_copy_exec += select->prof.fft_interpolate_resample_copy_exec;
      result.prof.fft_interpolate_resample_fill_exec += select->prof.fft_interpolate_resample_fill_exec;
      result.prof.fft_interpolate_inv_fft_exec += select->prof.fft_interpolate_inv_fft_exec;
      result.prof.spline_map_exec += select->prof.spline_map_exec;
      result.prof.blas_dot_exec += select->prof.blas_dot_exec;
      result.prof.mxx_fill_exec += select->prof.mxx_fill_exec;
      result.prof.axpy_exec += select->prof.axpy_exec;

      result.prof.pre_spindown_exec += select->prof.pre_spindown_exec;
      result.prof.spindown_exec += select->prof.spindown_exec;

      result.prof.phase_mod_exec += select->prof.phase_mod_exec;
      result.prof.zero_pad_exec += select->prof.zero_pad_exec;
      result.prof.fw2_fft_exec += select->prof.fw2_fft_exec;
      result.prof.compute_Fstat_exec += select->prof.compute_Fstat_exec;
      result.prof.normalize_Fstat_exec += select->prof.normalize_Fstat_exec;
      result.prof.find_peak_exec += select->prof.find_peak_exec;

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

Pipeline init_pipeline(const size_t nifo)
{
  Pipeline p;

  p.modvir_events = (cl_event*)malloc(nifo * sizeof(cl_event));
  p.tshift_pmod_events = (cl_event*)malloc(nifo * sizeof(cl_event));
  p.fft_interpolate_fw_fft_events = (cl_event**)malloc(nifo * sizeof(cl_event*));
  p.fft_interpolate_resample_copy_events = (cl_event**)malloc(nifo * sizeof(cl_event*));
  p.fft_interpolate_resample_fill_events = (cl_event**)malloc(nifo * sizeof(cl_event*));
  p.fft_interpolate_inv_fft_events = (cl_event**)malloc(nifo * sizeof(cl_event*));
  p.spline_map_events = (cl_event**)malloc(nifo * sizeof(cl_event*));
  p.spline_unmap_events = (cl_event**)malloc(nifo * sizeof(cl_event*));
  p.spline_blas_events = (cl_event**)malloc(nifo * sizeof(cl_event*));
  p.blas_dot_events = (cl_event**)malloc(nifo * sizeof(cl_event*));
  p.mxx_fill_events = (cl_event*)malloc(2 * sizeof(cl_event));
  p.axpy_events = (cl_event*)malloc(nifo * 2 * sizeof(cl_event));
  p.phase_mod_events = (cl_event*)malloc(nifo * sizeof(cl_event));
  p.zero_pad_events = (cl_event*)malloc(2 * sizeof(cl_event));
  p.fw2_fft_events = (cl_event*)malloc(2 * sizeof(cl_event));
  for (size_t n = 0; n < nifo; ++n)
  {
    p.fft_interpolate_fw_fft_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
    p.fft_interpolate_resample_copy_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
    p.fft_interpolate_resample_fill_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
    p.fft_interpolate_inv_fft_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
    p.spline_map_events[n] = (cl_event*)malloc(5 * sizeof(cl_event));
    p.spline_unmap_events[n] = (cl_event*)malloc(5 * sizeof(cl_event));
    p.spline_blas_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
    p.blas_dot_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
  }

  return p;
}

void release_pipeline(const size_t nifo,
                      Pipeline* p)
{
  // Release OpenCL events
  for (size_t n = 0; n < nifo; ++n)
  {
    for (size_t m = 0; m < 2; ++m)
    {
      clReleaseEvent(p->fft_interpolate_fw_fft_events[n][m]);
      clReleaseEvent(p->fft_interpolate_resample_copy_events[n][m]);
      clReleaseEvent(p->fft_interpolate_resample_fill_events[n][m]);
      clReleaseEvent(p->fft_interpolate_inv_fft_events[n][m]);
    }
    for (size_t m = 0; m < 5; ++m)
    {
      clReleaseEvent(p->spline_map_events[n][m]);
      clReleaseEvent(p->spline_unmap_events[n][m]);
    }
    for (size_t m = 0; m < 2; ++m)
    {
      clReleaseEvent(p->spline_blas_events[n][m]);
      clReleaseEvent(p->blas_dot_events[n][m]);
    }
  }
}

void free_pipeline(const size_t nifo,
                   Pipeline* p)
{
  // Free host-side memory
  for (size_t n = 0; n < nifo; ++n)
  {
    free(p->fft_interpolate_fw_fft_events[n]);
    free(p->fft_interpolate_resample_copy_events[n]);
    free(p->fft_interpolate_resample_fill_events[n]);
    free(p->fft_interpolate_inv_fft_events[n]);
    free(p->spline_map_events[n]);
    free(p->spline_unmap_events[n]);
    free(p->spline_blas_events[n]);
    free(p->blas_dot_events[n]);
  }
  free(p->modvir_events);
  free(p->tshift_pmod_events);
  free(p->fft_interpolate_fw_fft_events);
  free(p->fft_interpolate_resample_copy_events);
  free(p->fft_interpolate_resample_fill_events);
  free(p->fft_interpolate_inv_fft_events);
  free(p->spline_map_events);
  free(p->spline_unmap_events);
  free(p->spline_blas_events);
  free(p->blas_dot_events);
  free(p->mxx_fill_events);
  free(p->axpy_events);
  free(p->phase_mod_events);
  free(p->zero_pad_events);
  free(p->fw2_fft_events);
}

void release_spindown_events(const size_t nifo,
                             Pipeline* p)
{
  cl_int CL_err = CL_SUCCESS;

  for (size_t n = 0; n < nifo; ++n)
  {
    CL_err = clReleaseEvent(p->phase_mod_events[n]); checkErr(CL_err, "clReleaseEvent(phase_mod_events)");
  }

  for (size_t m = 0; m < 2; ++m)
  {
    CL_err = clReleaseEvent(p->zero_pad_events[m]); checkErr(CL_err, "clReleaseEvent(zero_pad_events)");
    CL_err = clReleaseEvent(p->fw2_fft_events[m]); checkErr(CL_err, "clReleaseEvent(fw2_fft_events)");
  }

  CL_err = clReleaseEvent(p->compute_Fstat_event); checkErr(CL_err, "clReleaseEvent(compute_Fstat_event)");
  CL_err = clReleaseEvent(p->normalize_Fstat_event); checkErr(CL_err, "clReleaseEvent(normalize_Fstat_event)");
  CL_err = clReleaseEvent(p->peak_map_event); checkErr(CL_err, "clReleaseEvent(peak_map_event)");
  CL_err = clReleaseEvent(p->peak_unmap_event); checkErr(CL_err, "clReleaseEvent(peak_unmap_event)");
}

Profiling_info init_profiling_info()
{
  Profiling_info result = {
    0, // modvir_exec,
    0, // tshift_pmod_exec,
    0, // fft_interpolate_fw_fft_exec,
    0, // fft_interpolate_resample_copy_exec,
    0, // fft_interpolate_resample_fill_exec,
    0, // fft_interpolate_inv_fft_exec,
    0, // spline_map_exec,
    0, // spline_unmap_exec,
    0, // spline_blas_exec,
    0, // blas_dot_exec,
    0, // mxx_fill_exec,
    0, // axpy_exec,
    0, // phase_mod_exec,
    0, // zero_pad_exec,
    0, // fw2_fft_exec,
    0, // compute_Fstat_exec,
    0, // normalize_Fstat_exec,
    0, // find_peak_exec
    0, // pre_spindown_exec
    0  // spindown_exec
  };

  return result;
}

void extract_non_spindown_profiling_info(const size_t nifo,
                                         const struct timespec* pre_spindown_start,
                                         const struct timespec* pre_spindown_end,
                                         const struct timespec* spindown_end,
                                         const Pipeline pipeline,
                                         Profiling_info* prof)
{
  cl_ulong start, end;
  size_t info_size;
  cl_int CL_err = CL_SUCCESS;

  for (size_t n = 0; n < nifo; ++n)
  {
      CL_err = clGetEventProfilingInfo(pipeline.modvir_events[n], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
      CL_err = clGetEventProfilingInfo(pipeline.modvir_events[n], CL_PROFILING_COMMAND_END  , sizeof(cl_ulong), &end,   &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
      prof->modvir_exec += end - start;

      CL_err = clGetEventProfilingInfo(pipeline.tshift_pmod_events[n], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
      CL_err = clGetEventProfilingInfo(pipeline.tshift_pmod_events[n], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
      prof->tshift_pmod_exec += end - start;

      for (size_t m = 0; m < 2; ++m)
      {
        CL_err = clGetEventProfilingInfo(pipeline.fft_interpolate_fw_fft_events[n][m], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
        CL_err = clGetEventProfilingInfo(pipeline.fft_interpolate_fw_fft_events[n][m], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
        prof->fft_interpolate_fw_fft_exec += end - start;
      }

      CL_err = clGetEventProfilingInfo(pipeline.fft_interpolate_resample_copy_events[n][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
      CL_err = clGetEventProfilingInfo(pipeline.fft_interpolate_resample_copy_events[n][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
      prof->fft_interpolate_resample_copy_exec += end - start;

      for (size_t m = 0; m < 2; ++m)
      {
          CL_err = clGetEventProfilingInfo(pipeline.fft_interpolate_inv_fft_events[n][m], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
          CL_err = clGetEventProfilingInfo(pipeline.fft_interpolate_inv_fft_events[n][m], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
          prof->fft_interpolate_inv_fft_exec += end - start;
      }

      cl_ulong earliest = CL_ULONG_MAX,
               latest = 0;
      for (size_t m = 0; m < 5; ++m)
      {
          CL_err = clGetEventProfilingInfo(pipeline.spline_map_events[n][m], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
          if (start < earliest) earliest = start;

          CL_err = clGetEventProfilingInfo(pipeline.spline_unmap_events[n][m], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
          if (end > latest) latest = end;
      }
      prof->spline_map_exec += latest - earliest;

      for (size_t m = 0; m < 2; ++m)
      {
          CL_err = clGetEventProfilingInfo(pipeline.blas_dot_events[n][m], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
          CL_err = clGetEventProfilingInfo(pipeline.blas_dot_events[n][m], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
          prof->blas_dot_exec += end - start;
      }
  }
  for (size_t n = 0; n < 2; ++n)
  {
      CL_err = clGetEventProfilingInfo(pipeline.mxx_fill_events[n], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
      CL_err = clGetEventProfilingInfo(pipeline.mxx_fill_events[n], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
      prof->mxx_fill_exec += end - start;
  }
  for (size_t n = 0; n < 2 * nifo; ++n)
  {
      CL_err = clGetEventProfilingInfo(pipeline.axpy_events[n], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
      CL_err = clGetEventProfilingInfo(pipeline.axpy_events[n], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
      prof->axpy_exec += end - start;
  }

  prof->pre_spindown_exec =
    (pre_spindown_end->tv_sec - pre_spindown_start->tv_sec) * 1000000000 +
    (pre_spindown_end->tv_nsec - pre_spindown_start->tv_nsec);

  prof->spindown_exec =
    (spindown_end->tv_sec - pre_spindown_end->tv_sec) * 1000000000 +
    (spindown_end->tv_nsec - pre_spindown_end->tv_nsec);
}

void extract_spindown_profiling_info(const size_t nifo,
                                     const Pipeline pipeline,
                                     Profiling_info* prof)
{
  cl_ulong start, end;
  size_t info_size;
  cl_int CL_err = CL_SUCCESS;

  for (size_t n = 0; n < nifo; ++n)
  {
    CL_err = clGetEventProfilingInfo(pipeline.phase_mod_events[n], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
    CL_err = clGetEventProfilingInfo(pipeline.phase_mod_events[n], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
    prof->phase_mod_exec += end - start;
  }

  for (size_t m = 0; m < 2; ++m)
  {
    CL_err = clGetEventProfilingInfo(pipeline.zero_pad_events[m], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
    CL_err = clGetEventProfilingInfo(pipeline.zero_pad_events[m], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
    prof->zero_pad_exec += end - start;

    CL_err = clGetEventProfilingInfo(pipeline.fw2_fft_events[m], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
    CL_err = clGetEventProfilingInfo(pipeline.fw2_fft_events[m], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
    prof->fw2_fft_exec += end - start;
  }

  CL_err = clGetEventProfilingInfo(pipeline.compute_Fstat_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
  CL_err = clGetEventProfilingInfo(pipeline.compute_Fstat_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
  prof->compute_Fstat_exec += end - start;

  CL_err = clGetEventProfilingInfo(pipeline.normalize_Fstat_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
  CL_err = clGetEventProfilingInfo(pipeline.normalize_Fstat_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
  prof->normalize_Fstat_exec += end - start;

  CL_err = clGetEventProfilingInfo(pipeline.peak_map_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
  CL_err = clGetEventProfilingInfo(pipeline.peak_unmap_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &info_size); checkErr(CL_err, "clGetEventProfilingInfo");
  prof->find_peak_exec += end - start;
}

void print_profiling_info(const Profiling_info prof)
{
    printf("Total pre-spindown calculation : %f seconds.\n", prof.pre_spindown_exec / 1000000000.);
    printf("Total spindown calculation     : %f seconds.\n", prof.spindown_exec / 1000000000.);
    printf("\n");
    printf("Pre-spindown details:\n\n");
    printf("\tModvir        : %f seconds.\n", prof.modvir_exec / 1000000000.);
    printf("\tTShift_pmod   : %f seconds.\n", prof.tshift_pmod_exec / 1000000000.);
    printf("\tFFT fw trans  : %f seconds.\n", prof.fft_interpolate_fw_fft_exec / 1000000000.);
    printf("\tFFT resample  : %f seconds.\n", prof.fft_interpolate_resample_copy_exec / 1000000000.);
    printf("\tFFT inv trans : %f seconds.\n", prof.fft_interpolate_inv_fft_exec / 1000000000.);
    printf("\tSpline interp : %f seconds.\n", prof.spline_map_exec / 1000000000.);
    printf("\tBLAS dot      : %f seconds.\n", prof.blas_dot_exec / 1000000000.);
    printf("\tCalc_mxx_fill : %f seconds.\n", prof.mxx_fill_exec / 1000000000.);
    printf("\tCalc_mxx_axpy : %f seconds.\n", prof.axpy_exec / 1000000000.);
    printf("\n");
    printf("Spindown details:\n\n");
    printf("\tPhase mod     : %f seconds.\n", prof.phase_mod_exec / 1000000000.);
    printf("\tZero pad      : %f seconds.\n", prof.zero_pad_exec / 1000000000.);
    printf("\tTime to freq  : %f seconds.\n", prof.fw2_fft_exec / 1000000000.);
    printf("\tCompute FStat : %f seconds.\n", prof.compute_Fstat_exec / 1000000000.);
    printf("\tNorm FStat    : %f seconds.\n", prof.normalize_Fstat_exec / 1000000000.);
    printf("\tFind peaks    : %f seconds.\n", prof.find_peak_exec / 1000000000.);
}
