// C behavioral defines
//
// MSVC: macro to include constants, such as M_PI (include before math.h)
#define _USE_MATH_DEFINES
// ISO: request safe versions of functions
#define __STDC_WANT_LIB_EXT1__ 1
// GCC: hope this macro is not actually needed
//#define _GNU_SOURCE

// Polgraw includes
#include <CL/util.h>        // checkErr
#include <struct.h>
#include <jobcore.h>
#include <auxi.h>
#include <settings.h>
#include <timer.h>
#include <spline_z.h>
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

  // Copy amod coefficients to device
  copy_amod_coeff(ifo, sett->nifo, cl_handles, aux);

  Search_results*** results = init_results(s_range);

  // Loop over hemispheres //
  for (int pm = s_range->pst; pm <= s_range->pmr[1]; ++pm)
  {
    // Two main loops over sky positions //
    for (int mm = s_range->mst; mm <= s_range->mr[1]; ++mm)
    {
      #pragma omp parallel
      for (int nn = s_range->nst; nn <= s_range->nr[1]; ++nn)
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
	       **fw_fft_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       *resample_postfft_events = (cl_event*)malloc(sett->nifo * sizeof(cl_event)),
	       **inv_fft_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       **spline_map_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       **spline_unmap_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       **spline_blas_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       **blas_dot_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
           *mxx_fill_events = (cl_event*)malloc(2 * sizeof(cl_event)),
	       *axpy_events = (cl_event*)malloc(sett->nifo * 2 * sizeof(cl_event)),
           *phase_mod_events = (cl_event*)malloc(sett->nifo * sizeof(cl_event)),
           *zero_pad_events = (cl_event*)malloc(2 * sizeof(cl_event)),
	       *fw2_fft_events = (cl_event*)malloc(2 * sizeof(cl_event*));
  for (int n = 0; n < sett->nifo; ++n)
  {
	fw_fft_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
	inv_fft_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
	spline_map_events[n] = (cl_event*)malloc(5 * sizeof(cl_event));
	spline_unmap_events[n] = (cl_event*)malloc(5 * sizeof(cl_event));
	spline_blas_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
	blas_dot_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
  }

  real_t sgnlt[NPAR], het0, sgnl0, ft, sinalt, cosalt, sindelt, cosdelt;

  sky_positions(pm, mm, nn,                                   // input
	            sett->M, sett->oms, sett->sepsm, sett->cepsm, // input
	            sgnlt, &het0, &sgnl0, &ft,                    // output
	            &sinalt, &cosalt, &sindelt, &cosdelt);        // output

  // Stateful function (local variable with static storage duration)
  static real_t *F;
  if (F == NULL) F = (real_t*)malloc(2 * sett->nfft * sizeof(real_t));

  // Loop for each detector
  for (int n = 0; n<sett->nifo; ++n)
  {
    modvir_events[n] =
      modvir_gpu(n, id, sett->N,                                  // input
                 sinalt, cosalt, sindelt, cosdelt,                // input
                 ifo[n].sig.cphir, ifo[n].sig.sphir,              // input
                 aux->ifo_amod_d, aux->sinmodf_d, aux->cosmodf_d, // input
                 ifo[n].sig.aa_d[id], ifo[n].sig.bb_d[id],        // output
                 cl_handles, 0, NULL);                            // sync

    // Calculate detector positions with respect to baricenter
    real3_t nSource = { cosalt * cosdelt,
                        sinalt * cosdelt,
                        sindelt };
    real_t shft1 = nSource.s[0] * ifo[n].sig.DetSSB[0].s[0] +
                   nSource.s[1] * ifo[n].sig.DetSSB[0].s[1] +
                   nSource.s[2] * ifo[n].sig.DetSSB[0].s[2];

    tshift_pmod_events[n] =
      tshift_pmod_gpu(n, id, shft1, het0, nSource,                                 // input
                      sett->oms, sett->N, sett->nfft, sett->interpftpad,           // input
                      ifo[n].sig.xDat_d, ifo[n].sig.aa_d[id], ifo[n].sig.bb_d[id], // input
                      ifo[n].sig.DetSSB_d,                                         // input
                      fft_arr->xa_d[id][n], fft_arr->xb_d[id][n],                  // output
                      ifo[n].sig.shft_d[id], ifo[n].sig.shftf_d[id],               // output
                      aux->tshift_d[id][n],                                        // output
                      cl_handles, 1, &modvir_events[n]);                           // sync

	fft_interpolate_gpu(n, id, sett->nfft, sett->Ninterp, sett->nyqst, plans,    // input
		                fft_arr->xa_d[id][n], fft_arr->xb_d[id][n],              // input / output
		                cl_handles, 1, &tshift_pmod_events[n],                   // sync
		                fw_fft_events, resample_postfft_events, inv_fft_events); // sync

	spline_interpolate_cpu(n, id, fft_arr->arr_len, sett->N, sett->interpftpad, ifo[n].sig.sig2, // input
		                   fft_arr->xa_d[id][n], fft_arr->xb_d[id][n], ifo[n].sig.shftf_d[id],   // input
		                   ifo[n].sig.xDatma_d[id], ifo[n].sig.xDatmb_d[id],                     // output
		                   blas_handles, cl_handles, 2, inv_fft_events[n],                       // sync
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

  // Spindown loop
  for (int ss = smin; ss <= smax; ++ss)
  {
    // Spindown parameter
    sgnlt[1] = ss*sett->M[5] + nn*sett->M[9] + mm*sett->M[13];

	real_t het1 = fmod(ss*sett->M[4], sett->M[0]);

    if (het1<0) het1 += sett->M[0];
	
	sgnl0 = het0 + het1; // are we reusing memory here? What does 'sgnl0' mean?

	// spline_interpolate_cpu is the last operation we should wait on for args to be ready
	phase_mod_events[0] = 
      phase_mod_1_gpu(0, id, sett->N, het1, sgnlt[1],                                          // input
                      ifo[0].sig.xDatma_d[id], ifo[0].sig.xDatmb_d[id], ifo[0].sig.shft_d[id], // input
                      fft_arr->xa_d[id][0], fft_arr->xb_d[id][0],                              // output
                      cl_handles, 5, spline_unmap_events[0]);                                  // sync

    for (int n = 1; n<sett->nifo; ++n)
    {
      // xY_d[id][0] is intentional, we're summing into the first xY_d arrays from xDatmY_d
      phase_mod_events[n] =
        phase_mod_2_gpu(n, id, sett->N, het1, sgnlt[1],                                          // input
                        ifo[n].sig.xDatma_d[id], ifo[n].sig.xDatmb_d[id], ifo[n].sig.shft_d[id], // input
                        fft_arr->xa_d[id][0], fft_arr->xb_d[id][0],                              // output
                        cl_handles, 1, phase_mod_events + (n - 1));                              // sync
    }

	zero_pad(0, id, sett,                                      // input
             fft_arr->xa_d[id][0], fft_arr->xb_d[id][0],       // input / output
             cl_handles, 1, &phase_mod_events[sett->nifo - 1], // sync
             zero_pad_events);                                 // sync

	time_to_frequency(0, id, sett, plans,                         // input
                      fft_arr->xa_d[id][0], fft_arr->xb_d[id][0], // input / output
                      cl_handles, 2, zero_pad_events,             // sync
                      fw2_fft_events);                            // sync

    (*FNum)++; // TODO: revisit this variable, needs atomic at least

    cl_event compute_Fstat_event =
	  compute_Fstat_gpu(0, id, sett->nmin, sett->nmax,
                        fft_arr->xa_d[id][0], fft_arr->xb_d[id][0], aux->maa_d[id], aux->mbb_d[id],
                        aux->F_d[id],
                        cl_handles, 2, fw2_fft_events);

	cl_event normalize_Fstat_event =
		normalize_FStat_gpu_wg_reduce(0, id, sett->nmin, sett->nmax, NAVFSTAT, // input
                                      aux->F_d[id],                            // input / output
                                      cl_handles, 1, &compute_Fstat_event);    // sync

#if 0
	cl_int CL_err = CL_SUCCESS;

	CL_err = clEnqueueReadBuffer(cl_handles->read_queues[0], F_d, CL_TRUE, 0, 2 * sett->nfft * sizeof(real_t), F, 0, NULL, NULL);
        // Normalize F-statistics 
        if (!(opts->white_flag))  // if the noise is not white noise
            FStat(F + sett->nmin, sett->nmax - sett->nmin, NAVFSTAT, 0);
#endif

    cl_event peak_map_event, peak_unmap_event;
    find_peaks(0, id, sett->nmin, sett->nmax, opts->trl, sgnl0, sett, aux->F_d[id], // input
               &results, sgnlt,                                                     // output
               cl_handles, 1, &normalize_Fstat_event,                               // sync
               &peak_map_event, &peak_unmap_event);                                 // sync
  } // for ss 

  printf("\n>>%zu\t%d\t%d\t[%d..%d]\n", results.sgnlc, mm, nn, smin, smax);

#ifndef VERBOSE
    printf("Number of signals found: %zu\n", results.sgnlc);
#endif

  return results;

} // jobcore

void sky_positions(const int pm,                  // hemisphere
                   const int mm,                  // grid 'sky position'
                   const int nn,                  // other grid 'sky position'
                   double* M,                     // M matrix from grid point to linear coord
	               real_t oms,
	               real_t sepsm,
	               real_t cepsm,
                   real_t* sgnlt,
                   real_t* het0,
                   real_t* sgnl0,
                   real_t* ft,
	               real_t* sinalt,
	               real_t* cosalt,
	               real_t* sindelt,
	               real_t* cosdelt)
{
  /* Matrix M(.,.) (defined on page 22 of PolGrawCWAllSkyReview1.pdf file)
  defines the transformation form integers (bin, ss, nn, mm) determining
  a grid point to linear coordinates omega, omegadot, alpha_1, alpha_2),
  where bin is the frequency bin number and alpha_1 and alpha_2 are
  defined on p. 22 of PolGrawCWAllSkyReview1.pdf file.

  [omega]                          [bin]
  [omegadot]       = M(.,.) \times [ss]
  [alpha_1/omega]                  [nn]
  [alpha_2/omega]                  [mm]

  Array M[.] is related to matrix M(.,.) in the following way;

  [ M[0] M[4] M[8]  M[12] ]
  M(.,.) =   [ M[1] M[5] M[9]  M[13] ]
  [ M[2] M[6] M[10] M[14] ]
  [ M[3] M[7] M[11] M[15] ]

  and

  M[1] = M[2] = M[3] = M[6] = M[7] = 0
  */

  // Grid positions
  real_t al1 = nn * M[10] + mm * M[14],
         al2 = nn * M[11] + mm * M[15];

  // check if the search is in an appropriate region of the grid
  // if not, returns NULL
  //if ((sqr(al1) + sqr(al2)) / sqr(sett->oms) > 1.) return NULL;

  // Change linear (grid) coordinates to real coordinates
  lin2ast(al1 / oms, al2 / oms, pm, sepsm, cepsm, // input
          sinalt, cosalt, sindelt, cosdelt);      // output

  // calculate declination and right ascention
  // written in file as candidate signal sky positions
  sgnlt[2] = asin(*sindelt);
  sgnlt[3] = fmod(atan2(*sinalt, *cosalt) + 2.*M_PI, 2.*M_PI);

  *het0 = fmod(nn*M[8] + mm * M[12], M[0]);
}

void copy_amod_coeff(const Detector_settings* ifo,
                     const cl_int nifo,
                     OpenCL_handles* cl_handles,
                     Aux_arrays* aux)
{
  cl_int CL_err = CL_SUCCESS;

  Ampl_mod_coeff* tmp =
    clEnqueueMapBuffer(cl_handles->exec_queues[0][0],
                       aux->ifo_amod_d,
                       CL_TRUE,
                       CL_MAP_WRITE_INVALIDATE_REGION,
                       0,
                       nifo * sizeof(Ampl_mod_coeff),
                       0,
                       NULL,
                       NULL,
                       &CL_err);

  for (size_t i = 0; i < nifo; ++i) tmp[i] = ifo[i].amod;

  cl_event unmap_event;
  clEnqueueUnmapMemObject(cl_handles->exec_queues[0][0], aux->ifo_amod_d, tmp, 0, NULL, &unmap_event);
  
  clWaitForEvents(1, &unmap_event);
  
  clReleaseEvent(unmap_event);
}

cl_event modvir_gpu(const cl_int idet,
                    const cl_int id,
                    const cl_int Np,
                    const real_t sinal,
                    const real_t cosal,
                    const real_t sindel,
                    const real_t cosdel,
                    const real_t cphir,
                    const real_t sphir,
                    const cl_mem ifo_amod_d,
                    const cl_mem sinmodf_d,
                    const cl_mem cosmodf_d,
                    cl_mem aa_d,
                    cl_mem bb_d,
                    const OpenCL_handles* cl_handles,
                    const cl_uint num_events_in_wait_list,
                    const cl_event* event_wait_list)
{
  cl_int CL_err = CL_SUCCESS;
  real_t cosalfr = cosal * (cphir) + sinal * (sphir),
         sinalfr = sinal * (cphir) - cosal * (sphir),
         c2d = sqr(cosdel),
         c2sd = sindel * cosdel;

  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 0, sizeof(cl_mem), &aa_d);             checkErr(CL_err, "clSetKernelArg(&aa_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 1, sizeof(cl_mem), &bb_d);             checkErr(CL_err, "clSetKernelArg(&bb_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 2, sizeof(real_t), &cosalfr);          checkErr(CL_err, "clSetKernelArg(&cosalfr)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 3, sizeof(real_t), &sinalfr);          checkErr(CL_err, "clSetKernelArg(&sinalfr)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 4, sizeof(real_t), &c2d);              checkErr(CL_err, "clSetKernelArg(&c2d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 5, sizeof(real_t), &c2sd);             checkErr(CL_err, "clSetKernelArg(&c2sd)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 6, sizeof(cl_mem), &sinmodf_d);        checkErr(CL_err, "clSetKernelArg(&sinmodf_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 7, sizeof(cl_mem), &cosmodf_d);        checkErr(CL_err, "clSetKernelArg(&cosmodf_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 8, sizeof(real_t), &sindel);           checkErr(CL_err, "clSetKernelArg(&sindel)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 9, sizeof(real_t), &cosdel);           checkErr(CL_err, "clSetKernelArg(&cosdel)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 10, sizeof(cl_int), &Np);              checkErr(CL_err, "clSetKernelArg(&Np)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 11, sizeof(cl_int), &idet);            checkErr(CL_err, "clSetKernelArg(&idet)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 12, sizeof(cl_mem), &ifo_amod_d);      checkErr(CL_err, "clSetKernelArg(&ifo_amod_d)");

  cl_event exec;
  size_t size_Np = (size_t)Np; // Helper variable to make pointer types match. Cast to silence warning

  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][Modvir], 1, NULL, &size_Np, NULL, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(cl_handles->kernels[Modvir])");

#ifdef TESTING
  clWaitForEvents(1, &exec);
  save_numbered_real_buffer(cl_handles->read_queues[id][idet], sinmodf_d, Np, idet, "aux_sinmodf");
  save_numbered_real_buffer(cl_handles->read_queues[id][idet], cosmodf_d, Np, idet, "aux_cosmodf");
  save_numbered_real_buffer(cl_handles->read_queues[id][idet], aa_d, Np, idet, "ifo_sig_aa");
  save_numbered_real_buffer(cl_handles->read_queues[id][idet], bb_d, Np, idet, "ifo_sig_bb");
#endif

    return exec;
}

cl_event tshift_pmod_gpu(const cl_int idet,
                         const cl_int id,
	                     const real_t shft1,
                         const real_t het0,
                         const real3_t ns,
                         const real_t oms,
                         const cl_int N,
                         const cl_int nfft,
                         const cl_int interpftpad,
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
                         const cl_event* event_wait_list)
{
  cl_int CL_err = CL_SUCCESS;

  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 0, sizeof(real_t), &shft1);        checkErr(CL_err, "clSetKernelArg(&shft1)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 1, sizeof(real_t), &het0);         checkErr(CL_err, "clSetKernelArg(&het0)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 2, sizeof(real3_t), &ns);          checkErr(CL_err, "clSetKernelArg(&ns0)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 3, sizeof(cl_mem), &xDat_d);       checkErr(CL_err, "clSetKernelArg(&xDat_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 4, sizeof(cl_mem), &xa_d);         checkErr(CL_err, "clSetKernelArg(&xa_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 5, sizeof(cl_mem), &xb_d);         checkErr(CL_err, "clSetKernelArg(&xb_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 6, sizeof(cl_mem), &shft_d);       checkErr(CL_err, "clSetKernelArg(&shft_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 7, sizeof(cl_mem), &shftf_d);      checkErr(CL_err, "clSetKernelArg(&shftf_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 8, sizeof(cl_mem), &tshift_d);     checkErr(CL_err, "clSetKernelArg(&tshift_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 9, sizeof(cl_mem), &aa_d);         checkErr(CL_err, "clSetKernelArg(&aa_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 10, sizeof(cl_mem), &bb_d);        checkErr(CL_err, "clSetKernelArg(&bb_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 11, sizeof(cl_mem), &DetSSB_d);    checkErr(CL_err, "clSetKernelArg(&DetSSB_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 12, sizeof(real_t), &oms);         checkErr(CL_err, "clSetKernelArg(&oms)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 13, sizeof(cl_int), &N);           checkErr(CL_err, "clSetKernelArg(&N)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 14, sizeof(cl_int), &nfft);        checkErr(CL_err, "clSetKernelArg(&nfft)");
  CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 15, sizeof(cl_int), &interpftpad); checkErr(CL_err, "clSetKernelArg(&interftpad)");

  cl_event exec;
  size_t size_nfft = (size_t)nfft; // Helper variable to make pointer types match. Cast to silence warning

  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][TShiftPMod], 1, NULL, &size_nfft, NULL, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(cl_handles->kernels[TShiftPMod])");

#ifdef TESTING
  clWaitForEvents(1, &exec);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, 2 * nfft, idet, "xa_time");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, 2 * nfft, idet, "xb_time");
  save_numbered_real_buffer(cl_handles->read_queues[id][idet], shft_d, N, idet, "ifo_sig_shft");
  save_numbered_real_buffer(cl_handles->read_queues[id][idet], shftf_d, N, idet, "ifo_sig_shftf");
#endif

    return exec;
}

void fft_interpolate_gpu(const cl_int idet,
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
                         cl_event** fw_fft_events,
	                     cl_event* resample_postfft_events,
	                     cl_event** inv_fft_events)
{
	// Forward FFT
	clfftStatus CLFFT_status = CLFFT_SUCCESS;
	CLFFT_status = clfftEnqueueTransform(plans->pl_int, CLFFT_FORWARD, 1, cl_handles->exec_queues[id] + idet, num_events_in_wait_list, event_wait_list, &fw_fft_events[idet][0], &xa_d, NULL, NULL);
	checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");
	CLFFT_status = clfftEnqueueTransform(plans->pl_int, CLFFT_FORWARD, 1, cl_handles->exec_queues[id] + idet, num_events_in_wait_list, event_wait_list, &fw_fft_events[idet][1], &xb_d, NULL, NULL);
	checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");

#ifdef TESTING
	clWaitForEvents(2, fw_fft_events[idet]);
	save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, nfft, idet, "xa_fourier");
	save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, nfft, idet, "xb_fourier");
#endif

	// Resample coefficients
	resample_postfft_events[idet] =
		resample_postfft_gpu(idet, id, nfft, Ninterp, nyqst,      // input
		                     xa_d, xb_d,                          // input / output
		                     cl_handles, 2, fw_fft_events[idet]); // sync

	// Backward fft (len Ninterp = nfft*interpftpad)
	clfftEnqueueTransform(plans->pl_inv, CLFFT_BACKWARD, 1, cl_handles->exec_queues[id] + idet, 1, &resample_postfft_events[idet], &inv_fft_events[idet][0], &xa_d, NULL, NULL);
	checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_BACKWARD)");
	clfftEnqueueTransform(plans->pl_inv, CLFFT_BACKWARD, 1, cl_handles->exec_queues[id] + idet, 1, &resample_postfft_events[idet], &inv_fft_events[idet][1], &xb_d, NULL, NULL);
	checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_BACKWARD)");

	// scale fft with clblas not needed (as opposed fftw), clFFT already scales

#ifdef TESTING
	clWaitForEvents(2, inv_fft_events[idet]);
	save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, Ninterp, idet, "xa_time_resampled");
	save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, Ninterp, idet, "xb_time_resampled");
#endif
}

cl_event resample_postfft_gpu(const cl_int idet,
                              const cl_int id,
                              const cl_int nfft,
	                          const cl_int Ninterp,
	                          const cl_int nyqst,
	                          cl_mem xa_d,
	                          cl_mem xb_d,
	                          OpenCL_handles* cl_handles,
	                          const cl_uint num_events_in_wait_list,
	                          const cl_event* event_wait_list)
{
    cl_int CL_err = CL_SUCCESS;

    clSetKernelArg(cl_handles->kernels[id][ResamplePostFFT], 0, sizeof(cl_mem), &xa_d);    checkErr(CL_err, "clSetKernelArg(&xa_d)");
    clSetKernelArg(cl_handles->kernels[id][ResamplePostFFT], 1, sizeof(cl_mem), &xb_d);    checkErr(CL_err, "clSetKernelArg(&xb_d)");
    clSetKernelArg(cl_handles->kernels[id][ResamplePostFFT], 2, sizeof(cl_int), &nfft);    checkErr(CL_err, "clSetKernelArg(&nfft)");
    clSetKernelArg(cl_handles->kernels[id][ResamplePostFFT], 3, sizeof(cl_int), &Ninterp); checkErr(CL_err, "clSetKernelArg(&Ninterp)");
    clSetKernelArg(cl_handles->kernels[id][ResamplePostFFT], 4, sizeof(cl_int), &nyqst);   checkErr(CL_err, "clSetKernelArg(&nyqst)");

    cl_event exec;
    size_t resample_length = (size_t)Ninterp - (nyqst + nfft); // Helper variable to make pointer types match. Cast to silence warning

    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][ResamplePostFFT], 1, NULL, &resample_length, NULL, 0, NULL, &exec);
	checkErr(CL_err, "clEnqueueNDRangeKernel(cl_handles->kernels[ResamplePostFFT])");

#ifdef TESTING
	clWaitForEvents(1, &exec);
	save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, sett->Ninterp, idet, "xa_fourier_resampled");
	save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, sett->Ninterp, idet, "xb_fourier_resampled");
#endif

	return exec;
}

void blas_dot(const cl_int idet,
              const cl_int id,
              const cl_uint n,
	          const cl_mem aa_d,
              const cl_mem bb_d,
              cl_mem aadot_d,             
	          cl_mem bbdot_d,
	          BLAS_handles* blas_handles,
              OpenCL_handles* cl_handles,
	          const cl_uint num_events_in_wait_list,
	          const cl_event* event_wait_list,
	          cl_event* blas_exec)
{
  clblasStatus status[2] = { clblasSuccess, clblasSuccess };

#ifdef COMP_FLOAT
  status[1] = clblasSdot(n, bbdot_d, 0, bb_d, 0, 1, bb_d, 0, 1, blas_handles->bbScratch_d[id][idet], 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[1]); checkErrBLAS(status[0], "clblasDdot()");
  status[0] = clblasSdot(n, aadot_d, 0, aa_d, 0, 1, aa_d, 0, 1, blas_handles->aaScratch_d[id][idet], 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[0]); checkErrBLAS(status[1], "clblasDdot()");
#else
  status[0] = clblasDdot(n, aadot_d, 0, aa_d, 0, 1, aa_d, 0, 1, blas_handles->aaScratch_d[id][idet], 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[0]); checkErrBLAS(status[0], "clblasDdot()");
  status[1] = clblasDdot(n, bbdot_d, 0, bb_d, 0, 1, bb_d, 0, 1, blas_handles->bbScratch_d[id][idet], 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[1]); checkErrBLAS(status[1], "clblasDdot()");
#endif // COMP_FLOAT
}

void calc_mxx(const cl_uint nifo,
              const cl_int id,
	          const cl_mem* aadots_d,
	          const cl_mem* bbdots_d,
	          const Detector_settings* ifo,
	          cl_mem maa_d,
	          cl_mem mbb_d,
	          OpenCL_handles* cl_handles,
	          const cl_uint num_events_in_wait_list,
	          const cl_event** event_wait_list,
	          cl_event* mxx_fill_events,
	          cl_event* axpy_events)
{
	cl_int CL_err = CL_SUCCESS;
	clblasStatus status[2] = { clblasSuccess, clblasSuccess };

	real_t pattern = 0;

	CL_err = clEnqueueFillBuffer(cl_handles->write_queues[id][0], maa_d, &pattern, sizeof(real_t), 0, sizeof(real_t), 0, NULL, &mxx_fill_events[0]); checkErr(CL_err, "clEnqueueFillBuffer(maa_d)");
	CL_err = clEnqueueFillBuffer(cl_handles->write_queues[id][0], mbb_d, &pattern, sizeof(real_t), 0, sizeof(real_t), 0, NULL, &mxx_fill_events[1]); checkErr(CL_err, "clEnqueueFillBuffer(mbb_d)");

    cl_event* input_wait_events = (cl_event*)malloc((nifo * 2 + 2) * sizeof(cl_event));

    cl_uint i;
    for (i = 0; i < nifo; ++i)
    {
        input_wait_events[i * 2 + 0] = event_wait_list[i][0];
        input_wait_events[i * 2 + 1] = event_wait_list[i][1];
    }
    input_wait_events[nifo * 2 + 0] = mxx_fill_events[0];
    input_wait_events[nifo * 2 + 1] = mxx_fill_events[1];

#ifdef COMP_FLOAT
	status[0] = clblasSaxpy(1, 1 / ifo[0].sig.sig2, aadots_d[0], 0, 1, maa_d, 0, 1, 1, &cl_handles->exec_queues[id][0], (nifo * 2 + 2), input_wait_events, &axpy_events[0]); checkErrBLAS(status[0], "clblasDaxpy()");
	status[1] = clblasSaxpy(1, 1 / ifo[0].sig.sig2, bbdots_d[0], 0, 1, mbb_d, 0, 1, 1, &cl_handles->exec_queues[id][0], (nifo * 2 + 2), input_wait_events, &axpy_events[1]); checkErrBLAS(status[1], "clblasDaxpy()");
#else
	status[0] = clblasDaxpy(1, 1 / ifo[0].sig.sig2, aadots_d[0], 0, 1, maa_d, 0, 1, 1, &cl_handles->exec_queues[id][0], (nifo * 2 + 2), input_wait_events, &axpy_events[0]); checkErrBLAS(status[0], "clblasDaxpy()");
    status[1] = clblasDaxpy(1, 1 / ifo[0].sig.sig2, bbdots_d[0], 0, 1, mbb_d, 0, 1, 1, &cl_handles->exec_queues[id][0], (nifo * 2 + 2), input_wait_events, &axpy_events[1]); checkErrBLAS(status[1], "clblasDaxpy()");
#endif // COMP_FLOAT

	for (i = 1; i < nifo; ++i)
	{
#ifdef COMP_FLOAT
		status[0] = clblasSaxpy(1, 1 / ifo[i].sig.sig2, aadots_d[i], 0, 1, maa_d, 0, 1, 1, &cl_handles->exec_queues[id][0], 2, &axpy_events[(i - 1) * 2 + 0], &axpy_events[i * 2 + 0]); checkErrBLAS(status[0], "clblasDaxpy()");
		status[1] = clblasSaxpy(1, 1 / ifo[i].sig.sig2, bbdots_d[i], 0, 1, mbb_d, 0, 1, 1, &cl_handles->exec_queues[id][0], 2, &axpy_events[(i - 1) * 2 + 1], &axpy_events[i * 2 + 1]); checkErrBLAS(status[1], "clblasDaxpy()");
#else
		status[0] = clblasDaxpy(1, 1 / ifo[i].sig.sig2, aadots_d[i], 0, 1, maa_d, 0, 1, 1, &cl_handles->exec_queues[id][0], 2, &axpy_events[(i - 1) * 2 + 0], &axpy_events[i * 2 + 0]); checkErrBLAS(status[0], "clblasDaxpy()");
		status[1] = clblasDaxpy(1, 1 / ifo[i].sig.sig2, bbdots_d[i], 0, 1, mbb_d, 0, 1, 1, &cl_handles->exec_queues[id][0], 2, &axpy_events[(i - 1) * 2 + 1], &axpy_events[i * 2 + 1]); checkErrBLAS(status[1], "clblasDaxpy()");
#endif // COMP_FLOAT
	}

    free(input_wait_events);
}

void spindown_range(const int mm,                  // grid 'sky position'
                    const int nn,                  // other grid 'sky position'
                    const real_t Smin,
                    const real_t Smax,
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

cl_event phase_mod_1_gpu(const cl_int idet,
                         const cl_int id,
                         const cl_int N,
                         const real_t het1,
                         const real_t sgnlt1,
                         const cl_mem xar,
                         const cl_mem xbr,
                         const cl_mem shft,
                         cl_mem xa,
                         cl_mem xb,
                         OpenCL_handles* cl_handles,
                         const cl_uint num_events_in_wait_list,
                         const cl_event* event_wait_list)
{
  cl_int CL_err = CL_SUCCESS;

  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 0, sizeof(cl_mem), &xa);	  checkErr(CL_err, "clSetKernelArg(&xa)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 1, sizeof(cl_mem), &xb);	  checkErr(CL_err, "clSetKernelArg(&xb)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 2, sizeof(cl_mem), &xar);	  checkErr(CL_err, "clSetKernelArg(&xar)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 3, sizeof(cl_mem), &xbr);	  checkErr(CL_err, "clSetKernelArg(&xbr)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 4, sizeof(real_t), &het1);	  checkErr(CL_err, "clSetKernelArg(&het1)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 5, sizeof(real_t), &sgnlt1); checkErr(CL_err, "clSetKernelArg(&sgnlt1)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 6, sizeof(cl_mem), &shft);	  checkErr(CL_err, "clSetKernelArg(&shft)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 7, sizeof(cl_int), &N);      checkErr(CL_err, "clSetKernelArg(&N)");

  cl_event exec;
  size_t size_N = (size_t)N; // Helper variable to make pointer types match. Cast to silence warning

  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][PhaseMod1], 1, NULL, &size_N, NULL, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(PhaseMod1)");

#ifdef TESTING
  CL_err = clWaitForEvents(1, &exec); checkErr(CL_err, "clWaitForEvents(PhaseMod1)");
  
  save_numbered_real_buffer(cl_handles->read_queues[id][idet], ifo[0].sig.shft_d, N, idet, "pre_fft_phasemod_ifo_sig_shft");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], ifo[0].sig.xDatma_d, N, idet, "pre_fft_phasemod_ifo_sig_xDatma");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], ifo[0].sig.xDatmb_d, N, idet, "pre_fft_phasemod_ifo_sig_xDatmb");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], fft_arr->xa_d, N, idet, "pre_fft_phasemod_xa");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], fft_arr->xb_d, N, idet, "pre_fft_phasemod_xb");
#endif

  return exec;
}

cl_event phase_mod_2_gpu(const cl_int idet,
                         const cl_int id,
                         const cl_int N,
                         const real_t het1,
                         const real_t sgnlt1,
                         const cl_mem xar,
                         const cl_mem xbr,
                         const cl_mem shft,
                         cl_mem xa,
                         cl_mem xb,
                         OpenCL_handles* cl_handles,
                         const cl_uint num_events_in_wait_list,
                         const cl_event* event_wait_list)
{
  cl_int CL_err = CL_SUCCESS;

  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 0, sizeof(cl_mem), &xa);	  checkErr(CL_err, "clSetKernelArg(&xa)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 1, sizeof(cl_mem), &xb);	  checkErr(CL_err, "clSetKernelArg(&xb)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 2, sizeof(cl_mem), &xar);	  checkErr(CL_err, "clSetKernelArg(&xar)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 3, sizeof(cl_mem), &xbr);	  checkErr(CL_err, "clSetKernelArg(&xbr)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 4, sizeof(real_t), &het1);	  checkErr(CL_err, "clSetKernelArg(&het1)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 5, sizeof(real_t), &sgnlt1); checkErr(CL_err, "clSetKernelArg(&sgnlt1)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 6, sizeof(cl_mem), &shft);	  checkErr(CL_err, "clSetKernelArg(&shft)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 7, sizeof(cl_int), &N);      checkErr(CL_err, "clSetKernelArg(&N)");

  cl_event exec;
  size_t size_N = (size_t)N; // Helper variable to make pointer types match. Cast to silence warning

  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][PhaseMod2], 1, NULL, &size_N, NULL, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(PhaseMod1)");

#ifdef TESTING
  CL_err = clWaitForEvents(1, &exec); checkErr(CL_err, "clWaitForEvents(PhaseMod1)");

  save_numbered_real_buffer(cl_handles->read_queues[id][idet], ifo[0].sig.shft_d, N, idet, "pre_fft_phasemod_ifo_sig_shft");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], ifo[0].sig.xDatma_d, N, idet, "pre_fft_phasemod_ifo_sig_xDatma");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], ifo[0].sig.xDatmb_d, N, idet, "pre_fft_phasemod_ifo_sig_xDatmb");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], fft_arr->xa_d, N, idet, "pre_fft_phasemod_xa");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], fft_arr->xb_d, N, idet, "pre_fft_phasemod_xb");
#endif

  return exec;
}

void zero_pad(const cl_int idet,
              const cl_int id,
              const Search_settings *sett,
              cl_mem xa_d,
              cl_mem xb_d,
              OpenCL_handles* cl_handles,
              const cl_uint num_events_in_wait_list,
              const cl_event* event_wait_list,
              cl_event* zero_pad_events)
{
  cl_int CL_err = CL_SUCCESS;

#ifdef _WIN32
  complex_t pattern = { 0, 0 };
#else
  complex_t pattern = 0;
#endif

  // Zero pad from offset until the end
  CL_err = clEnqueueFillBuffer(cl_handles->write_queues[id][idet], xa_d, &pattern, sizeof(complex_t), sett->N * sizeof(complex_t), (sett->fftpad*sett->nfft - sett->N) * sizeof(complex_t), 0, NULL, &zero_pad_events[0]);
  checkErr(CL_err, "clEnqueueFillBuffer");
  CL_err = clEnqueueFillBuffer(cl_handles->write_queues[id][idet], xb_d, &pattern, sizeof(complex_t), sett->N * sizeof(complex_t), (sett->fftpad*sett->nfft - sett->N) * sizeof(complex_t), 0, NULL, &zero_pad_events[1]);
  checkErr(CL_err, "clEnqueueFillBuffer");

#ifdef TESTING
  clWaitForEvents(2, zero_pad_events);
  // Wasteful
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, sett->Ninterp, 0, "pre_fft_post_zero_xa");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, sett->Ninterp, 0, "pre_fft_post_zero_xb");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, sett->Ninterp, 1, "pre_fft_post_zero_xa");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, sett->Ninterp, 1, "pre_fft_post_zero_xb");
#endif
}

void time_to_frequency(const cl_int idet,
                       const cl_int id,
                       const Search_settings *sett,
                       const FFT_plans* plans,
                       cl_mem xa_d,
                       cl_mem xb_d,
                       OpenCL_handles* cl_handles,
                       const cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* fw2_fft_events)
{
  clfftStatus CLFFT_status = CLFFT_SUCCESS;

  clfftEnqueueTransform(plans->plan, CLFFT_FORWARD, 1, &cl_handles->exec_queues[id][idet], 0, NULL, &fw2_fft_events[0], &xa_d, NULL, NULL);
  checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");
  clfftEnqueueTransform(plans->plan, CLFFT_FORWARD, 1, &cl_handles->exec_queues[id][idet], 0, NULL, &fw2_fft_events[1], &xb_d, NULL, NULL);
  checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");

#ifdef TESTING
  clWaitForEvents(2, fw2_fft_events);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, sett->nfftf, 0, "post_fft_phasemod_xa");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, sett->nfftf, 0, "post_fft_phasemod_xb");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, sett->nfftf, 1, "post_fft_phasemod_xa");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, sett->nfftf, 1, "post_fft_phasemod_xb");
#endif
}

cl_event compute_Fstat_gpu(const cl_int idet,
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
                           const cl_event* event_wait_list)
{
  cl_int CL_err = CL_SUCCESS;
  cl_int N = nmax - nmin;

  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 0, sizeof(cl_mem), &xa_d);
  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 1, sizeof(cl_mem), &xb_d);
  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 2, sizeof(cl_mem), &F_d);
  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 3, sizeof(cl_mem), &maa_d);
  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 4, sizeof(cl_mem), &mbb_d);
  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 5, sizeof(cl_int), &N);

  cl_event exec;
  size_t size_N = (size_t)N,
         size_nmin = (size_t)nmin; // Helper variable to make pointer types match. Cast to silence warning

  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][ComputeFStat], 1, &size_nmin, &size_N, NULL, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(ComputeFStat)");

#ifdef TESTING
  // Wasteful
  clWaitForEvents(1, &exec);
  save_numbered_real_buffer(cl_handles->read_queues[id][idet], F_d, sett->nmax - sett->nmin, 0, "Fstat");
  save_numbered_real_buffer(cl_handles->read_queues[id][idet], F_d, sett->nmax - sett->nmin, 1, "Fstat");
#endif

	return exec;
}

cl_event normalize_FStat_gpu_wg_reduce(const cl_int idet,
                                       const cl_int id,
                                       const cl_int nmin,
                                       const cl_int nmax,
                                       const cl_uint nav,
                                       cl_mem F_d,
                                       OpenCL_handles* cl_handles,
                                       const cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list)
{
  cl_int CL_err = CL_SUCCESS;
  size_t max_wgs;             // maximum supported wgs on the device (limited by register count)
  cl_ulong local_size;        // local memory size in bytes

  CL_err = clGetKernelWorkGroupInfo(cl_handles->kernels[id][NormalizeFStatWG],
                                    cl_handles->devs[id],
                                    CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(size_t),
                                    &max_wgs,
                                    NULL);
  checkErr(CL_err, "clGetKernelWorkGroupInfo(FStatSimple, CL_KERNEL_WORK_GROUP_SIZE)");

  CL_err = clGetDeviceInfo(cl_handles->devs[id],
                           CL_DEVICE_LOCAL_MEM_SIZE,
                           sizeof(cl_ulong),
                           &local_size,
                           NULL);
  checkErr(CL_err, "clGetDeviceInfo(FStatSimple, CL_DEVICE_LOCAL_MEM_SIZE)");

  // How long is the array of local memory (shared size in num gentypes)
  cl_uint ssi = (cl_uint)(local_size / sizeof(real_t)); // Assume integer is enough to store gentype count (well, it better)

  clSetKernelArg(cl_handles->kernels[id][NormalizeFStatWG], 0, sizeof(cl_mem), &F_d);
  clSetKernelArg(cl_handles->kernels[id][NormalizeFStatWG], 1, ssi * sizeof(real_t), NULL);
  clSetKernelArg(cl_handles->kernels[id][NormalizeFStatWG], 2, sizeof(cl_uint), &ssi);
  clSetKernelArg(cl_handles->kernels[id][NormalizeFStatWG], 3, sizeof(cl_uint), &nav);

  size_t gsi = ((nmax - nmin)/nav)*nav; // truncate nmax-nmin to the nearest multiple of nav
  size_t wgs = gsi < max_wgs ? gsi : max_wgs; // std::min
  size_t off = nmin;

  // Check preconditions of kernel before launch
  assert(ssi <= nav);
  assert(wgs <= ssi);
  assert(((wgs != 0) && !(wgs & (wgs - 1)))); // Method 9. @ https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
  assert(!(gsi % nav));

  cl_event exec;
  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][NormalizeFStatWG], 1, &off, &gsi, &wgs, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(FStatSimple)");

  return exec;
}

void find_peaks(const cl_int idet,
                const cl_int id,
                const cl_int nmin,
                const cl_int nmax,
                const real_t trl,
                const real_t sgnl0,
                const Search_settings *sett,
                const cl_mem F_d,
                Search_results* results,
                real_t* sgnlt,
                OpenCL_handles* cl_handles,
                const cl_uint num_events_in_wait_list,
                const cl_event* event_wait_list,
	            cl_event* peak_map_event,
                cl_event* peak_unmap_event)
{
  cl_int CL_err = CL_SUCCESS;

  real_t* F = clEnqueueMapBuffer(cl_handles->read_queues[id][0],
                                 F_d,
                                 CL_TRUE,
                                 CL_MAP_READ,
                                 0,
                                 (nmax - nmin) * sizeof(real_t),
                                 num_events_in_wait_list,
                                 event_wait_list,
                                 peak_map_event,
                                 &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(F_d)");

  for (int i = nmin; i < nmax; i++)
  {
    real_t Fc = F[i];

    if (Fc > trl) // if F-stat exceeds trl (critical value)
    {             // Find local maximum for neighboring signals
      int ii = i;

      while (++i < nmax && F[i] > trl)
      {
        if (F[i] >= Fc)
        {
          ii = i;
          Fc = F[i];
        } // if F[i] 
      } // while i 

      // Candidate signal frequency
      sgnlt[0] = 2.*M_PI*ii / ((real_t)sett->fftpad*sett->nfft) + sgnl0;
      // Signal-to-noise ratio
      sgnlt[4] = sqrt(2.*(Fc - sett->nd));

      // Add new parameters to output array
	  results->sgnlc++; // increase found number
      results->sgnlv = (real_t *)realloc(results->sgnlv, NPAR*(results->sgnlc) * sizeof(real_t));

      for (int j = 0; j < NPAR; ++j) // save new parameters
        results->sgnlv[NPAR*(results->sgnlc - 1) + j] = (real_t)sgnlt[j];

#ifdef VERBOSE
			printf("\nSignal %d: %d %d %d %d %d \tsnr=%.2f\n",
				*sgnlc, pm, mm, nn, ss, ii, sgnlt[4]);
#endif 
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
                            sizeof(FLOAT_TYPE),
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
        Search_results* select = &results[mm - s_range->mr[0]]
                                         [nn - s_range->nr[0]];

        // Add new parameters to output array
        size_t old_sgnlc = result.sgnlc;
        result.sgnlc += select->sgnlc;

        result.sgnlv = (real_t*)realloc(result.sgnlv,
                                         (result.sgnlc) * NPAR * sizeof(real_t));
        memcpy(result.sgnlv + (old_sgnlc * NPAR),
               select->sgnlv,
               select->sgnlc * NPAR * sizeof(real_t));
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

#ifdef GPUFSTAT
/* WARNING
   This won't necessarily work for other values than:
   NAV = 4096
   N = nmax - nmin = 507904
   For those given above it works fine.
   Reason is the "reduction depth", i.e. number of required \
   reduction kernel calls.
*/
void FStat_gpu(FLOAT_TYPE *F_d, int N, int nav, FLOAT_TYPE *mu_d, FLOAT_TYPE *mu_t_d) {

  int nav_blocks = N/nav;           //number of blocks
  int nav_threads = nav/BLOCK_SIZE; //number of blocks computing one nav-block
  int blocks = N/BLOCK_SIZE;

  //    CudaSafeCall ( cudaMalloc((void**)&cu_mu_t, sizeof(float)*blocks) );
  //    CudaSafeCall ( cudaMalloc((void**)&cu_mu, sizeof(float)*nav_blocks) );

  //sum fstat in blocks
  reduction_sum<BLOCK_SIZE_RED><<<blocks, BLOCK_SIZE_RED, BLOCK_SIZE_RED*sizeof(FLOAT_TYPE)>>>(F_d, mu_t_d, N);
  CudaCheckError();

  //sum blocks computed above and return 1/mu (number of divisions: blocks), then fstat_norm doesn't divide (potential number of divisions: N)
  reduction_sum<<<nav_blocks, nav_threads, nav_threads*sizeof(FLOAT_TYPE)>>>(mu_t_d, mu_d, blocks);
  CudaCheckError();
  
  //divide by mu/(2*NAV)
  fstat_norm<<<blocks, BLOCK_SIZE>>>(F_d, mu_d, N, nav);
  CudaCheckError();
  
}
#endif
