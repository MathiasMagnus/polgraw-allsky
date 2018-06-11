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
            int* Fnum,
            cl_mem F_d)
{
    cl_int CL_err = CL_SUCCESS;
    int pm, mm, nn;    // hemisphere, sky positions 
    int sgnlc;         // number of candidates
    FLOAT_TYPE *sgnlv;    // array with candidates data

    char outname[512];
#ifdef _WIN32
    int low_state;
#endif // WIN32
    FILE *state;

#if TIMERS>0
    struct timespec tstart = get_current_time(), tend;
#endif

    // Copy amod coefficients to device
    copy_amod_coeff(ifo, sett->nifo, cl_handles, aux);

    int cand_buffer_count = 0;

    //allocate vector for FStat_gpu
    int nav_blocks = (sett->nmax - sett->nmin) / NAV;     //number of nav-blocks
    int blocks = (sett->nmax - sett->nmin) / BLOCK_SIZE;  //number of blocks for Fstat-smoothing

    aux->mu_t_d = clCreateBuffer(cl_handles->ctx, CL_MEM_READ_WRITE, blocks * sizeof(real_t), NULL, &CL_err);
    aux->mu_d = clCreateBuffer(cl_handles->ctx, CL_MEM_READ_WRITE, nav_blocks * sizeof(real_t), NULL, &CL_err);

    state = NULL;
    if (opts->checkp_flag)
#ifdef _WIN32
    {
        _sopen_s(&low_state, opts->qname,
            _O_RDWR | _O_CREAT,   // Allowed operations
            _SH_DENYNO,           // Allowed sharing
            _S_IREAD | _S_IWRITE);// Permission settings

        state = _fdopen(low_state, "w");
    }
#else
        state = fopen(opts->qname, "w");
#endif // WIN32

    // Loop over hemispheres //
    for (pm = s_range->pst; pm <= s_range->pmr[1]; ++pm)
    {
        sprintf(outname, "%s/triggers_%03d_%03d%s_%d.bin",
                opts->prefix,
                opts->ident,
                opts->band,
                opts->label,
                pm);

        // Two main loops over sky positions //
        for (mm = s_range->mst; mm <= s_range->mr[1]; ++mm)
        {
            for (nn = s_range->nst; nn <= s_range->nr[1]; ++nn)
            {
                if (opts->checkp_flag)
                {
#ifdef _WIN32
                    if (_chsize(low_state, 0))
                    {
                        printf("Failed to resize file");
                        exit(EXIT_FAILURE);
                    }
#else
                    if (ftruncate(fileno(state), 0))
                    {
                        printf("Failed to resize file");
                        exit(EXIT_FAILURE);
                    }
#endif // WIN32
                    fprintf(state, "%d %d %d %d %d\n", pm, mm, nn, s_range->sst, *Fnum);
                    fseek(state, 0, SEEK_SET);
                }

                // Loop over spindowns is inside job_core() //
                sgnlv = job_core(pm,            // hemisphere
                                 mm,            // grid 'sky position'
                                 nn,            // other grid 'sky position'
                                 ifo,           // detector settings
                                 sett,          // search settings
                                 opts,          // cmd opts
                                 s_range,       // range for searching
                                 plans,         // fftw plans 
                                 fft_arr,       // arrays for fftw
                                 aux,           // auxiliary arrays
                                 F_d,           // F-statistics array
                                 &sgnlc,        // reference to array with the parameters
                                                // of the candidate signal
                                                // (used below to write to the file)
                                 Fnum,          // Candidate signal number
                                 cl_handles,    // handles to OpenCL resources
                                 blas_handles   // handle for scaling
                );

                // Get back to regular spin-down range
                s_range->sst = s_range->spndr[0];

                // Add trigger parameters to a file //

                // if any signals found (Fstat>Fc)
                if (sgnlc)
                {
                    FILE* fc = fopen(outname, "w");
                    if (fc == NULL) perror("Failed to open output file.");

                    size_t count = fwrite((void *)(sgnlv), sizeof(FLOAT_TYPE), sgnlc*NPAR, fc);
                    if (count < sgnlc*NPAR) perror("Failed to write output file.");

                    int close = fclose(fc);
                    if (close == EOF) perror("Failed to close output file.");

                } // if sgnlc
                sgnlc = 0;
            } // for nn
            s_range->nst = s_range->nr[0];
        } // for mm
        s_range->mst = s_range->mr[0];
    } // for pm

    if (opts->checkp_flag)
        fclose(state);

    // cublasDestroy(scale);

#if TIMERS>0
    tend = get_current_time();
    // printf("tstart = %d . %d\ntend = %d . %d\n", tstart.tv_sec, tstart.tv_usec, tend.tv_sec, tend.tv_usec);
    double time_elapsed = get_time_difference(tstart, tend);
    printf("Time elapsed: %e s\n", time_elapsed);
#endif

} // end of search

real_t* job_core(const int pm,                  // hemisphere
                 const int mm,                  // grid 'sky position'
                 const int nn,                  // other grid 'sky position'
                 Detector_settings* ifo,        // detector settings
                 Search_settings *sett,         // search settings
                 Command_line_opts *opts,       // cmd opts
                 Search_range *s_range,         // range for searching
                 FFT_plans *plans,              // plans for fftw
                 FFT_arrays *fft_arr,           // arrays for fftw
                 Aux_arrays *aux,               // auxiliary arrays
                 cl_mem F_d,                    // F-statistics array
                 int *sgnlc,                    // reference to array with the parameters of the candidate signal
                                                // (used below to write to the file)
                 int *FNum,                     // candidate signal number
                 OpenCL_handles* cl_handles,    // handles to OpenCL resources
                 BLAS_handles* blas_handles)    // handle for scaling
{
  // return values
  real_t* sgnlv = NULL;
  *sgnlc = 0; // Redundant, already zeroed out externally.
              // Will be removed once return value is struct.

  // Allocate storage for events to synchronize pipeline
  cl_event *modvir_events = (cl_event*)malloc(sett->nifo * sizeof(cl_event)),
	       *tshift_pmod_events = (cl_event*)malloc(sett->nifo * sizeof(cl_event)),
	       **fw_fft_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*)),
	       *resample_postfft_events = (cl_event*)malloc(sett->nifo * sizeof(cl_event)),
	       **inv_fft_events = (cl_event**)malloc(sett->nifo * sizeof(cl_event*));
  for (int n = 0; n < sett->nifo; ++n)
  {
	fw_fft_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
	inv_fft_events[n] = (cl_event*)malloc(2 * sizeof(cl_event));
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
    modvir_events[n] = modvir_gpu(n, sett->N,                                      // input
                                  sinalt, cosalt, sindelt, cosdelt,                // input
                                  ifo[n].sig.cphir, ifo[n].sig.sphir,              // input
                                  aux->ifo_amod_d, aux->sinmodf_d, aux->cosmodf_d, // input
                                  ifo[n].sig.aa_d, ifo[n].sig.bb_d,                // output
                                  cl_handles, 0, NULL);                            // sync

    // Calculate detector positions with respect to baricenter
    real3_t nSource = { cosalt * cosdelt,
                        sinalt * cosdelt,
                        sindelt };
    real_t shft1 = nSource.s[0] * ifo[n].sig.DetSSB[0].s[0] +
                   nSource.s[1] * ifo[n].sig.DetSSB[0].s[1] +
                   nSource.s[2] * ifo[n].sig.DetSSB[0].s[2];

    tshift_pmod_events[n] = tshift_pmod_gpu(shft1, het0, nSource,                                                               // input
                                            sett->oms, sett->N, sett->nfft, sett->interpftpad,                                  // input
                                            ifo[n].sig.xDat_d, ifo[n].sig.aa_d, ifo[n].sig.bb_d, ifo[n].sig.DetSSB_d,           // input
                                            fft_arr->xa_d, fft_arr->xb_d, ifo[n].sig.shft_d, ifo[n].sig.shftf_d, aux->tshift_d, // output
                                            cl_handles, 1, &modvir_events[n]);                                                  // sync
#ifdef TESTING
    save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xa_d, 2 * sett->nfft, n, "xa_time");
    save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xb_d, 2 * sett->nfft, n, "xb_time");
    save_numbered_real_buffer(cl_handles->exec_queues[0], ifo[n].sig.shft_d, sett->N, n, "ifo_sig_shft");
    save_numbered_real_buffer(cl_handles->exec_queues[0], ifo[n].sig.shftf_d, sett->N, n, "ifo_sig_shftf");
#endif		
	fft_interpolate_gpu(n, sett->nfft, sett->Ninterp, nyqst, plans,				 // input
		                fft_arr->xa_d, fft_arr->xb_d,							 // input / output
		                cl_handles, 1, &tshift_pmod_events[n],					 // sync
		                fw_fft_events, resample_postfft_events, inv_fft_events); // sync

	clWaitForEvents(2, inv_fft_events[n]);

        //scale fft with cublas (not needed, clFFT already scales)
        //ft = (double)sett->interpftpad / sett->Ninterp;
        //blas_scale(fft_arr->xa_d,
        //           fft_arr->xb_d,
        //           sett->Ninterp,
        //           ft,
        //           cl_handles,
        //           blas_handles);

        // Spline interpolation to xDatma, xDatmb arrays
        //gpu_interp(fft_arr->xa_d,       //input data
        //           sett->Ninterp,       //input data length
        //           aux->tshift_d,       //output time domain
        //           ifo[n].sig.xDatma_d, //output values
        //           sett->N,             //output data length
        //           aux->diag_d,         //diagonal
        //           aux->ldiag_d,        //lower diagonal
        //           aux->udiag_d,        //upper diagonal
        //           aux->B_d,            //coefficient matrix
        //           cl_handles);
        //
        //gpu_interp(fft_arr->xb_d,       //input data
        //           sett->Ninterp,       //input data length
        //           aux->tshift_d,       //output time domain
        //           ifo[n].sig.xDatmb_d, //output values
        //           sett->N,             //output data length
        //           aux->diag_d,         //diagonal
        //           aux->ldiag_d,        //lower diagonal
        //           aux->udiag_d,        //upper diagonal
        //           aux->B_d,            //coefficient matrix
        //           cl_handles);

        // Spline interpolation to xDatma, xDatmb arrays
        {
            cl_int CL_err;
            void *xa_d, *xb_d, *shftf, *xDatma, *xDatmb;

            xa_d = clEnqueueMapBuffer(cl_handles->exec_queues[0],
                                      fft_arr->xa_d,
                                      CL_TRUE,
                                      CL_MAP_READ,
                                      0,
                                      fft_arr->arr_len * sizeof(complex_t),
                                      0,
                                      NULL,
                                      NULL,
                                      &CL_err);
            checkErr(CL_err, "clEnqueueMapBuffer(fft_arr->xa_d)");

            xb_d = clEnqueueMapBuffer(cl_handles->exec_queues[0],
                                      fft_arr->xb_d,
                                      CL_TRUE,
                                      CL_MAP_READ,
                                      0,
                                      fft_arr->arr_len * sizeof(complex_t),
                                      0,
                                      NULL,
                                      NULL,
                                      &CL_err);
            checkErr(CL_err, "clEnqueueMapBuffer(fft_arr->xb_d)");

            shftf = clEnqueueMapBuffer(cl_handles->exec_queues[0],
                                       ifo[n].sig.shftf_d,
                                       CL_TRUE,
                                       CL_MAP_READ,
                                       0,
                                       sett->N * sizeof(real_t),
                                       0,
                                       NULL,
                                       NULL,
                                       &CL_err);
            checkErr(CL_err, "clEnqueueMapBuffer(ifo[n].sig.shftf_d)");

            xDatma = clEnqueueMapBuffer(cl_handles->exec_queues[0],
                                        ifo[n].sig.xDatma_d,
                                        CL_TRUE,
                                        CL_MAP_WRITE,
                                        0,
                                        sett->N * sizeof(complex_devt),
                                        0,
                                        NULL,
                                        NULL,
                                        &CL_err);
            checkErr(CL_err, "clEnqueueMapBuffer(ifo[n].sig.xDatma_d)");

            xDatmb = clEnqueueMapBuffer(cl_handles->exec_queues[0],
                                        ifo[n].sig.xDatmb_d,
                                        CL_TRUE,
                                        CL_MAP_WRITE,
                                        0,
                                        sett->N * sizeof(complex_devt),
                                        0,
                                        NULL,
                                        NULL,
                                        &CL_err);
            checkErr(CL_err, "clEnqueueMapBuffer(ifo[n].sig.xDatmb_d)");

            splintpad(xa_d, shftf, sett->N, sett->interpftpad, xDatma);
            splintpad(xb_d, shftf, sett->N, sett->interpftpad, xDatmb);

            cl_event unmaps[5];
            CL_err = clEnqueueUnmapMemObject(cl_handles->exec_queues[0], fft_arr->xa_d, xa_d, 0, NULL, &unmaps[0]); checkErr(CL_err, "clEnqueueUnMapMemObject(fft_arr->xa_d)");
            CL_err = clEnqueueUnmapMemObject(cl_handles->exec_queues[0], fft_arr->xb_d, xb_d, 0, NULL, &unmaps[1]); checkErr(CL_err, "clEnqueueUnMapMemObject(fft_arr->xb_d)");
            CL_err = clEnqueueUnmapMemObject(cl_handles->exec_queues[0], ifo[n].sig.shftf_d, shftf, 0, NULL, &unmaps[2]); checkErr(CL_err, "clEnqueueUnMapMemObject(ifo[n].sig.shftf_d)");
            CL_err = clEnqueueUnmapMemObject(cl_handles->exec_queues[0], ifo[n].sig.xDatma_d, xDatma, 0, NULL, &unmaps[3]); checkErr(CL_err, "clEnqueueUnMapMemObject(ifo[n].sig.xDatma_d, xDatma)");
            CL_err = clEnqueueUnmapMemObject(cl_handles->exec_queues[0], ifo[n].sig.xDatmb_d, xDatmb, 0, NULL, &unmaps[4]); checkErr(CL_err, "clEnqueueUnMapMemObject(ifo[n].sig.xDatma_d, xDatmb)");
            CL_err = clWaitForEvents(5, unmaps);
            checkErr(CL_err, "clWaitForEvents(5, unmaps)");

            int j;
            for (j = 0; j < 5; ++j)
                clReleaseEvent(unmaps[j]);
        }
#ifdef TESTING
        save_numbered_complex_buffer(cl_handles->exec_queues[0], ifo[n].sig.xDatma_d, sett->N, n, "ifo_sig_xDatma");
        save_numbered_complex_buffer(cl_handles->exec_queues[0], ifo[n].sig.xDatmb_d, sett->N, n, "ifo_sig_xDatmb");
#endif
        ft = 1. / ifo[n].sig.sig2;

        blas_scale(ifo[n].sig.xDatma_d,
                   ifo[n].sig.xDatmb_d,
                   sett->N,
                   ft,
                   cl_handles,
                   blas_handles);
#ifdef TESTING
        save_numbered_complex_buffer(cl_handles->exec_queues[0], ifo[n].sig.xDatma_d, sett->N, n, "rescaled_ifo_sig_xDatma");
        save_numbered_complex_buffer(cl_handles->exec_queues[0], ifo[n].sig.xDatmb_d, sett->N, n, "rescaled_ifo_sig_xDatmb");
#endif
    } // end of detector loop 

    real_t _maa = 0;
    real_t _mbb = 0;

    for (int n = 0; n<sett->nifo; ++n)
    {
        real_t* temp = blas_dot(ifo[n].sig.aa_d, ifo[n].sig.bb_d, sett->N, cl_handles, blas_handles);
        real_t aatemp = temp[0],
               bbtemp = temp[1];

        _maa += temp[0] / ifo[n].sig.sig2;
        _mbb += temp[1] / ifo[n].sig.sig2;

        free(temp);
    }

    // Copy sums to constant memory
    {
        cl_event write_event[2];
        clEnqueueWriteBuffer(cl_handles->write_queues[0], aux->maa_d, CL_FALSE, 0, sizeof(real_t), &_maa, 0, NULL, &write_event[0]);
        clEnqueueWriteBuffer(cl_handles->write_queues[0], aux->mbb_d, CL_FALSE, 0, sizeof(real_t), &_mbb, 0, NULL, &write_event[1]);

        clWaitForEvents(2, write_event);
        for (size_t i = 0; i < 2; ++i) clReleaseEvent(write_event[i]);
    }

    // Spindown loop //

#if TIMERS>2
    struct timespec tstart, tend;
    double spindown_timer = 0;
    int spindown_counter = 0;
#endif

    // Check if the signal is added to the data 
    // or the range file is given:  
    // if not, proceed with the wide range of spindowns 
    // if yes, use smin = s_range->sst, smax = s_range->spndr[1]
    int smin = s_range->sst, smax = s_range->spndr[1];
    if (!strcmp(opts->addsig, "") && !strcmp(opts->range, "")) {

        // Spindown range defined using Smin and Smax (settings.c)  
        smin = (int)trunc((sett->Smin - nn*sett->M[9] - mm*sett->M[13]) / sett->M[5]);  // Cast is intentional and safe (silences warning).
        smax = (int)trunc(-(nn*sett->M[9] + mm*sett->M[13] + sett->Smax) / sett->M[5]); // Cast is intentional and safe (silences warning).
    }

    printf("\n>>%d\t%d\t%d\t[%d..%d]\n", *FNum, mm, nn, smin, smax);

    // No-spindown calculations
    if (opts->s0_flag) smin = smax;

    // if spindown parameter is taken into account, smin != smax
    int ss;
    for (ss = smin; ss <= smax; ++ss)
    {

#if TIMERS>2
        tstart = get_current_time();
#endif 

        // Spindown parameter
        sgnlt[1] = ss*sett->M[5] + nn*sett->M[9] + mm*sett->M[13];

        //    // Spindown range
        //    if(sgnlt[1] >= -sett->Smax && sgnlt[1] <= sett->Smax) { 

        int ii;
        real_t Fc, het1;

#ifdef VERBOSE
        //print a 'dot' every new spindown
        printf("."); fflush(stdout);
#endif 

        het1 = fmod(ss*sett->M[4], sett->M[0]);
        if (het1<0) het1 += sett->M[0];

        sgnl0 = het0 + het1;
        // printf("%d  %d\n", BLOCK_SIZE, (sett->N + BLOCK_SIZE - 1)/BLOCK_SIZE );

        phase_mod_1_gpu(fft_arr->xa_d,
                        fft_arr->xb_d,
                        ifo[0].sig.xDatma_d,
                        ifo[0].sig.xDatmb_d,
                        het1,
                        sgnlt[1],
                        ifo[0].sig.shft_d,
                        sett->N,
                        cl_handles);
#ifdef TESTING
        save_numbered_real_buffer(cl_handles->exec_queues[0], ifo[0].sig.shft_d, sett->N, 0, "pre_fft_phasemod_ifo_sig_shft");
        save_numbered_complex_buffer(cl_handles->exec_queues[0], ifo[0].sig.xDatma_d, sett->N, 0, "pre_fft_phasemod_ifo_sig_xDatma");
        save_numbered_complex_buffer(cl_handles->exec_queues[0], ifo[0].sig.xDatmb_d, sett->N, 0, "pre_fft_phasemod_ifo_sig_xDatmb");
        save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xa_d, sett->N, 0, "pre_fft_phasemod_xa");
        save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xb_d, sett->N, 0, "pre_fft_phasemod_xb");
#endif

        for (int n = 1; n<sett->nifo; ++n)
        {
            phase_mod_2_gpu(fft_arr->xa_d,
                            fft_arr->xb_d,
                            ifo[n].sig.xDatma_d,
                            ifo[n].sig.xDatmb_d,
                            het1,
                            sgnlt[1],
                            ifo[n].sig.shft_d,
                            sett->N,
                            cl_handles);
#ifdef TESTING
            save_numbered_real_buffer(cl_handles->exec_queues[0], ifo[n].sig.shft_d, sett->N, n, "pre_fft_phasemod_ifo_sig_shft");
            save_numbered_complex_buffer(cl_handles->exec_queues[0], ifo[n].sig.xDatma_d, sett->N, n, "pre_fft_phasemod_ifo_sig_xDatma");
            save_numbered_complex_buffer(cl_handles->exec_queues[0], ifo[n].sig.xDatmb_d, sett->N, n, "pre_fft_phasemod_ifo_sig_xDatmb");
            save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xa_d, sett->N, n, "pre_fft_phasemod_xa");
            save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xb_d, sett->N, n, "pre_fft_phasemod_xb");
#endif
        }

        // initialize arrays to 0. with integer 0
        // assuming double , remember to change when switching to float
        {
            cl_int CL_err = CL_SUCCESS;
#ifdef _WIN32
            complex_t pattern = { 0, 0 };
#else
            complex_t pattern = 0;
#endif

            cl_event fill_event[2];

            // Zero pad from offset until the end
            //CL_err = clEnqueueFillBuffer(cl_handles->write_queues[0], fft_arr->xa_d, &pattern, sizeof(complex_t), sett->N * sizeof(complex_t), (sett->nfftf - sett->N) * 2 * sizeof(complex_t), 0, NULL, &fill_event[0]);
            CL_err = clEnqueueFillBuffer(cl_handles->write_queues[0], fft_arr->xa_d, &pattern, sizeof(complex_t), sett->N * sizeof(complex_t), (sett->fftpad*sett->nfft - sett->N) * sizeof(complex_t), 0, NULL, &fill_event[0]);
            checkErr(CL_err, "clEnqueueFillBuffer");
            CL_err = clEnqueueFillBuffer(cl_handles->write_queues[0], fft_arr->xb_d, &pattern, sizeof(complex_t), sett->N * sizeof(complex_t), (sett->fftpad*sett->nfft - sett->N) * sizeof(complex_t), 0, NULL, &fill_event[1]);
            checkErr(CL_err, "clEnqueueFillBuffer");

            clWaitForEvents(2, fill_event);

            clReleaseEvent(fill_event[0]);
            clReleaseEvent(fill_event[1]);
#ifdef TESTING
            // Wasteful
            save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xa_d, sett->Ninterp, 0, "pre_fft_post_zero_xa");
            save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xb_d, sett->Ninterp, 0, "pre_fft_post_zero_xb");
            save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xa_d, sett->Ninterp, 1, "pre_fft_post_zero_xa");
            save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xb_d, sett->Ninterp, 1, "pre_fft_post_zero_xb");
#endif
        }

        // fft length fftpad*nfft
        {
            clfftStatus CLFFT_status = CLFFT_SUCCESS;
            cl_event fft_exec[2];
            clfftEnqueueTransform(plans->plan, CLFFT_FORWARD, 1, cl_handles->exec_queues, 0, NULL, &fft_exec[0], &fft_arr->xa_d, NULL, NULL /*May be slow, consider using tmp_buffer*/);
            checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");
            clfftEnqueueTransform(plans->plan, CLFFT_FORWARD, 1, cl_handles->exec_queues, 0, NULL, &fft_exec[1], &fft_arr->xb_d, NULL, NULL /*May be slow, consider using tmp_buffer*/);
            checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");

            clWaitForEvents(2, fft_exec);
        }
#ifdef TESTING
        save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xa_d, sett->nfftf, 0, "post_fft_phasemod_xa");
        save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xb_d, sett->nfftf, 0, "post_fft_phasemod_xb");
        save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xa_d, sett->nfftf, 1, "post_fft_phasemod_xa");
        save_numbered_complex_buffer(cl_handles->exec_queues[0], fft_arr->xb_d, sett->nfftf, 1, "post_fft_phasemod_xb");
#endif
        (*FNum)++;

        compute_Fstat_gpu(fft_arr->xa_d,
                          fft_arr->xb_d,
                          F_d,
                          aux->maa_d,
                          aux->mbb_d,
                          sett->nmin,
                          sett->nmax,
                          cl_handles);
#ifdef TESTING
        save_numbered_real_buffer(cl_handles->exec_queues[0], F_d, sett->nmax - sett->nmin, 0, "Fstat");
        save_numbered_real_buffer(cl_handles->exec_queues[0], F_d, sett->nmax - sett->nmin, 1, "Fstat");
#endif
        cl_int CL_err = CL_SUCCESS;

        CL_err = clEnqueueReadBuffer(cl_handles->read_queues[0], F_d, CL_TRUE, 0, 2 * sett->nfft * sizeof(real_t), F, 0, NULL, NULL);
#ifdef GPUFSTAT
        if (!(opts->white_flag))  // if the noise is not white noise
            FStat_gpu(F_d + sett->nmin, sett->nmax - sett->nmin, NAV, aux->mu_d, aux->mu_t_d);

#else
        // Normalize F-statistics 
        if (!(opts->white_flag))  // if the noise is not white noise
            FStat(F + sett->nmin, sett->nmax - sett->nmin, NAVFSTAT, 0);
#endif

        /*
        FILE *f1 = fopen("fstat-gpu.dat", "w");
        for(i=sett->nmin; i<sett->nmax; i++)
        fprintf(f1, "%d   %lf\n", i, F[i]);

        fclose(f1);
        printf("wrote fstat-gpu.dat | ss=%d  \n", ss);
        //exit(EXIT_SUCCESS);
        */

        for (int i = sett->nmin; i<sett->nmax; i++) {
            if ((Fc = F[i]) > opts->trl) { // if F-stat exceeds trl (critical value)
                                           // Find local maximum for neighboring signals 
                ii = i;

                while (++i < sett->nmax && F[i] > opts->trl) {
                    if (F[i] >= Fc) {
                        ii = i;
                        Fc = F[i];
                    } // if F[i] 
                } // while i 

                  // Candidate signal frequency
                sgnlt[0] = 2.*M_PI*ii / ((double)sett->fftpad*sett->nfft) + sgnl0;
                // Signal-to-noise ratio
                sgnlt[4] = sqrt(2.*(Fc - sett->nd));

                (*sgnlc)++; // increase found number

                            // Add new parameters to output array 
                sgnlv = (FLOAT_TYPE *)realloc(sgnlv, NPAR*(*sgnlc) * sizeof(FLOAT_TYPE));

                for (int j = 0; j<NPAR; ++j) // save new parameters
                    sgnlv[NPAR*(*sgnlc - 1) + j] = (FLOAT_TYPE)sgnlt[j];

#ifdef VERBOSE
                printf("\nSignal %d: %d %d %d %d %d \tsnr=%.2f\n",
                    *sgnlc, pm, mm, nn, ss, ii, sgnlt[4]);
#endif 

            } // if Fc > trl 

        } // for i


#if TIMERS>2
        tend = get_current_time();
        spindown_timer += get_time_difference(tstart, tend);
        spindown_counter++;
#endif


        //    } // if sgnlt[1] 

    } // for ss 


#ifndef VERBOSE
    printf("Number of signals found: %d\n", *sgnlc);
#endif 

#if TIMERS>2
    printf("\nTotal spindown loop time: %e s, mean spindown time: %e s (%d runs)\n",
        spindown_timer, spindown_timer / spindown_counter, spindown_counter);
#endif

    // Non-VLA free _tmp1
    //for (int x = 0; x < sett->nifo; ++x)
    //    free(_tmp1[x]);
    //free(_tmp1);

    return sgnlv;

} // jobcore

void sky_positions(const int pm,                  // hemisphere
                   const int mm,                  // grid 'sky position'
                   const int nn,                  // other grid 'sky position'
                   double* M,                     // M matrix from grid point to linear coord
	               real_t sepsm,
	               real_t cepsm,
	               real_t oms,
                   real_t* sgnlt,
                   real_t* het0,
                   real_t* sgnl0,
                   real_t* ft)
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
  real_t sinalt, cosalt, sindelt, cosdelt;
  lin2ast(al1 / oms, al2 / oms, pm, sepsm, cepsm, // input
          &sinalt, &cosalt, &sindelt, &cosdelt);  // output

																		// calculate declination and right ascention
																		// written in file as candidate signal sky positions
	sgnlt[2] = asin(sindelt);
	sgnlt[3] = fmod(atan2(sinalt, cosalt) + 2.*M_PI, 2.*M_PI);

	*het0 = fmod(nn*M[8] + mm * M[12], M[0]);
}

/// <summary>Copies amplitude modulation coefficients to constant memory.</summary>
///
void copy_amod_coeff(Detector_settings* ifo,
                     cl_int nifo,
                     OpenCL_handles* cl_handles,
                     Aux_arrays* aux)
{
    cl_int CL_err = CL_SUCCESS;

    Ampl_mod_coeff* tmp = clEnqueueMapBuffer(cl_handles->exec_queues[0],
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
    clEnqueueUnmapMemObject(cl_handles->exec_queues[0], aux->ifo_amod_d, tmp, 0, NULL, &unmap_event);

    clWaitForEvents(1, &unmap_event);

    clReleaseEvent(unmap_event);
}

cl_event modvir_gpu(const cl_int idet,
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
  size_t size_Np = (size_t)Np; // Helper variable to make pointer types match. Cast to silence warning

  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 0, sizeof(cl_mem), &aa_d);             checkErr(CL_err, "clSetKernelArg(&aa_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 1, sizeof(cl_mem), &bb_d);             checkErr(CL_err, "clSetKernelArg(&bb_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 2, sizeof(real_t), &cosalfr);          checkErr(CL_err, "clSetKernelArg(&cosalfr)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 3, sizeof(real_t), &sinalfr);          checkErr(CL_err, "clSetKernelArg(&sinalfr)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 4, sizeof(real_t), &c2d);              checkErr(CL_err, "clSetKernelArg(&c2d)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 5, sizeof(real_t), &c2sd);             checkErr(CL_err, "clSetKernelArg(&c2sd)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 6, sizeof(cl_mem), &sinmodf_d);        checkErr(CL_err, "clSetKernelArg(&sinmodf_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 7, sizeof(cl_mem), &cosmodf_d);        checkErr(CL_err, "clSetKernelArg(&cosmodf_d)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 8, sizeof(real_t), &sindel);           checkErr(CL_err, "clSetKernelArg(&sindel)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 9, sizeof(real_t), &cosdel);           checkErr(CL_err, "clSetKernelArg(&cosdel)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 10, sizeof(cl_int), &Np);              checkErr(CL_err, "clSetKernelArg(&Np)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 11, sizeof(cl_int), &idet);            checkErr(CL_err, "clSetKernelArg(&idet)");
  CL_err = clSetKernelArg(cl_handles->kernels[Modvir], 12, sizeof(cl_mem), &ifo_amod_d);      checkErr(CL_err, "clSetKernelArg(&ifo_amod_d)");

  cl_event exec;
  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[0], cl_handles->kernels[Modvir], 1, NULL, &size_Np, NULL, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(cl_handles->kernels[Modvir])");

#ifdef TESTING
  clWaitForEvents(1, &exec);
  save_numbered_real_buffer(cl_handles->exec_queues[0], sinmodf_d, Np, n, "aux_sinmodf");
  save_numbered_real_buffer(cl_handles->exec_queues[0], cosmodf_d, Np, n, "aux_cosmodf");
  save_numbered_real_buffer(cl_handles->exec_queues[0], aa_d, Np, n, "ifo_sig_aa");
  save_numbered_real_buffer(cl_handles->exec_queues[0], bb_d, Np, n, "ifo_sig_bb");
#endif

    return exec;
}

cl_event tshift_pmod_gpu(const real_t shft1,
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

    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 0, sizeof(real_t), &shft1);        checkErr(CL_err, "clSetKernelArg(&shft1)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 1, sizeof(real_t), &het0);         checkErr(CL_err, "clSetKernelArg(&het0)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 2, sizeof(real3_t), &ns);          checkErr(CL_err, "clSetKernelArg(&ns0)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 3, sizeof(cl_mem), &xDat_d);       checkErr(CL_err, "clSetKernelArg(&xDat_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 4, sizeof(cl_mem), &xa_d);         checkErr(CL_err, "clSetKernelArg(&xa_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 5, sizeof(cl_mem), &xb_d);         checkErr(CL_err, "clSetKernelArg(&xb_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 6, sizeof(cl_mem), &shft_d);       checkErr(CL_err, "clSetKernelArg(&shft_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 7, sizeof(cl_mem), &shftf_d);      checkErr(CL_err, "clSetKernelArg(&shftf_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 8, sizeof(cl_mem), &tshift_d);     checkErr(CL_err, "clSetKernelArg(&tshift_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 9, sizeof(cl_mem), &aa_d);         checkErr(CL_err, "clSetKernelArg(&aa_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 10, sizeof(cl_mem), &bb_d);        checkErr(CL_err, "clSetKernelArg(&bb_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 11, sizeof(cl_mem), &DetSSB_d);    checkErr(CL_err, "clSetKernelArg(&DetSSB_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 12, sizeof(real_t), &oms);         checkErr(CL_err, "clSetKernelArg(&oms)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 13, sizeof(cl_int), &N);           checkErr(CL_err, "clSetKernelArg(&N)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 14, sizeof(cl_int), &nfft);        checkErr(CL_err, "clSetKernelArg(&nfft)");
    CL_err = clSetKernelArg(cl_handles->kernels[TShiftPMod], 15, sizeof(cl_int), &interpftpad); checkErr(CL_err, "clSetKernelArg(&interftpad)");

    cl_event exec;
    size_t size_nfft = (size_t)nfft; // Helper variable to make pointer types match. Cast to silence warning

    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[0], cl_handles->kernels[TShiftPMod], 1, NULL, &size_nfft, NULL, num_events_in_wait_list, event_wait_list, &exec);
    checkErr(CL_err, "clEnqueueNDRangeKernel(cl_handles->kernels[TShiftPMod])");

    return exec;
}

void fft_interpolate_gpu(const cl_int idet,
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
	CLFFT_status = clfftEnqueueTransform(plans->pl_int, CLFFT_FORWARD, 1, cl_handles->exec_queues, num_events_in_wait_list, event_wait_list, &fw_fft_events[idet][0], &xa_d, NULL, NULL /*May be slow, consider using tmp_buffer*/);
	checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");
	CLFFT_status = clfftEnqueueTransform(plans->pl_int, CLFFT_FORWARD, 1, cl_handles->exec_queues, num_events_in_wait_list, event_wait_list, &fw_fft_events[idet][1], &xb_d, NULL, NULL /*May be slow, consider using tmp_buffer*/);
	checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");

#ifdef TESTING
	save_numbered_complex_buffer(cl_handles->exec_queues[0], xa_d, nfft, idet, "xa_fourier");
	save_numbered_complex_buffer(cl_handles->exec_queues[0], xb_d, nfft, idet, "xb_fourier");
#endif

	// Resample coefficients
	resample_postfft_events[idet] =
		resample_postfft_gpu(nfft,	                              // input
		                     Ninterp,                             // input
		                     nyqst,                               // input
		                     xa_d,                                // input / output
		                     xb_d,                                // input / output
		                     cl_handles, 2, fw_fft_events[idet]); // sync

#ifdef TESTING
	save_numbered_complex_buffer(cl_handles->exec_queues[0], xa_d, sett->Ninterp, idet, "xa_fourier_resampled");
	save_numbered_complex_buffer(cl_handles->exec_queues[0], xb_d, sett->Ninterp, idet, "xb_fourier_resampled");
#endif

	// Backward fft (len Ninterp = nfft*interpftpad)
	clfftEnqueueTransform(plans->pl_inv, CLFFT_BACKWARD, 1, cl_handles->exec_queues, 1, &resample_postfft_events[idet], &inv_fft_events[idet][0], &xa_d, NULL, NULL /*May be slow, consider using tmp_buffer*/);
	checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_BACKWARD)");
	clfftEnqueueTransform(plans->pl_inv, CLFFT_BACKWARD, 1, cl_handles->exec_queues, 1, &resample_postfft_events[idet], &inv_fft_events[idet][1], &xb_d, NULL, NULL /*May be slow, consider using tmp_buffer*/);
	checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_BACKWARD)");

#ifdef TESTING
	save_numbered_complex_buffer(cl_handles->exec_queues[0], xa_d, Ninterp, idet, "xa_time_resampled");
	save_numbered_complex_buffer(cl_handles->exec_queues[0], xb_d, Ninterp, idet, "xb_time_resampled");
#endif
}

cl_event resample_postfft_gpu(const cl_int nfft,
	                          const cl_int Ninterp,
	                          const cl_int nyqst,
	                          cl_mem xa_d,
	                          cl_mem xb_d,
	                          OpenCL_handles* cl_handles,
	                          const cl_uint num_events_in_wait_list,
	                          const cl_event* event_wait_list)
{
    cl_int CL_err = CL_SUCCESS;

    clSetKernelArg(cl_handles->kernels[ResamplePostFFT], 0, sizeof(cl_mem), &xa_d);    checkErr(CL_err, "clSetKernelArg(&xa_d)");
    clSetKernelArg(cl_handles->kernels[ResamplePostFFT], 1, sizeof(cl_mem), &xb_d);    checkErr(CL_err, "clSetKernelArg(&xb_d)");
    clSetKernelArg(cl_handles->kernels[ResamplePostFFT], 2, sizeof(cl_int), &nfft);    checkErr(CL_err, "clSetKernelArg(&nfft)");
    clSetKernelArg(cl_handles->kernels[ResamplePostFFT], 3, sizeof(cl_int), &Ninterp); checkErr(CL_err, "clSetKernelArg(&Ninterp)");
    clSetKernelArg(cl_handles->kernels[ResamplePostFFT], 4, sizeof(cl_int), &nyqst);   checkErr(CL_err, "clSetKernelArg(&nyqst)");

    cl_event exec;
    size_t resample_length = (size_t)Ninterp - (nyqst + nfft); // Helper variable to make pointer types match. Cast to silence warning

    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[0], cl_handles->kernels[ResamplePostFFT], 1, NULL, &resample_length, NULL, 0, NULL, &exec);
	checkErr(CL_err, "clEnqueueNDRangeKernel(cl_handles->kernels[ResamplePostFFT])");

	return exec;
}

/// <summary>Scales vectors with a constant.</summary>
///
void blas_scale(cl_mem xa_d,
                cl_mem xb_d,
                cl_uint n,
                real_t a,
                OpenCL_handles* cl_handles,
                BLAS_handles* blas_handles)
{
    clblasStatus status[2];
    cl_event blas_exec[2];
#ifdef COMP_FLOAT
    status[0] = clblasSscal(n * 2, a, xa_d, 0, 1, 1, cl_handles->exec_queues, 0, NULL, &blas_exec[0]); checkErrBLAS(status[0], "clblasSscal(xa_d)");
    status[1] = clblasSscal(n * 2, a, xb_d, 0, 1, 1, cl_handles->exec_queues, 0, NULL, &blas_exec[1]); checkErrBLAS(status[1], "clblasSscal(xb_d)");
#else
    status[0] = clblasDscal(n * 2, a, xa_d, 0, 1, 1, cl_handles->exec_queues, 0, NULL, &blas_exec[0]); checkErrBLAS(status[0], "clblasDscal(xa_d)");
    status[1] = clblasDscal(n * 2, a, xb_d, 0, 1, 1, cl_handles->exec_queues, 0, NULL, &blas_exec[1]); checkErrBLAS(status[1], "clblasDscal(xb_d)");
#endif // COMP_FLOAT

    clWaitForEvents(2, blas_exec);

    for (size_t i = 0; i < 2; ++i) clReleaseEvent(blas_exec[i]);
}

/// <summary>Calculates the inner product of both <c>x</c> and <c>y</c>.</summary>
/// <remarks>The function allocates an array of 2 and gives ownership to the caller.</remarks>
/// <remarks>Consider making the temporaries persistent, either providing them via function params or give static storage duration.</remarks>
///
real_t* blas_dot(cl_mem x,
                 cl_mem y,
                 cl_uint n,
                 OpenCL_handles* cl_handles,
                 BLAS_handles* blas_handles)
{
    cl_int CL_err = CL_SUCCESS;
    clblasStatus status[2];
    cl_event blas_exec[2], unmap;
    cl_mem result_buf, scratch_buf[2];

    real_t* result = (real_t*)malloc(2 * sizeof(real_t));

    result_buf = clCreateBuffer(cl_handles->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 2 * sizeof(real_t), NULL, &CL_err);
    scratch_buf[0] = clCreateBuffer(cl_handles->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, n * sizeof(real_t), NULL, &CL_err);
    scratch_buf[1] = clCreateBuffer(cl_handles->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, n * sizeof(real_t), NULL, &CL_err);

#ifdef COMP_FLOAT
    status[0] = clblasSdot(n, result_buf, 0, x, 0, 1, x, 0, 1, scratch_buf[0], 1, cl_handles->exec_queues, 0, NULL, &blas_exec[0]);
    status[1] = clblasSdot(n, result_buf, 1, y, 0, 1, y, 0, 1, scratch_buf[1], 1, cl_handles->exec_queues, 0, NULL, &blas_exec[1]);
#else
    status[0] = clblasDdot(n, result_buf, 0, x, 0, 1, x, 0, 1, scratch_buf[0], 1, cl_handles->exec_queues, 0, NULL, &blas_exec[0]);
    status[1] = clblasDdot(n, result_buf, 1, y, 0, 1, y, 0, 1, scratch_buf[1], 1, cl_handles->exec_queues, 0, NULL, &blas_exec[1]);
#endif // COMP_FLOAT

    void* res = clEnqueueMapBuffer(cl_handles->read_queues[0], result_buf, CL_TRUE, CL_MAP_READ, 0, 2 * sizeof(real_t), 2, blas_exec, NULL, &CL_err);
    checkErr(CL_err, "clEnqueueMapMemObject(result_buf)");

#ifdef _WIN32
    errno_t CRT_err = memcpy_s(result, 2 * sizeof(real_t), res, 2 * sizeof(real_t));
    if (CRT_err != 0)
        exit(EXIT_FAILURE);
#else
    memcpy(result, res, 2 * sizeof(real_t));
#endif

    CL_err = clEnqueueUnmapMemObject(cl_handles->read_queues[0], result_buf, res, 0, NULL, &unmap);
    checkErr(CL_err, "clEnqueueUnmapMemObject(result_buf)");

    clWaitForEvents(1, &unmap);

    // cleanup
    clReleaseEvent(blas_exec[0]);
    clReleaseEvent(blas_exec[1]);
    clReleaseEvent(unmap);
    clReleaseMemObject(result_buf);
    clReleaseMemObject(scratch_buf[0]);
    clReleaseMemObject(scratch_buf[1]);

    return result;
}

/// <summary>The purpose of this function was undocumented.</summary>
///
void phase_mod_1_gpu(cl_mem xa,
                     cl_mem xb,
                     cl_mem xar,
                     cl_mem xbr,
                     real_t het1,
                     real_t sgnlt1,
                     cl_mem shft,
                     cl_int N,
                     OpenCL_handles* cl_handles)
{
    cl_int CL_err = CL_SUCCESS;

    clSetKernelArg(cl_handles->kernels[PhaseMod1], 0, sizeof(cl_mem), &xa);
    clSetKernelArg(cl_handles->kernels[PhaseMod1], 1, sizeof(cl_mem), &xb);
    clSetKernelArg(cl_handles->kernels[PhaseMod1], 2, sizeof(cl_mem), &xar);
    clSetKernelArg(cl_handles->kernels[PhaseMod1], 3, sizeof(cl_mem), &xbr);
    clSetKernelArg(cl_handles->kernels[PhaseMod1], 4, sizeof(real_t), &het1);
    clSetKernelArg(cl_handles->kernels[PhaseMod1], 5, sizeof(real_t), &sgnlt1);
    clSetKernelArg(cl_handles->kernels[PhaseMod1], 6, sizeof(cl_mem), &shft);
    clSetKernelArg(cl_handles->kernels[PhaseMod1], 7, sizeof(cl_int), &N);

    cl_event exec;
    size_t size_N = (size_t)N; // Helper variable to make pointer types match. Cast to silence warning

    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[0], cl_handles->kernels[PhaseMod1], 1, NULL, &size_N, NULL, 0, NULL, &exec);
    checkErr(CL_err, "clEnqueueNDRangeKernel(PhaseMod1)");

    clWaitForEvents(1, &exec);

    clReleaseEvent(exec);
}

/// <summary>The purpose of this function was undocumented.</summary>
///
void phase_mod_2_gpu(cl_mem xa,
                     cl_mem xb,
                     cl_mem xar,
                     cl_mem xbr,
                     real_t het1,
                     real_t sgnlt1,
                     cl_mem shft,
                     cl_int N,
                     OpenCL_handles* cl_handles)
{
    cl_int CL_err = CL_SUCCESS;

    clSetKernelArg(cl_handles->kernels[PhaseMod2], 0, sizeof(cl_mem), &xa);
    clSetKernelArg(cl_handles->kernels[PhaseMod2], 1, sizeof(cl_mem), &xb);
    clSetKernelArg(cl_handles->kernels[PhaseMod2], 2, sizeof(cl_mem), &xar);
    clSetKernelArg(cl_handles->kernels[PhaseMod2], 3, sizeof(cl_mem), &xbr);
    clSetKernelArg(cl_handles->kernels[PhaseMod2], 4, sizeof(real_t), &het1);
    clSetKernelArg(cl_handles->kernels[PhaseMod2], 5, sizeof(real_t), &sgnlt1);
    clSetKernelArg(cl_handles->kernels[PhaseMod2], 6, sizeof(cl_mem), &shft);
    clSetKernelArg(cl_handles->kernels[PhaseMod2], 7, sizeof(cl_int), &N);

    cl_event exec;
    size_t size_N = (size_t)N; // Helper variable to make pointer types match. Cast to silence warning

    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[0], cl_handles->kernels[PhaseMod2], 1, 0, &size_N, NULL, 0, NULL, &exec);
    checkErr(CL_err, "clEnqueueNDRangeKernel(PhaseMod2)");

    clWaitForEvents(1, &exec);

    clReleaseEvent(exec);
}

/// <summary>Compute F-statistics.</summary>
/// 
void compute_Fstat_gpu(cl_mem xa,
                       cl_mem xb,
                       cl_mem F,
                       cl_mem maa_d,
                       cl_mem mbb_d,
                       cl_int nmin,
                       cl_int nmax,
                       OpenCL_handles* cl_handles)
{
    cl_int CL_err = CL_SUCCESS;
    cl_int N = nmax - nmin;

    clSetKernelArg(cl_handles->kernels[ComputeFStat], 0, sizeof(cl_mem), &xa);
    clSetKernelArg(cl_handles->kernels[ComputeFStat], 1, sizeof(cl_mem), &xb);
    clSetKernelArg(cl_handles->kernels[ComputeFStat], 2, sizeof(cl_mem), &F);
    clSetKernelArg(cl_handles->kernels[ComputeFStat], 3, sizeof(cl_mem), &maa_d);
    clSetKernelArg(cl_handles->kernels[ComputeFStat], 4, sizeof(cl_mem), &mbb_d);
    clSetKernelArg(cl_handles->kernels[ComputeFStat], 5, sizeof(cl_int), &N);

    cl_event exec;
    size_t size_N = (size_t)N,
           size_nmin = (size_t)nmin; // Helper variable to make pointer types match. Cast to silence warning

    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[0], cl_handles->kernels[ComputeFStat], 1, &size_nmin, &size_N, NULL, 0, NULL, &exec);
    checkErr(CL_err, "clEnqueueNDRangeKernel(ComputeFStat)");

    clWaitForEvents(1, &exec);

    clReleaseEvent(exec);
}

/// <summary>Compute F-statistics.</summary>
///
void FStat_gpu_simple(cl_mem F_d,
    cl_uint nfft,
    cl_uint nav,
    OpenCL_handles* cl_handles)
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint N = nfft / nav;
    size_t max_wgs;             // maximum supported wgs on the device (limited by register count)
    cl_ulong local_size;        // local memory size in bytes
    cl_uint ssi;                // shared size in num gentypes

    CL_err = clGetKernelWorkGroupInfo(cl_handles->kernels[FStatSimple], cl_handles->devs[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wgs, NULL);
    checkErr(CL_err, "clGetKernelWorkGroupInfo(FStatSimple, CL_KERNEL_WORK_GROUP_SIZE)");

    CL_err = clGetDeviceInfo(cl_handles->devs[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_size, NULL);
    checkErr(CL_err, "clGetDeviceInfo(FStatSimple, CL_DEVICE_LOCAL_MEM_SIZE)");

    // How long is the array of local memory
    ssi = (cl_uint)(local_size / sizeof(real_t)); // Assume integer is enough to store gentype count (well, it better)

    clSetKernelArg(cl_handles->kernels[FStatSimple], 0, sizeof(cl_mem), &F_d);
    clSetKernelArg(cl_handles->kernels[FStatSimple], 1, ssi * sizeof(real_t), NULL);
    clSetKernelArg(cl_handles->kernels[FStatSimple], 2, sizeof(cl_uint), &ssi);
    clSetKernelArg(cl_handles->kernels[FStatSimple], 3, sizeof(cl_uint), &nav);

    size_t gsi = N;

    size_t wgs = gsi < max_wgs ? gsi : max_wgs;

    cl_event exec;
    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[0], cl_handles->kernels[FStatSimple], 1, NULL, &gsi, &wgs, 0, NULL, &exec);
    checkErr(CL_err, "clEnqueueNDRangeKernel(FStatSimple)");

    clWaitForEvents(1, &exec);

    clReleaseEvent(exec);
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
