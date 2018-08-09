#ifndef __JOBCORE_H__
#define __JOBCORE_H__

// Polgraw includes
#include <struct.h>     // Search_settings, Command_line_opts, Search_range, FFT_plans, FFT_arrays, Aux_arrays
#include <floats.h>     // FLOAT_TYPE, HOST_COMPLEX_TYPE

// clBLAS includes
#include <clBLAS.h>


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
            cl_mem F_d);

/// <summary>Main job function.</summary>
/// <remarks>The output is stored in single or double precision. (<c>real_t</c> defined in struct.h)</remarks>
/// <todo>Make the output a struct of size_t and real_t* instead of intricate external <c>sgnlc</c>.</todo>
/// <todo>Make sky position dependent setup a callback function and make it awaitable through an event.</todo>
/// <todo>Make multiple <c>xa_d, xb_d</c> buffers for each detector, cause currently there's a race</todo>
///
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
                        BLAS_handles* blas_handles);   // handle for scaling

/// <summary>Calculates sky-position dependant quantities.</summary>
///
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
	               real_t* cosdelt);

/// <summary>Copies amplitude modulation coefficients to constant memory.</summary>
///
void copy_amod_coeff(Detector_settings* ifo,
                     cl_int nifo,
                     OpenCL_handles* cl_handles,
                     Aux_arrays* aux);

/// <summary>Calculate the amplitude modulation functions aa and bb of the given detector (in signal sub-structs: ifo->sig.aa, ifo->.sig.bb).</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
/// <remarks>Becomes a blocking call when <c>TESTING</c> is enabled</remarks>
///
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
                    const cl_event* event_wait_list);

/// <summary>The purpose of this function was undocumented.</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
/// <remarks>Becomes a blocking call when <c>TESTING</c> is enabled</remarks>
///
cl_event tshift_pmod_gpu(const cl_int idet,
                         const cl_int id,
	                     const real_t shft1,
                         const real_t het0,
                         const real3_t ns,
                         //const real_t ns1,
                         //const real_t ns2,
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
                         const cl_event* event_wait_list);

/// <summary>Interpolates in Fourier-space.</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
/// <remarks>Storage for the events must be provided by the caller.</remarks>
/// <todo>Create persistent storage for FFT temporary.</todo>
///
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
	                     cl_event** inv_fft_events);

/// <summary>Shifts frequencies and remove those over Nyquist.</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
///
cl_event resample_postfft_gpu(const cl_int idet,
                              const cl_int id,
                              const cl_int nfft,
                              const cl_int Ninterp,
                              const cl_int nyqst,
	                          cl_mem xa_d,
	                          cl_mem xb_d,
	                          OpenCL_handles* cl_handles,
	                          const cl_uint num_events_in_wait_list,
	                          const cl_event* event_wait_list);

/// <summary>Scales vectors with a constant.</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
/// <remarks>Storage for the events must be provided by the caller.</remarks>
///
void blas_scale(const cl_uint n,
	            const real_t a,
	            cl_mem xa_d,
	            cl_mem xb_d,
	            BLAS_handles* blas_handles,
	            OpenCL_handles* cl_handles,
	            const cl_uint num_events_in_wait_list,
	            const cl_event* event_wait_list,
	            cl_event* blas_exec);

/// <summary>Calculates the inner product of both <c>x</c> and <c>y</c>.</summary>
///
void blas_dot(const cl_uint n,
	          const cl_mem aa_d,
              const cl_mem bb_d,
	          cl_mem aadot_d,
              cl_mem bbdot_d,
	          BLAS_handles* blas_handles,
              OpenCL_handles* cl_handles,
	          const cl_uint num_events_in_wait_list,
	          const cl_event* event_wait_list,
	          cl_event* blas_exec);

/// <summary>Sum up results from all detectors.</summary>
/// <todo>Factor out the plain axpy call to separate function</todo>
/// <todo>Create an alternate implementation not doing a series of
///       scalar axpy on device, but mapping/unmapping and summing on host.</todo>
///
void calc_mxx(const cl_uint nifo,
	          const cl_mem aadot_d,
	          const cl_mem bbdot_d,
	          const Detector_settings* ifo,
	          cl_mem maa_d,
	          cl_mem mbb_d,
	          OpenCL_handles* cl_handles,
	          const cl_uint num_events_in_wait_list,
	          const cl_event* event_wait_list,
	          cl_event* mxx_fill_events,
	          cl_event* axpy_events);

/// <summary>Calculates spindown values to process.</summary>
///
void spindown_range(const int mm,                  // grid 'sky position'
                    const int nn,                  // other grid 'sky position'
                    const real_t Smin,
                    const real_t Smax,
                    const double* M,               // M matrix from grid point to linear coord
                    const Search_range* s_range,
                    const Command_line_opts *opts,
                    int* smin,
                    int* smax);

/// <summary>The purpose of this function was undocumented.</summary>
/// <todo>Merge phase_mod_1 and phase_mod_2 via zeroing out result arrays initially.</todo>
///
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
                         const cl_event* event_wait_list);

/// <summary>The purpose of this function was undocumented.</summary>
/// <todo>Merge phase_mod_1 and phase_mod_2 via zeroing out result arrays initially.</todo>
///
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
                         const cl_event* event_wait_list);

/// <summary>Zero pad from offset until the end of the buffer.</summary>
///
void zero_pad(const cl_int idet,
              const cl_int id,
              const Search_settings *sett,
              cl_mem xa_d,
              cl_mem xb_d,  
              OpenCL_handles* cl_handles,
              const cl_uint num_events_in_wait_list,
              const cl_event* event_wait_list,
              cl_event* zero_pad_events);

/// <summary>Transform data from time-domain to frequency domain for F-statistics.</summary>
///
void time_to_frequency(const cl_int idet,
                       const cl_int id,
                       const Search_settings *sett,
                       const FFT_plans* plans,
                       cl_mem xa_d,
                       cl_mem xb_d,
                       OpenCL_handles* cl_handles,
                       const cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* fw2_fft_events);

/// <summary>Compute F-statistics.</summary>
/// 
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
                           const cl_event* event_wait_list);

/// <summary>Normalize F-statistics.</summary>
///
cl_event normalize_FStat_gpu_wg_reduce(const cl_int idet,
                                       const cl_int id,
                                       const cl_uint nfft,
                                       const cl_uint nav,
                                       cl_mem F_d,
                                       OpenCL_handles* cl_handles,
                                       const cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list);

/// <summary>Looks for peaks above treshold <c>trl</c> and persists them.</summary>
///
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
                cl_event* peak_unmap_event);

/// <summary>Saves all the signal candidates to disk.</summary>
///
void save_and_free_results(const Command_line_opts* opts,
                           const Search_range* s_range,
                           const Search_results*** results);

/// <summary>Combines the search results of a hemisphere to be saved on disk.</summary>
///
Search_results combine_results(const Search_range* s_range,
                               const Search_results** results);

/// <summary>Initializes search results with empty data where checkpoint file contained data.</summary>
///
Search_results*** init_results(const Search_range* s_range);

double FStat (double *, int, int, int);
void FStat_gpu(FLOAT_TYPE *F_d, int N, int nav, FLOAT_TYPE *mu_d, FLOAT_TYPE *mu_t_d);

#endif
