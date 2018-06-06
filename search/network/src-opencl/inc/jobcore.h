#ifndef __JOBCORE_H__
#define __JOBCORE_H__

// Polgraw includes
#include <struct.h>     // Search_settings, Command_line_opts, Search_range, FFT_plans, FFT_arrays, Aux_arrays
#include <floats.h>     // FLOAT_TYPE, HOST_COMPLEX_TYPE

// clBLAS includes
#include <clBLAS.h>


#define BLOCK_SIZE 256
#define BLOCK_SIZE_RED 128
#define BLOCK_DIM(n, b) ((n)/b + ((n)%b==0 ? 0 : 1))
#define NAV_THREADS 16

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
///
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
                 cl_mem F,                      // F-statistics array
                 int *sgnlc,                    // reference to array with the parameters of the candidate signal
                                                // (used below to write to the file)
                 int *FNum,                     // candidate signal number
                 OpenCL_handles* cl_handles,    // handles to OpenCL resources
                 BLAS_handles* blas_handles);   // handle for scaling

/// <summary>Copies amplitude modulation coefficients to constant memory.</summary>
///
void copy_amod_coeff(Detector_settings* ifo,
                     cl_int nifo,
                     OpenCL_handles* cl_handles,
                     Aux_arrays* aux);

/// <summary>Calculate the amplitude modulation functions aa and bb of the given detector (in signal sub-structs: ifo->sig.aa, ifo->.sig.bb).</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
///
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
                    const cl_event* event_wait_list);

/// <summary>The purpose of this function was undocumented.</summary>
/// <remarks>Ownership of the event created internally is transfered to the caller.</remarks>
///
cl_event tshift_pmod_gpu(const real_t shft1,
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

/// <summary>Shifts frequencies and remove those over Nyquist.</summary>
///
void resample_postfft_gpu(cl_mem xa_d,
                          cl_mem xb_d,
                          cl_int nfft,
                          cl_int Ninterp,
                          cl_int nyqst,
                          OpenCL_handles* cl_handles);

/// <summary>Scales vectors with a constant.</summary>
///
void blas_scale(cl_mem xa_d,
                cl_mem xa_b,
                cl_uint n,
                real_t a,
                OpenCL_handles* cl_handles,
                BLAS_handles* blas_handles);

/// <summary>Calculates the inner product of both <c>x</c> and <c>y</c>.</summary>
/// <remarks>The function allocates an array of 2 and gives ownership to the caller.</remarks>
/// <remarks>Consider making the temporaries persistent, either providing them via function params or give static storage duration.</remarks>
///
real_t* blas_dot(cl_mem x,
                 cl_mem y,
                 cl_uint n,
                 OpenCL_handles* cl_handles,
                 BLAS_handles* blas_handles);

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
                     OpenCL_handles* cl_handles);

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
                     OpenCL_handles* cl_handles);

/// <summary>Compute F-statistics.</summary>
/// 
void compute_Fstat_gpu(cl_mem xa,
                       cl_mem xb,
                       cl_mem F,
                       cl_mem maa_d,
                       cl_mem mbb_d,
                       cl_int nmin,
                       cl_int nmax,
                       OpenCL_handles* cl_handles);

/// <summary>Compute F-statistics.</summary>
///
void FStat_gpu_simple(cl_mem F_d,
                      cl_uint nfft,
                      cl_uint nav,
                      OpenCL_handles* cl_handles);

/// <summary>Saves the designated array into a file with the specified name.</summary>
///
void save_array(HOST_COMPLEX_TYPE *arr, int N, const char* file);

/// <summary>Prints the first 'n' values of a host side real array.</summary>
///
void print_real_array(real_t* arr, size_t count, const char* msg);

/// <summary>Prints the first 'n' values of a host side complex array.</summary>
///
void print_complex_array(complex_t* arr, size_t count, const char* msg);

/// <summary>Prints the first 'n' values of a device side real array.</summary>
///
void print_real_buffer(cl_command_queue queue, cl_mem buf, size_t count, const char* msg);

/// <summary>Prints the first 'n' values of a device side complex array.</summary>
///
void print_complex_buffer(cl_command_queue queue, cl_mem buf, size_t count, const char* msg);

double FStat (double *, int, int, int);
void FStat_gpu(FLOAT_TYPE *F_d, int N, int nav, FLOAT_TYPE *mu_d, FLOAT_TYPE *mu_t_d);

#endif
