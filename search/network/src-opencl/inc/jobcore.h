#ifndef __JOBCORE_H__
#define __JOBCORE_H__

// Polgraw includes
#include <struct.h>     // Search_settings, Command_line_opts, Search_range, FFT_plans, FFT_arrays, Aux_arrays
#include <floats.h>     // FLOAT_TYPE, HOST_COMPLEX_TYPE
#include <signal_params.h>

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
            int* Fnum);

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

/// <summary>Calculates spindown values to process.</summary>
///
void spindown_range(const int mm,                  // grid 'sky position'
                    const int nn,                  // other grid 'sky position'
                    const double Smin,
                    const double Smax,
                    const double* M,               // M matrix from grid point to linear coord
                    const Search_range* s_range,
                    const Command_line_opts *opts,
                    int* smin,
                    int* smax);

/// <summary>Saves all the signal candidates to disk.</summary>
///
void save_and_free_results(const Command_line_opts* opts,
                           const Search_range* s_range,
                           Search_results*** results);

/// <summary>Combines the search results of a hemisphere to be saved on disk.</summary>
///
Search_results combine_results(const Search_range* s_range,
                               const Search_results** results);

/// <summary>Initializes search results with empty data where checkpoint file contained data.</summary>
///
Search_results*** init_results(const Search_range* s_range);

/// <summary>Allocates OpenCL event objects to synchronize and profile the pipeline.</summary>
///
Pipeline init_pipeline(const size_t nifo);

/// <summary>Deallocates OpenCL event objects to synchronize and profile the pipeline.</summary>
///
void free_pipeline(const size_t nifo,
                   Pipeline* p);

/// <summary>Initializes pipeline profiling data.</summary>
///
Profiling_info init_profiling_info();

/// <summary>Extracts profiling info from OpenCL events and CRT time points.</summary>
///
void extract_profiling_info(const struct timespec* pre_spindown_start,
                            const struct timespec* pre_spindown_end,
                            const struct timespec* spindown_end,
                            const Pipeline pipeline,
                            Profiling_info* prof);

/// <summary>Prints profiling info to console.</summary>
///
void print_profiling_info(const Profiling_info prof);

#endif
