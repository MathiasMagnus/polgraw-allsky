#pragma once

// Polgraw includes
#include <floats.h>
#include <signal_params.h>

// clBLAS includes
#include <clBLAS.h>

// clFFT includes
#include <clFFT.h>

// OpenCL includes
#include <CL/cl.h>

// Standard C includes
#include <time.h>       // timespec

#define MAX_DETECTORS 2        // Maximum number of detectors in network
#define DETNAME_LENGTH 3       // Detector name length (H1, L1, V1...) + null terminator
#define XDATNAME_LENGTH 512    // Maximum length of input file name xdat*bin
#define MAX_DEVICES 8

#define MAXL 2048              // Max number of known lines for a detector

// Command line option struct for search 
typedef struct _comm_line_opts
{  
  int white_flag,  // white noise flag
      s0_flag,     // no spin-down flag
      checkp_flag, // checkpointing flag
      veto_flag,   // veto lines flag
      help_flag;
  
  int ident, band, hemi;
  double trl;
  double fpo_val;
  
  char prefix[512], dtaprefix[512], label[512],
       range[512], getrange[512], qname[512],
       usedet[32], addsig[512], *wd,
       plat_ids[512], dev_types[512], dev_ids[512];

} Command_line_opts;


/* input signal arrays */
typedef struct _signals
{
  double* xDat;
  cl_mem* xDat_d;
  DetSSB_real3* DetSSB;
  cl_mem *DetSSB_d;          // Ephemeris of the detector
  cl_mem *aa_d, *bb_d;      // Amplitude modulation functions
  cl_mem *shftf_d, *shft_d; // Resampling and time-shifting
  cl_mem *xDatma_d, *xDatmb_d;
  
  double epsm,
         phir,
         sepsm,           // sin(epsm)
         cepsm,           // cos(epsm)
         sphir,           // sin(phi_r)
         cphir,           // cos(phi_r)
         crf0,            // number of 0s as: N/(N-Nzeros)
         sig2;            // variance of signal

  int Nzeros;

} Signals;


/* fftw arrays */
typedef struct _fft_arrays
{
  cl_mem **xa_d,
         **xb_d;
  int arr_len;

} FFT_arrays;

/// <summary>Persistent storage of temporary arrays for BLAS operations.</summary>
///
typedef struct _blas_handles
{
  cl_mem **aaScratch_d,
         **bbScratch_d;

} BLAS_handles;


/// <summary>Range of sky coordinates to search.</summary>
///
typedef struct _search_range
{
  int pmr[2], mr[2], nr[2], spndr[2];
  int pst, mst, nst, sst;

  int coord_count;
  int* sky_coords; // [{pm,mm,nn},...]

} Search_range;

/// <summary>Struct holding OpenCL-related user preferences.</summary>
///
typedef struct _opencl_settings
{
  cl_uint count;
  cl_uint plat_ids[MAX_DEVICES];
  cl_device_type dev_types[MAX_DEVICES];
  cl_uint dev_ids[MAX_DEVICES];

} OpenCL_settings;

/// <summary>Struct holding OpenCL device information.</summary>
///
typedef struct _opencl_handles
{
  cl_uint count;
  cl_platform_id* plats;
  cl_device_id* devs;
  cl_context* ctxs;
  cl_command_queue **write_queues,
                   **exec_queues,
                   **read_queues;
  cl_program* progs;
  cl_kernel** kernels;

} OpenCL_handles;

/// <summary>Enum for human readable kernel indicies.</summary>
///
enum Kernel
{
  Modvir = 0,
  TShiftPMod,
  ResamplePostFFT,
  PhaseMod1,
  PhaseMod2,
  ComputeFStat,
  NormalizeFStatWG
};
static const cl_uint kernel_count = 7;

/* FFTW plans  */ 
typedef struct _fft_plans
{
  clfftPlanHandle *plan,    // main plan
                  *pl_int,  // interpolation forward
                  *pl_inv;  // interpolation backward
} FFT_plans;


/* Auxiluary arrays  */ 
typedef struct _aux_arrays
{
  cl_mem *ifo_amod_d;            // constant buffers of detector settings
  cl_mem **tshift_d;
  cl_mem **aadots_d, **bbdots_d; // array of sub-buffers pointing into xxdot_d
  cl_mem *maa_d, *mbb_d;
  cl_mem *F_d;                   // F-statistics

} Aux_arrays;


// Search settings //
typedef struct _search_settings
{
  double fpo,    // Band frequency
         dt,     // Sampling time
         B,      // Bandwidth
         oms,    // Dimensionless angular frequency (fpo)
         omr,    // C_OMEGA_R * dt
                 // (dimensionless Earth's angular frequency)
         Smin,   // Minimum spindown
         Smax,   // Maximum spindown
         sepsm,  // sin(epsm)
         cepsm;  // cos(epsm)
  
  int nfft,       // length of fft
      nyqst,      // Nyquist frequency
      nod,        // number of days of observation
      N,          // number of data points
      nfftf,      // nfft * fftpad
      nmax,       // first and last point
      nmin,       // of Fstat
      s,          // number of spindowns
      nd,         // degrees of freedom
      interpftpad,
      fftpad,     // zero padding
      Ninterp,    // for resampling (set in plan_fftw() init.c)
      nifo;       // number of detectors

  double* M;      // Grid-generating matrix (or Fisher matrix,
                  // in case of coincidences)

  double vedva[4][4];   // transformation matrix: its columns are
                        // eigenvectors, each component multiplied
                        // by sqrt(eigval), see init.c manage_grid_matrix():
                        // sett->vedva[i][j]  = eigvec[i][j]*sqrt(eigval[j])

  double lines[MAXL][2]; // Array for lines in given band
  int numlines_band;     // number of lines in band

} Search_settings;

/// <summary>Amplitude modulation function coefficients</summary>
///
typedef struct _ampl_mod_coeff
{
  ampl_mod_real c1, c2, c3, c4, c5, c6, c7, c8, c9;

} Ampl_mod_coeff;


  /* Detector and its data related settings 
   */ 

typedef struct _detector
{
  char xdatname[XDATNAME_LENGTH]; 
  char name[DETNAME_LENGTH]; 
 
  double ephi,      // Geographical latitude phi in radians
         elam,      // Geographical longitude in radians
         eheight,   // Height h above the Earth ellipsoid in meters
         egam;      // Orientation of the detector gamma

  Ampl_mod_coeff amod; 
  Signals sig;  

  double lines[MAXL][2]; // Array for lines: column values
                         // are beginning and end of line to veto
  int numlines;                        
 
} Detector_settings; 


  /* Array of detectors (network) 
   */ 

//extern Detector_settings ifo[MAX_DETECTORS]; // moved to main()

// Command line option struct for coincidences 
typedef struct _comm_line_opts_coinc
{  
  int help_flag; 
  
  int shift, // Cell shifts  (4 digit number corresponding to fsda, e.g. 0101)
      scale, // Cell scaling (4 digit number corresponding to fsda, e.g. 4824)
      refr;  // Reference frame

  // Minimal number of coincidences recorded in the output
  int mincoin; 

  double fpo, refgps, narrowdown, snrcutoff; 
  
  char prefix[512], dtaprefix[512], trigname[512], refloc[512], *wd;
  
} Command_line_opts_coinc;

typedef struct _triggers
{
  int frameinfo[256][3];    // Info about candidates in frames:
                            // - [0] frame number, [1] initial number
                            // of candidates, [2] number of candidates
                            // after sorting

  int frcount, goodcands; 

} Candidate_triggers; 

/// <summary>Used to store OpenCL event objects to synchronize and profile the pipeline</summary>
///
typedef struct _pipeline
{
  cl_event *modvir_events,
           *tshift_pmod_events;
  cl_event **fft_interpolate_fw_fft_events,
           **fft_interpolate_resample_copy_events,
           **fft_interpolate_resample_fill_events,
           **fft_interpolate_inv_fft_events,
           **spline_map_events,
           **spline_unmap_events,
           **spline_blas_events,
           **blas_dot_events;
  cl_event *mxx_fill_events,
           *axpy_events,
           *phase_mod_events,
           *zero_pad_events,
           *fw2_fft_events;
  cl_event compute_Fstat_event,
           normalize_Fstat_event,
           peak_map_event,
           peak_unmap_event;

} Pipeline;

/// <summary>Holds execution durations of various sorts in nanoseconds.</summary>
///
typedef struct _profiling_info
{
  cl_ulong modvir_exec,
           tshift_pmod_exec,
           fft_interpolate_fw_fft_exec,
           fft_interpolate_resample_copy_exec,
           fft_interpolate_resample_fill_exec,
           fft_interpolate_inv_fft_exec,
           spline_map_exec,
           spline_unmap_exec,
           spline_blas_exec,
           blas_dot_exec,
           mxx_fill_exec,
           axpy_exec,
           phase_mod_exec,
           zero_pad_exec,
           fw2_fft_exec,
           compute_Fstat_exec,
           normalize_Fstat_exec,
           find_peak_exec;

  unsigned long long pre_spindown_exec,
                     spindown_exec;

} Profiling_info;

/// <summary>Used to communicate candidate signals between each <c>job_core</c> invocation and <c>search</c>.</summary>
///
typedef struct _search_results
{
    size_t sgnlc;       // Size of candidate signal array
    double* sgnlv;      // Array of candidate signals
    Profiling_info prof;// Profiling info of pipeline exec

} Search_results;
