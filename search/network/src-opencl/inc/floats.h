#pragma once

// Polgraw includes
#include <precision.h>

// OpenCL includes
#include <CL/cl.h>      // cl_floatN, cl_doubleN

// Standard C includes
#include <complex.h>      // _Fcomplex, _Dcomplex

#if AMPL_MOD_DOUBLE
typedef double ampl_mod_real;
#else
typedef float ampl_mod_real;
#endif

#if XDAT_DOUBLE
typedef double xDat_real;
#else
typedef float xDat_real;
#endif

#if DETSSB_DOUBLE
typedef cl_double3 DetSSB_real3;
#else
typedef cl_float3 DetSSB_real3;
#endif

#if XDATM_DOUBLE
  #ifdef _WIN32
  typedef _Dcomplex xDatm_complex;
  #else
  typedef complex double xDatm_complex;
  #endif
#else
  #ifdef _WIN32
  typedef _Fcomplex xDatm_complex;
  #else
  typedef complex float xDatm_complex;
  #endif
#endif

#if FFT_DOUBLE
  #ifdef _WIN32
  typedef _Dcomplex fft_complex;
  #else
  typedef complex double fft_complex;
  #endif
#else
  #ifdef _WIN32
  typedef _Fcomplex fft_complex;
  #else
  typedef complex float fft_complex;
  #endif
#endif

#if SHIFT_DOUBLE
typedef double shift_real;
#else
typedef float shift_real;
#endif

#if FSTAT_DOUBLE
typedef double fstat_real;
#else
typedef float fstat_real;
#endif

#if MODVIR_DOUBLE
typedef double modvir_real;
#else
typedef float modvir_real;
#endif

#if TSHIFT_PMOD_DOUBLE
typedef cl_double tshift_pmod_real;
typedef cl_double3 tshift_pmod_real3;
#else
typedef cl_float tshift_pmod_real;
typedef cl_float3 tshift_pmod_real3;
#endif

#if SPLINE_DOUBLE
typedef double spline_real;
  #ifdef _WIN32
  typedef _Dcomplex spline_complex;
  #else
  typedef complex double spline_complex;
  #endif
#else
typedef float spline_real;
  #ifdef _WIN32
  typedef _Fcomplex spline_complex;
  #else
  typedef complex float spline_complex;
  #endif
#endif

#if PHASE_MOD_DOUBLE
typedef double phase_mod_real;
  #ifdef _WIN32
  typedef _Dcomplex phase_mod_complex;
  #else
  typedef complex double phase_mod_complex;
  #endif
#else
typedef float phase_mod_real;
  #ifdef _WIN32
  typedef _Fcomplex phase_mod_complex;
  #else
  typedef complex float phase_mod_complex;
  #endif
#endif

#if INTERIM_FSTAT_DOUBLE
typedef double interim_fstat_real;
#else
typedef float interim_fstat_real;
#endif
