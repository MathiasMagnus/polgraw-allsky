#ifndef __FLOATS_H_CL__
#define __FLOATS_H_CL__

// Polgraw includes
#include <../inc/precision.h>

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
typedef double3 DetSSB_real3;
#else
typedef float3 DetSSB_real3;
#endif

#if XDATM_DOUBLE
typedef double2 xDatm_complex;
#else
typedef float2 xDatm_complex;
#endif

#if FFT_DOUBLE
typedef double2 fft_complex;
#else
typedef float2 fft_complex;
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
typedef double tshift_pmod_real;
typedef double3 tshift_pmod_real3;
#else
typedef float tshift_pmod_real;
typedef float3 tshift_pmod_real3;
#endif

#if SPLINE_DOUBLE
typedef double spline_real;
typedef double2 spline_complex;
#else
typedef float spline_real;
typedef float2 spline_complex;
#endif

#if PHASE_MOD_DOUBLE
typedef double phase_mod_real;
typedef double2 phase_mod_complex;
#else
typedef float phase_mod_real;
typedef float2 phase_mod_complex;
#endif

#if INTERIM_FSTAT_DOUBLE
typedef double interim_fstat_real;
typedef double2 interim_fstat_complex;
#else
typedef float interim_fstat_real;
typedef float2 interim_fstat_complex;
#endif

#endif // __FLOATS_H_CL__
