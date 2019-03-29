#ifndef __FSTAT_H_CL__
#define __FSTAT_H_CL__

// Polgraw includes
#include <floats.h.cl>       // fft_complex, 

/// <summary>Compute F-statistics.</summary>
/// 
kernel void compute_Fstat(global fft_complex* xa,
                          global fft_complex* xb,
                          global fstat_real* F,
                          global ampl_mod_real* maa_d,
                          global ampl_mod_real* mbb_d,
                          int N);

/// <summary>Compute F-statistics.</summary>
/// <precondition>lsi less than or equal to nav</precondition>
/// <precondition>lsi be a divisor of nav</precondition>
/// <precondition>lsi be an integer power of 2</precondition>
/// <precondition>nav be a divisor of gsi</precondition>
/// 
kernel void normalize_Fstat_wg_reduce(global fstat_real* F,
                                      local fstat_real* shared,
                                      unsigned int nav);

#endif // __FSTAT_H_CL__
