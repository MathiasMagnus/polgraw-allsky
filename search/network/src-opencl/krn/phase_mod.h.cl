#ifndef __PHASE_MOD_H_CL__
#define __PHASE_MOD_H_CL__

// Polgraw includes
#include <floats.h.cl>       // phase_mod_real, phase_mod_complex

/// <summary>The purpose of this function was undocumented.</summary>
///
kernel void phase_mod_1(global fft_complex* xa,
                        global fft_complex* xb,
                        global xDatm_complex* xar,
                        global xDatm_complex* xbr,
                        phase_mod_real het1,
                        phase_mod_real sgnlt1,
                        global shift_real* shft,
                        int N);

/// <summary>The purpose of this function was undocumented.</summary>
///
kernel void phase_mod_2(global fft_complex* xa,
                        global fft_complex* xb,
                        global xDatm_complex* xar,
                        global xDatm_complex* xbr,
                        phase_mod_real het1,
                        phase_mod_real sgnlt1,
                        global shift_real* shft,
                        int N);

#endif // __PHASE_MOD_H_CL__
