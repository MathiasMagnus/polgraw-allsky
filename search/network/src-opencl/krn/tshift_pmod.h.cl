#ifndef __TSHIFT_PMOD_H_CL__
#define __TSHIFT_PMOD_H_CL__

// Polgraw includes
#include <floats.h.cl>       // tshift_pmod_real, tshift_pmod_real3

/// <summary>The purpose of this function was undocumented.</summary>
///
kernel void tshift_pmod(int N,
                        int nfft,
                        int interpftpad,
                        tshift_pmod_real shft1,
                        tshift_pmod_real het0,
                        tshift_pmod_real oms,
                        tshift_pmod_real3 ns,
                        global const xDat_real* xDat,
                        global const ampl_mod_real* aa,
                        global const ampl_mod_real* bb,
                        global const DetSSB_real3* DetSSB,
                        global fft_complex* xa,
                        global fft_complex* xb,
                        global shift_real* shft,
                        global shift_real* shftf,
                        global shift_real* tshift);

#endif // __TSHIFT_PMOD_H_CL__
