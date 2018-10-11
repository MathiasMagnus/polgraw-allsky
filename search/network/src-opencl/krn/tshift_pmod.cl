#include <tshift_pmod.h.cl>

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
                        global shift_real* tshift)
{
  size_t i = get_global_id(0);

  if (i < N)
  {
    tshift_pmod_real S = ns.x * DetSSB[i].x
                       + ns.y * DetSSB[i].y
                       + ns.z * DetSSB[i].z,
                     phase = -het0 * i - oms * S,
                     c = cos(phase),
                     s = sin(phase);

    xa[i].x = xDat[i] * aa[i] * c;
    xa[i].y = xDat[i] * aa[i] * s;
    xb[i].x = xDat[i] * bb[i] * c;
    xb[i].y = xDat[i] * bb[i] * s;

    shft[i] = S;
    shftf[i] = S - shft1;
    tshift[i] = interpftpad * (i - (S - shft1));
  }
  else if (i < nfft)
  {
      xa[i].x = xa[i].y = xb[i].x = xb[i].y = 0.;
  }
}
