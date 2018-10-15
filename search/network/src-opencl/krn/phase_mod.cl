#include <phase_mod.h.cl>

#include <complex_op.h.cl>


/// <summary>The purpose of this function was undocumented.</summary>
///
kernel void phase_mod_1(global fft_complex* xa,
                        global fft_complex* xb,
                        global xDatm_complex* xar,
                        global xDatm_complex* xbr,
                        phase_mod_real het1,
                        phase_mod_real sgnlt1,
                        global shift_real* shft,
                        int N)
{
  size_t idx = get_global_id(0);

  if (idx < N)
  {
    //         _tmp1[0][i]
    //                |     aux->t2[i]
    //                |         |
    //                |         |        (double)(2*i)*ifo[n].sig.shft[i]
    //                |         |               |
    phase_mod_real tmp10i = idx * idx + 2 * idx * (phase_mod_real)(shft[idx]);
    fft_complex xDatma = xar[idx];
    fft_complex xDatmb = xbr[idx];

#if PHASE_MOD_DOUBLE
    phase_mod_real phase = idx * het1 + sgnlt1 * tmp10i;
    phase_mod_complex exph = cbuild(cos(phase), -sin(phase));

    xa[idx] = cmulcc(cbuild(xDatma.x, xDatma.y), exph);
    xb[idx] = cmulcc(cbuild(xDatmb.x, xDatmb.y), exph);
#else
    phase_mod_real phase = idx * het1 + sgnlt1 * tmp10i;
    phase_mod_complex exph = fcbuild(cos(phase), -sin(phase));

    xa[idx] = fcmulcc(fcbuild(xDatma.x, xDatma.y), exph);
    xb[idx] = fcmulcc(fcbuild(xDatmb.x, xDatmb.y), exph);
#endif
  }
}

/// <summary>The purpose of this function was undocumented.</summary>
///
kernel void phase_mod_2(global fft_complex* xa,
                        global fft_complex* xb,
                        global xDatm_complex* xar,
                        global xDatm_complex* xbr,
                        phase_mod_real het1,
                        phase_mod_real sgnlt1,
                        global shift_real* shft,
                        int N)
{
  size_t idx = get_global_id(0);

  if (idx < N)
  {
    //         _tmp1[0][i]
    //                |     aux->t2[i]
    //                |         |
    //                |         |        (double)(2*i)*ifo[n].sig.shft[i]
    //                |         |               |
    phase_mod_real tmp10i = idx * idx + 2 * idx * (phase_mod_real)(shft[idx]);
    fft_complex xDatma = xar[idx];
    fft_complex xDatmb = xbr[idx];

#if PHASE_MOD_DOUBLE
    phase_mod_real phase = idx * het1 + sgnlt1 * tmp10i;
    phase_mod_complex exph = cbuild(cos(phase), -sin(phase));

    xa[idx] = cmulcc(cbuild(xDatma.x, xDatma.y), exph);
    xb[idx] = cmulcc(cbuild(xDatmb.x, xDatmb.y), exph);
#else
    phase_mod_real phase = idx * het1 + sgnlt1 * tmp10i;
    phase_mod_complex exph = fcbuild(cos(phase), -sin(phase));

    xa[idx] = fcmulcc(fcbuild(xDatma.x, xDatma.y), exph);
    xb[idx] = fcmulcc(fcbuild(xDatmb.x, xDatmb.y), exph);
#endif
  }
}
