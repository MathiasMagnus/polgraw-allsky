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
    xDatm_complex xDatma = xar[idx],
                  xDatmb = xbr[idx];

#if PHASE_MOD_DOUBLE
    phase_mod_real phase = idx * het1 + sgnlt1 * tmp10i;
    phase_mod_complex exph = cbuild(cos(phase), -sin(phase));
#if XDATM_DOUBLE
    phase_mod_complex result_a = cmulcc(xDatma, exph),
                      result_b = cmulcc(xDatmb, exph);
#else
    phase_mod_complex result_a = cmulcc(cbuild(fcreal(xDatma), fcimag(xDatma)), exph),
                      result_b = cmulcc(cbuild(fcreal(xDatmb), fcimag(xDatmb)), exph);
#endif
#if FFT_DOUBLE
    xa[idx] = result_a;
    xb[idx] = result_b;
#else
    xa[idx] = fcbuild((phase_mod_real)creal(result_a), (phase_mod_real)cimag(result_a));
    xb[idx] = fcbuild((phase_mod_real)creal(result_b), (phase_mod_real)cimag(result_b));
#endif
#else
    phase_mod_real phase = idx * het1 + sgnlt1 * tmp10i;
    phase_mod_complex exph = fcbuild(cos(phase), -sin(phase));
#if XDATM_DOUBLE
    phase_mod_complex result_a = fcmulcc(fcbuild((phase_mod_real)creal(xDatma), (phase_mod_real)cimag(xDatma)), exph),
                      result_b = fcmulcc(fcbuild((phase_mod_real)creal(xDatmb), (phase_mod_real)cimag(xDatmb)), exph);
#else
    phase_mod_complex result_a = fcmulcc(xDatma, exph),
                      result_b = fcmulcc(xDatmb, exph);
    
#endif
#if FFT_DOUBLE
    xa[idx] = cbuild(fcreal(result_a), fcimag(result_a));
    xb[idx] = cbuild(fcreal(result_b), fcimag(result_b));
#else
    xa[idx] = result_a;
    xb[idx] = result_b;
#endif
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
    xDatm_complex xDatma = xar[idx],
                  xDatmb = xbr[idx];

#if PHASE_MOD_DOUBLE
    phase_mod_real phase = idx * het1 + sgnlt1 * tmp10i;
    phase_mod_complex exph = cbuild(cos(phase), -sin(phase));
#if XDATM_DOUBLE
    phase_mod_complex result_a = cmulcc(xDatma, exph),
                      result_b = cmulcc(xDatmb, exph);
#else
    phase_mod_complex result_a = cmulcc(cbuild(fcreal(xDatma), fcimag(xDatma)), exph),
                      result_b = cmulcc(cbuild(fcreal(xDatmb), fcimag(xDatmb)), exph);
#endif
#if FFT_DOUBLE
    xa[idx] += result_a;
    xb[idx] += result_b;
#else
    xa[idx] += fcbuild((phase_mod_real)creal(result_a), (phase_mod_real)cimag(result_a));
    xb[idx] += fcbuild((phase_mod_real)creal(result_b), (phase_mod_real)cimag(result_b));
#endif
#else
    phase_mod_real phase = idx * het1 + sgnlt1 * tmp10i;
    phase_mod_complex exph = fcbuild(cos(phase), -sin(phase));
#if XDATM_DOUBLE
    phase_mod_complex result_a = fcmulcc(fcbuild((phase_mod_real)creal(xDatma), (phase_mod_real)cimag(xDatma)), exph),
                      result_b = fcmulcc(fcbuild((phase_mod_real)creal(xDatmb), (phase_mod_real)cimag(xDatmb)), exph);
#else
    phase_mod_complex result_a = fcmulcc(xDatma, exph),
                      result_b = fcmulcc(xDatmb, exph);
    
#endif
#if FFT_DOUBLE
    xa[idx] += cbuild(fcreal(result_a), fcimag(result_a));
    xb[idx] += cbuild(fcreal(result_b), fcimag(result_b));
#else
    xa[idx] += result_a;
    xb[idx] += result_b;
#endif
#endif
  }
}
