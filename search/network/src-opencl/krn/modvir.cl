#include <modvir.h.cl>

kernel void modvir(const int idet,
                   const int Np,
                   const modvir_real sinalfr,
                   const modvir_real cosalfr,
                   const modvir_real sindel,
                   const modvir_real cosdel,
                   const modvir_real c2d,
                   const modvir_real c2sd,
                   const modvir_real omr,
                   global Ampl_mod_coeff* amod,
                   global ampl_mod_real* aa,
                   global ampl_mod_real* bb)
{
    size_t idx = get_global_id(0);

    modvir_real cosmod = cos(omr * idx),
                sinmod = sin(omr * idx),
                c = cosalfr * cosmod + sinalfr * sinmod,
                s = sinalfr * cosmod - cosalfr * sinmod,
                c2s = 2.*c*c,
                cs = c * s;

    aa[idx] =
        amod[idet].c1*(2. - c2d)*c2s +
        amod[idet].c2*(2. - c2d)*2.*cs +
        amod[idet].c3*c2sd*c +
        amod[idet].c4*c2sd*s -
        amod[idet].c1*(2. - c2d) +
        amod[idet].c5*c2d;

    bb[idx] =
        amod[idet].c6*sindel*c2s +
        amod[idet].c7*sindel*2.*cs +
        amod[idet].c8*cosdel*c +
        amod[idet].c9*cosdel*s -
        amod[idet].c6*sindel;
}
