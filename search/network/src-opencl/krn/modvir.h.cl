#ifndef __MODVIR_H_CL__
#define __MODVIR_H_CL__

// Polgraw includes
#include <floats.h.cl>       // modvir_real

/// <summary>Amplitude modulation function coefficients</summary>
///
typedef struct _ampl_mod_coeff
{
    ampl_mod_real c1, c2, c3, c4, c5, c6, c7, c8, c9;

} Ampl_mod_coeff;

/// <summary>Calculate the amplitude modulation functions of a given detector.</summary>
///
kernel void modvir(const int idet,
                   const int Np,
                   const modvir_real sinalfr,
                   const modvir_real cosalfr,
                   const modvir_real sindel,
                   const modvir_real cosdel,
                   const modvir_real c2d,
                   const modvir_real c2sd,
                   const modvir_real omr,
                   constant Ampl_mod_coeff* amod,
                   global ampl_mod_real* aa,
                   global ampl_mod_real* bb);

#endif // __MODVIR_H_CL__
