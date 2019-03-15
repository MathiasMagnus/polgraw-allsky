// C behavioral defines
//
// MSVC: macro to include constants, such as M_PI (include before math.h)
#define _USE_MATH_DEFINES

#include <sky_positions.h>

// Standard C includes
#include <math.h>       // M_PI, asin, fmod, atan2

#define sqr(x) ((x)*(x))

bool sky_positions(const int pm,                  // hemisphere
                   const int mm,                  // grid 'sky position'
                   const int nn,                  // other grid 'sky position'
                   double* M,                     // M matrix from grid point to linear coord
                   double oms,
                   double sepsm,
                   double cepsm,
                   signal_params_t* sgnlt,
                   double* het0,
                   double* sinalt,
                   double* cosalt,
                   double* sindelt,
                   double* cosdelt)
{
    /* Matrix M(.,.) (defined on page 22 of PolGrawCWAllSkyReview1.pdf file)
    defines the transformation form integers (bin, ss, nn, mm) determining
    a grid point to linear coordinates omega, omegadot, alpha_1, alpha_2),
    where bin is the frequency bin number and alpha_1 and alpha_2 are
    defined on p. 22 of PolGrawCWAllSkyReview1.pdf file.

    [omega]                          [bin]
    [omegadot]       = M(.,.) \times [ss]
    [alpha_1/omega]                  [nn]
    [alpha_2/omega]                  [mm]

    Array M[.] is related to matrix M(.,.) in the following way;

    [ M[0] M[4] M[8]  M[12] ]
    M(.,.) =   [ M[1] M[5] M[9]  M[13] ]
    [ M[2] M[6] M[10] M[14] ]
    [ M[3] M[7] M[11] M[15] ]

    and

    M[1] = M[2] = M[3] = M[6] = M[7] = 0
    */

    // Grid positions
    double al1 = nn * M[10] + mm * M[14],
           al2 = nn * M[11] + mm * M[15];

    // check if the search is in an appropriate region of the grid
    // if not, returns false
    if ((sqr(al1) + sqr(al2)) / sqr(oms) > 1.) return false;
    else
    {
      // Change linear (grid) coordinates to real coordinates
      lin2ast(al1 / oms, al2 / oms, pm, sepsm, cepsm, // input
              sinalt, cosalt, sindelt, cosdelt);      // output

      // calculate declination and right ascention
      // written in file as candidate signal sky positions
      sgnlt[declination] = asin(*sindelt);
      sgnlt[ascension] = fmod(atan2(*sinalt, *cosalt) + 2. * M_PI, 2. * M_PI);

      *het0 = fmod(nn * M[8] + mm * M[12], M[0]);

      return true;
    }
}

void lin2ast(const double be1,
             const double be2,
             const int pm,
             const double sepsm,
             const double cepsm,
             double *sinal,
             double *cosal,
             double *sindel,
             double *cosdel)
{
  *sindel = be1 * sepsm - (2 * pm - 3)*sqrt(1. - sqr(be1) - sqr(be2))*cepsm;
  *cosdel = sqrt(1. - sqr(*sindel));
  *sinal = (be1 - sepsm * (*sindel)) / (cepsm*(*cosdel));
  *cosal = be2 / (*cosdel);
}
