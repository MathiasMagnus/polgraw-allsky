#pragma once

// Polgraw includes
#include <signal_params.h>  // signal_params_t
#include <stdbool.h>        // bool

/// <summary>Calculates sky-position dependant quantities.</summary>
/// <returns>Returns true if the sky coordinate is valid, false otherwise.</returns>
///
bool sky_positions(const int pm,                  // hemisphere
                   const int mm,                  // grid 'sky position'
                   const int nn,                  // other grid 'sky position'
                   double* M,                     // M matrix from grid point to linear coord
                   double oms,
                   double sepsm,
                   double cepsm,
                   signal_params_t* sgnlt,
                   double* het0,
                   double* ft,
                   double* sinalt,
                   double* cosalt,
                   double* sindelt,
                   double* cosdelt);

/// <summary>Change linear (grid) coordinates to real coordinates</summary>
/// <remarks>lin2ast described in Phys. Rev. D 82, 022005 (2010) (arXiv:1003.0844)</remarks>
///
void lin2ast(const double be1,
             const double be2,
             const int pm,
             const double sepsm,
             const double cepsm,
             double *sinal,
             double *cosal,
             double *sindel,
             double *cosdel);
