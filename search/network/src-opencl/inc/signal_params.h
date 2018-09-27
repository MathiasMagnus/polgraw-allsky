#pragma once

#define NPAR 5      // no. of trigger parameters //

typedef double signal_params_t;

enum sgnlt_e
{
    frequency = 0,
    spindown = 1,
    declination = 2,
    ascension = 3,
    signal_to_noise = 4
};
