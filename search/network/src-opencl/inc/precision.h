#ifndef __PRECISION_H__
#define __PRECISION_H__


// Calculating amplitude modulation coefficients is done in double on host
// Should the results be stored in double?
#define AMPL_MOD_DOUBLE 1

// The input time-domain array is read from disk, which is stored in doubles
// Should the array be stored in double?
#define XDAT_DOUBLE 1

// The detector position for every data point w.r.t Solar System Baricenter is read from disk, which is stored in doubles
// Should the array be stored in double?
#define DETSSB_DOUBLE 1

// Interpolated, resampled time-domain data
// Should the array be stored in double?
#define XDATM_DOUBLE 0

// Time-fourier transform precision
// Should the arrays be stored (and transformed) in double?
#define FFT_DOUBLE 1

// Resampling and time-shift values
// Should the array be stored as double?
#define SHIFT_DOUBLE 0

// F-Statistic values
// Should the array be stored as double?
#define FSTAT_DOUBLE 1

// Intermediate types used during amplification modulation calculation
#define MODVIR_DOUBLE 1

// Intermediate types used during resampling and time-shift calculations
#define TSHIFT_PMOD_DOUBLE 0

// Intermediate types used during spline interpolation
#define SPLINE_DOUBLE 1

// Intermediate types used during phase modulation
#define PHASE_MOD_DOUBLE 1

// Intermediate types used during the calculation of the F-Statistics
#define INTERIM_FSTAT_DOUBLE 1


#if (SPLINE_DOUBLE != FFT_DOUBLE) || \
    (SPLINE_DOUBLE != XDATM_DOUBLE) || \
    (FFT_DOUBLE != XDATM_DOUBLE)

//#error "Current limitation is that FFT, spline interpolation and resampled time-domain data has to be in the same precision."
#endif

#endif // __PRECISION_H__
