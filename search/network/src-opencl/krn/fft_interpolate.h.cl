#ifndef __FFT_INTERPOLATE_H_CL__
#define __FFT_INTERPOLATE_H_CL__

// Polgraw includes
#include <floats.h.cl>       // complex_fft


/// <summary>Shifts frequencies and remove those over Nyquist.</summary>
///
kernel void resample_postfft(global complex_fft *xa,
                             global complex_fft *xb,
                             int nfft,
                             int Ninterp,
                             int nyqst);

#endif // __FFT_INTERPOLATE_H_CL__
