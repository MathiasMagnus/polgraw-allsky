#include <fft_interpolate.h.cl>


kernel void resample_postfft(global fft_complex *xa,
                             global fft_complex *xb,
                             int nfft,
                             int Ninterp,
                             int nyqst)
{
    size_t idx = get_global_id(0);

	int i = nyqst + Ninterp - nfft + idx;
    int j = nyqst + idx;

    xa[i].x = xa[j].x;
    xa[i].y = xa[j].y;
    xb[i].x = xb[j].x;
    xb[i].y = xb[j].y;

	mem_fence(CLK_GLOBAL_MEM_FENCE);

	xa[j].x = 0;
	xa[j].y = 0;
	xb[j].x = 0;
	xb[j].y = 0;
}
