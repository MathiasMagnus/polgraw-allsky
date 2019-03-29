#include <fstat.h.cl>

#include <complex_op.h.cl>

/// <summary>Compute F-statistics.</summary>
/// 
kernel void compute_Fstat(global fft_complex* xa,
                          global fft_complex* xb,
                          global fstat_real* F,
                          global ampl_mod_real* maa_d,
                          global ampl_mod_real* mbb_d,
                          int N)
{
  size_t i = get_global_id(0);

  fft_complex xa_in = xa[i],
              xb_in = xb[i];
#if INTERIM_FSTAT_DOUBLE
#if FFT_DOUBLE
  interim_fstat_complex xai = xa_in,
                        xbi = xb_in;
#else
  interim_fstat_complex xai = cbuild(fcreal(xa_in), fcimag(xa_in)),
                        xbi = cbuild(fcreal(xb_in), fcimag(xb_in));
#endif
#else
#if FFT_DOUBLE
  interim_fstat_complex xai = fcbuild((interim_fstat_real)creal(xa_in), (interim_fstat_real)cimag(xa_in)),
                        xbi = fcbuild((interim_fstat_real)creal(xb_in), (interim_fstat_real)cimag(xb_in));
#else
  interim_fstat_complex xai = xa_in,
                        xbi = xb_in;
#endif
#endif
  interim_fstat_real maa = maa_d[0],
                     mbb = mbb_d[0];

#if INTERIM_FSTAT_DOUBLE
  F[i] = (fstat_real)((pown(creal(xai), 2) + pown(cimag(xai), 2)) / maa +
                      (pown(creal(xbi), 2) + pown(cimag(xbi), 2)) / mbb);
#else
  F[i] = (fstat_real)((pown(fcreal(xai), 2) + pown(fcimag(xai), 2)) / maa +
                      (pown(fcreal(xbi), 2) + pown(fcimag(xbi), 2)) / mbb);
#endif
}

/// <summary>Compute F-statistics.</summary>
/// <precondition>lsi less than or equal to nav</precondition>
/// <precondition>lsi be a divisor of nav</precondition>
/// <precondition>lsi be an integer power of 2</precondition>
/// <precondition>nav be a divisor of gsi</precondition>
/// 
kernel void normalize_Fstat_wg_reduce(global fstat_real* F,
                                      local fstat_real* shared,
                                      unsigned int nav)
{
  size_t lid = get_local_id(0),
         lsi = get_local_size(0),
         grp = get_group_id(0),
         off = get_global_offset(0);

  // Load all of nav into local memory
  {
    event_t copy = async_work_group_copy(shared,
                                         (F + off) + grp * nav,
                                         nav,
                                         0);
    wait_group_events(1, &copy);
  }

  // Pass responsible for handling nav potentially having more elements than the work-group has threads
  //
  // ASSERT: lsi <= nav
  // ASSERT: nav % lsi == 0
  if (lsi != nav)
  {
      interim_fstat_real mu = 0;

      for (size_t p = 0; p < nav; p += lsi) mu += shared[p + lid];

      shared[lid] = mu;
  }

  // Divide and conquer inside the work-group
  //
  // ASSERT: lsi is a power of 2
  for (size_t mid = lsi / 2; mid != 0; mid = mid >> 1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < mid) shared[lid] += shared[mid + lid];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  interim_fstat_real mu = shared[0] / (2 * nav);

  // Normalize in global
  //
  // ASSERT: gsi % nav == 0
  for (size_t p = 0; p < nav; p += lsi)
  {
    F[off + grp * nav + p + lid] /= mu;

    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}
