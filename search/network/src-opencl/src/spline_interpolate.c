#include <spline_interpolate.h>

#ifdef _MSC_VER
#include <complex_op.h>
#endif

// Polgraw includes
#include <CL/util.h>    // checkErr
#include <log.h>        // save_numbered_complex_buffer

// Standard C includes
#include <math.h>       // floor(f)
#include <stdlib.h>     // mallloc, free

void spline_interpolate(const cl_int idet,
                        const cl_int id,
                        const size_t arr_len,
                        const size_t N,
                        const int interpftpad,
                        const double sig2,
                        const cl_mem xa_d,
                        const cl_mem xb_d,
                        const cl_mem shftf_d,
                        cl_mem xDatma_d,
                        cl_mem xDatmb_d,
                        BLAS_handles* blas_handles,
                        OpenCL_handles* cl_handles,
                        const cl_uint num_events_in_wait_list,
                        const cl_event* event_wait_list,
                        cl_event* spline_map_events,
                        cl_event* spline_unmap_events,
                        cl_event* spline_blas_events)
{
  cl_int CL_err = CL_SUCCESS;

  fft_complex* xa =
    (fft_complex*)clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
                                     xa_d,
                                     CL_FALSE,
                                     CL_MAP_READ,
                                     0,
                                     arr_len * sizeof(fft_complex),
                                     num_events_in_wait_list,
                                     event_wait_list,
                                     &spline_map_events[0],
                                     &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(fft_arr->xa_d)");

  fft_complex* xb =
    (fft_complex*)clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
                                     xb_d,
                                     CL_FALSE,
                                     CL_MAP_READ,
                                     0,
                                     arr_len * sizeof(fft_complex),
                                     num_events_in_wait_list,
                                     event_wait_list,
                                     &spline_map_events[1],
                                     &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(fft_arr->xb_d)");

  shift_real* shftf =
    (shift_real*)clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
                                    shftf_d,
                                    CL_FALSE,
                                    CL_MAP_READ,
                                    0,
                                    N * sizeof(shift_real),
                                    num_events_in_wait_list,
                                    event_wait_list,
                                    &spline_map_events[2],
                                    &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(ifo[n].sig.shftf_d)");

  xDatm_complex* xDatma =
    (xDatm_complex*)clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
                                       xDatma_d,
                                       CL_FALSE,
                                       CL_MAP_WRITE,
                                       0,
                                       N * sizeof(xDatm_complex),
                                       num_events_in_wait_list,
                                       event_wait_list,
                                       &spline_map_events[3],
                                       &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(ifo[n].sig.xDatma_d)");

  xDatm_complex* xDatmb =
    (xDatm_complex*)clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
                                       xDatmb_d,
                                       CL_FALSE,
                                       CL_MAP_WRITE,
                                       0,
                                       N * sizeof(xDatm_complex),
                                       num_events_in_wait_list,
                                       event_wait_list,
                                       &spline_map_events[4],
                                       &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(ifo[n].sig.xDatmb_d)");

  // Wait for maps and do actual interpolation
  clWaitForEvents(5, spline_map_events);
  splintpad(xa, shftf, (int)N, interpftpad, xDatma); // cast silences warn on interface of old/new code
  splintpad(xb, shftf, (int)N, interpftpad, xDatmb); // cast silences warn on interface of old/new code

  CL_err = clEnqueueUnmapMemObject(cl_handles->write_queues[id][idet], xa_d, xa, 0, NULL, &spline_unmap_events[0]);         checkErr(CL_err, "clEnqueueUnMapMemObject(xa_d)");
  CL_err = clEnqueueUnmapMemObject(cl_handles->write_queues[id][idet], xb_d, xb, 0, NULL, &spline_unmap_events[1]);         checkErr(CL_err, "clEnqueueUnMapMemObject(xb_d)");
  CL_err = clEnqueueUnmapMemObject(cl_handles->write_queues[id][idet], shftf_d, shftf, 0, NULL, &spline_unmap_events[2]);   checkErr(CL_err, "clEnqueueUnMapMemObject(shftf_d)");
  CL_err = clEnqueueUnmapMemObject(cl_handles->write_queues[id][idet], xDatma_d, xDatma, 0, NULL, &spline_unmap_events[3]); checkErr(CL_err, "clEnqueueUnMapMemObject(xDatma_d)");
  CL_err = clEnqueueUnmapMemObject(cl_handles->write_queues[id][idet], xDatmb_d, xDatmb, 0, NULL, &spline_unmap_events[4]); checkErr(CL_err, "clEnqueueUnMapMemObject(xDatmb_d)");

#ifdef TESTING
  clWaitForEvents(5, spline_unmap_events);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xDatma_d, N, idet, "ifo_sig_xDatma", XDATM_DOUBLE);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xDatmb_d, N, idet, "ifo_sig_xDatmb", XDATM_DOUBLE);
#endif

  blas_scale(idet, id, N, 1. / sig2,
             xDatma_d,
             xDatmb_d,
             blas_handles,
             cl_handles,
             2, spline_unmap_events + 3, // it is enough to wait on the last 2 of the unmaps: xDatma, xDatmb
             spline_blas_events);
#ifdef TESTING
  clWaitForEvents(2, spline_blas_events);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xDatma_d, N, idet, "rescaled_ifo_sig_xDatma", XDATM_DOUBLE);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xDatmb_d, N, idet, "rescaled_ifo_sig_xDatmb", XDATM_DOUBLE);
#endif
}

void blas_scale(const cl_int idet,
                const cl_int id,
                const size_t n,
                const double a,
                cl_mem a_d,
                cl_mem b_d,
                BLAS_handles* blas_handles,
                OpenCL_handles* cl_handles,
                const cl_uint num_events_in_wait_list,
                const cl_event* event_wait_list,
                cl_event* blas_exec)
{
  clblasStatus status[2];
  (void*)blas_handles; // Currently not needed, but maybe later

#if XDATM_DOUBLE
  status[0] = clblasDscal(n * 2, a, a_d, 0, 1, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[0]); checkErrBLAS(status[0], "clblasDscal(xa_d)");
  status[1] = clblasDscal(n * 2, a, b_d, 0, 1, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[1]); checkErrBLAS(status[1], "clblasDscal(xb_d)");
#else
  status[0] = clblasSscal(n * 2, (float)a, a_d, 0, 1, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[0]); checkErrBLAS(status[0], "clblasSscal(xa_d)");
  status[1] = clblasSscal(n * 2, (float)a, b_d, 0, 1, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[1]); checkErrBLAS(status[1], "clblasSscal(xb_d)");
#endif // COMP_FLOAT
}

void splintpad(fft_complex* ya,
               shift_real* shftf,
               int N,
               int interpftpad,
               xDatm_complex* out)
{
  /* Cubic spline with "natural" boundary conditions.
  Input:
  ya[i] - value of the function being interpolated in x_i = i,
  for i = 0 .. (interpftpad*N-1)	(changed on exit);
  Interpolating spline will be calculated at the points
  interpftpad*(i-shftf[i]), for i = 0 .. (N-1);
  N - number of output data points.
  Output:
  out[i] - value of the interpolating function
  at interpftpad*(i-shftf[i]).
  */
  spline_complex *y2 = (spline_complex*)malloc(interpftpad*N * sizeof(spline_complex)), //vector twice-size of N
                 *u = (spline_complex*)malloc(interpftpad*N * sizeof(spline_complex));

  spline(ya, interpftpad*N, y2, u);
  for (int i = 0; i<N; ++i) {
    spline_real x = interpftpad * (i - (spline_real)(shftf[i]));
    out[i] = splint(ya, y2, interpftpad*N, x);
  } /* for i */

  free(y2);
  free(u);
}

void spline(const fft_complex* y,
            int n,
            spline_complex* y2,
            spline_complex* u)
{
#ifndef _WIN32
  y2[0] = u[0] = 0.;

  for (int i = 1; i < n - 1; ++i) {
    //p = .5*y2[i-1]+2.;
    //y2[i] = -.5/p;
    //u[i] = y[i+1]-2.*y[i]+y[i-1];
    //u[i] = (3.*u[i]-.5*u[i-1])/p;
    spline_complex invp = 2. / (y2[i - 1] + 4.);
    y2[i] = -.5*invp;
    u[i] = y[i - 1] - 2.*y[i] + y[i + 1];
    u[i] = (-.5*u[i - 1] + 3.*u[i])*invp;
  }
  spline_complex qn = 0,
            un = 0;
  y2[n - 1] = (un - qn * u[n - 2]) / (qn*y2[n - 2] + 1.);
  for (int k = n - 2; k >= 0; --k)
    y2[k] = y2[k] * y2[k + 1] + u[k];
#else
#if SPLINE_DOUBLE
  y2[0] = u[0] = cbuild(0, 0);

  for (int i = 1; i < n - 1; ++i) {
#if FFT_DOUBLE
    spline_complex yim1 = y[i - 1],
                   yi   = y[i],
                   yip1 = y[i + 1];
#else
    spline_complex yim1 = cbuild(crealf(y[i - 1]), cimagf(y[i - 1])),
                   yi   = cbuild(crealf(y[i]), cimagf(y[i])),
                   yip1 = cbuild(crealf(y[i + 1]), cimagf(y[i + 1]));
#endif
    spline_complex invp = cdivrc(2., caddcr(y2[i - 1], 4.));
    y2[i] = cmulrc(-.5, invp);
    u[i] = caddcc(caddcc(yim1, cmulrc(-2., yi)), yip1);
    u[i] = cmulcc(caddcc(cmulrc(-.5, u[i - 1]), cmulrc(3., u[i])), invp);
  }
  spline_complex qn = cbuild(0, 0),
                 un = cbuild(0, 0);
  y2[n - 1] = cdivcc(csubcc(un, cmulcc(qn, u[n - 2])), caddcr(cmulcc(qn, y2[n - 2]), 1.));
  for (int k = n - 2; k >= 0; --k)
    y2[k] = caddcc(cmulcc(y2[k], y2[k + 1]), u[k]);
#else
  y2[0] = u[0] = fcbuild(0, 0);

  for (int i = 1; i < n - 1; ++i) {
#if FFT_DOUBLE
    spline_complex yim1 = fcbuild((spline_real)creal(y[i - 1]), cimag((spline_real)y[i - 1])),
                   yi   = fcbuild((spline_real)creal(y[i]), (spline_real)cimag(y[i])),
                   yip1 = fcbuild(y[i + 1]);
#else
    spline_complex yim1 = y[i - 1],
                   yi   = y[i],
                   yip1 = y[i + 1];
#endif
    spline_complex invp = fcdivrc(2., fcaddcr(y2[i - 1], 4.));
    y2[i] = fcmulrc(-.5, invp);
    u[i] = fcaddcc(fcaddcc(y[i - 1], fcmulrc(-2., y[i])), y[i + 1]);
    u[i] = fcmulcc(fcaddcc(fcmulrc(-.5, u[i - 1]), fcmulrc(3., u[i])), invp);
  }
  spline_complex qn = fcbuild(0, 0),
                 un = fcbuild(0, 0);
  y2[n - 1] = fcdivcc(fcsubcc(un, fcmulcc(qn, u[n - 2])), fcaddcr(fcmulcc(qn, y2[n - 2]), 1.));
  for (int k = n - 2; k >= 0; --k)
    y2[k] = fcaddcc(fcmulcc(y2[k], y2[k + 1]), u[k]);
#endif
#endif
}

xDatm_complex splint(fft_complex *ya,
                     spline_complex *y2a,
                     int n,
                     spline_real x)
{
#ifndef _WIN32
  int klo, khi;
  spline_real b, a;

  if (x<0 || x>n - 1)
    return 0.;
  klo = floor(x);
  khi = klo + 1;
  a = khi - x;
  b = x - klo;
  return a * ya[klo] + b * ya[khi] + ((a*a*a - a)*y2a[klo] + (b*b*b - b)*y2a[khi]) / 6.0;
#else
#if SPLINE_DOUBLE
  int klo, khi;
  spline_real b, a;

  if (x<0 || x>n - 1)
#if XDATM_DOUBLE
    return cbuild(0, 0);
#else
    return fcbuild(0, 0);
#endif
  klo = (int)floor(x); // Explicit cast silences warning C4244: '=': conversion from 'double' to 'int', possible loss of data
  khi = klo + 1;
  a = khi - x;
  b = x - klo;
#if FFT_DOUBLE
  spline_complex ya_klo = ya[klo],
                 ya_khi = ya[khi];
#else
  spline_complex ya_klo = cbuild(crealf(ya[klo]), cimagf(ya[klo])),
                 ya_khi = cbuild(crealf(ya[khi]), cimagf(ya[khi]));
#endif
  spline_complex y2a_klo = y2a[klo],
                 y2a_khi = y2a[khi],
                 result = caddcc(caddcc(cmulrc(a, ya_klo), cmulrc(b, ya_khi)), cdivcr(caddcc(cmulrc(a*a*a - a, y2a_klo), cmulrc(b*b*b - b, y2a_khi)), 6.0));
#if XDATM_DOUBLE
  return result;
#else
  return fcbuild((spline_real)creal(result), (spline_real)cimag(result));
#endif
#else
  int klo, khi;
  spline_real b, a;

  if (x<0 || x>n - 1)
#if XDATM_DOUBLE
      return cbuild(0, 0);
#else
      return fcbuild(0, 0);
#endif
  klo = (int)floorf(x); // Explicit cast silences warning C4244: '=': conversion from 'double' to 'int', possible loss of data
  khi = klo + 1;
  a = khi - x;
  b = x - klo;
#if FFT_DOUBLE
  spline_complex ya_klo = fcbuild((spline_real)creal(ya[klo]), (spline_real)cimag(ya[klo])),
                 ya_khi = fcbuild((spline_real)creal(ya[khi]), (spline_real)cimag(ya[khi]));
#else
  spline_complex ya_klo = ya[klo],
                 ya_khi = ya[khi];
#endif
  spline_complex y2a_klo = y2a[klo],
                 y2a_khi = y2a[khi],
                 result = (fcaddcc(fcmulrc(a, ya_klo), fcmulrc(b, ya_khi)), fcdivcr(fcaddcc(fcmulrc(a*a*a - a, y2a_klo), fcmulrc(b*b*b - b, y2a_khi)), 6.0));
#if XDATM_DOUBLE
  return cbuild(fcreal(result), fcimag(result));
#else
  return result;
#endif
#endif
#endif // _WIN32
}
