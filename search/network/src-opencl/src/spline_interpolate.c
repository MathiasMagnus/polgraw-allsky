#include <spline_interpolate.h>

#ifdef _WIN32
#define COMP_DOUBLE SPLINE_DOUBLE
#include <complex_op.h>
#endif


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
    clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
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
    clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
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
    clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
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
    clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
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
    clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
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
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xDatma_d, N, idet, "ifo_sig_xDatma");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xDatmb_d, N, idet, "ifo_sig_xDatmb");
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
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xDatma_d, N, idet, "rescaled_ifo_sig_xDatma");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xDatmb_d, N, idet, "rescaled_ifo_sig_xDatmb");
#endif
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
      spline_real x = interpftpad * (i - shftf[i]);
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
        complex_t invp = 2. / (y2[i - 1] + 4.);
        y2[i] = -.5*invp;
        u[i] = y[i - 1] - 2.*y[i] + y[i + 1];
        u[i] = (-.5*u[i - 1] + 3.*u[i])*invp;
    }
    complex_t qn = 0,
              un = 0;
    y2[n - 1] = (un - qn * u[n - 2]) / (qn*y2[n - 2] + 1.);
    for (int k = n - 2; k >= 0; --k)
        y2[k] = y2[k] * y2[k + 1] + u[k];
#else
    y2[0] = u[0] = cbuild(0, 0);

    for (int i = 1; i < n - 1; ++i) {
        spline_complex invp = cdivrc(2., caddcr(y2[i - 1], 4.));
        y2[i] = cmulrc(-.5, invp);
        u[i] = caddcc(caddcc(y[i - 1], cmulrc(-2., y[i])), y[i + 1]);
        u[i] = cmulcc(caddcc(cmulrc(-.5, u[i - 1]), cmulrc(3., u[i])), invp);
    }
    spline_complex qn = cbuild(0, 0),
                   un = cbuild(0, 0);
    y2[n - 1] = cdivcc(csubcc(un, cmulcc(qn, u[n - 2])), caddcr(cmulcc(qn, y2[n - 2]), 1.));
    for (int k = n - 2; k >= 0; --k)
        y2[k] = caddcc(cmulcc(y2[k], y2[k + 1]), u[k]);
#endif
}

xDatm_complex splint(fft_complex *ya,
                     fft_complex *y2a,
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
    int klo, khi;
    spline_real b, a;

    if (x<0 || x>n - 1)
        return cbuild(0, 0);
    klo = (int)floor(x); // Explicit cast silences warning C4244: '=': conversion from 'double' to 'int', possible loss of data
    khi = klo + 1;
    a = khi - x;
    b = x - klo;
    return caddcc(caddcc(cmulrc(a, ya[klo]), cmulrc(b, ya[khi])), cdivcr(caddcc(cmulrc(a*a*a - a, y2a[klo]), cmulrc(b*b*b - b, y2a[khi])), 6.0));
#endif // _WIN32
}
