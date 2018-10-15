#include <fft_interpolate.h>

// Polgraw includes
#include <precision.h>  // fft_complex
#include <CL/util.h>    // checkErr


void fft_interpolate(const cl_int idet,
                     const cl_int id,
                     const cl_int nfft,
                     const cl_int Ninterp,
                     const cl_int nyqst,
                     const FFT_plans* plans,
                     cl_mem xa_d,
                     cl_mem xb_d,
                     OpenCL_handles* cl_handles,
                     const cl_uint num_events_in_wait_list,
                     const cl_event* event_wait_list,
                     cl_event* fw_fft_events,
                     cl_event* resample_copy_events,
                     cl_event* resample_fill_events,
                     cl_event* inv_fft_events)
{
  clfftStatus CLFFT_status = CLFFT_SUCCESS;

  // Forward FFT
  CLFFT_status = clfftEnqueueTransform(plans->pl_int[id], CLFFT_FORWARD, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &fw_fft_events[0], &xa_d, NULL, NULL);
  checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");
  CLFFT_status = clfftEnqueueTransform(plans->pl_int[id], CLFFT_FORWARD, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &fw_fft_events[1], &xb_d, NULL, NULL);
  checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");

#ifdef TESTING
  clWaitForEvents(2, fw_fft_events);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, nfft, idet, "xa_fourier");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, nfft, idet, "xb_fourier");
#endif

  // Resample coefficients
  resample_postfft(idet, id, nfft, Ninterp, nyqst,    // input
    xa_d, xb_d,                                       // input / output
    cl_handles, 2, fw_fft_events,                     // sync
    resample_copy_events,                             // sync
    resample_fill_events);                            // sync

  // Backward fft (len Ninterp = nfft*interpftpad)
  clfftEnqueueTransform(plans->pl_inv[id], CLFFT_BACKWARD, 1, &cl_handles->exec_queues[id][idet], 1, resample_fill_events, &inv_fft_events[0], &xa_d, NULL, NULL);
  checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_BACKWARD)");
  clfftEnqueueTransform(plans->pl_inv[id], CLFFT_BACKWARD, 1, &cl_handles->exec_queues[id][idet], 1, resample_fill_events, &inv_fft_events[1], &xb_d, NULL, NULL);
  checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_BACKWARD)");

  // scale fft with clblas not needed (as opposed fftw), clFFT already scales

#ifdef TESTING
  clWaitForEvents(2, inv_fft_events);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, Ninterp, idet, "xa_time_resampled");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, Ninterp, idet, "xb_time_resampled");
#endif
}

void resample_postfft(const cl_int idet,
                      const cl_int id,
                      const cl_int nfft,
                      const cl_int Ninterp,
                      const cl_int nyqst,
                      cl_mem xa_d,
                      cl_mem xb_d,
                      OpenCL_handles* cl_handles,
                      const cl_uint num_events_in_wait_list,
                      const cl_event* event_wait_list,
                      cl_event* copy_events,
                      cl_event* fill_events)
{
  cl_int CL_err = CL_SUCCESS;

  //clSetKernelArg(cl_handles->kernels[id][ResamplePostFFT], 0, sizeof(cl_mem), &xa_d);    checkErr(CL_err, "clSetKernelArg(&xa_d)");
  //clSetKernelArg(cl_handles->kernels[id][ResamplePostFFT], 1, sizeof(cl_mem), &xb_d);    checkErr(CL_err, "clSetKernelArg(&xb_d)");
  //clSetKernelArg(cl_handles->kernels[id][ResamplePostFFT], 2, sizeof(cl_int), &nfft);    checkErr(CL_err, "clSetKernelArg(&nfft)");
  //clSetKernelArg(cl_handles->kernels[id][ResamplePostFFT], 3, sizeof(cl_int), &Ninterp); checkErr(CL_err, "clSetKernelArg(&Ninterp)");
  //clSetKernelArg(cl_handles->kernels[id][ResamplePostFFT], 4, sizeof(cl_int), &nyqst);   checkErr(CL_err, "clSetKernelArg(&nyqst)");
  //
  //cl_event exec;
  //size_t resample_length = (size_t)Ninterp - (nyqst + nfft); // Helper variable to make pointer types match. Cast to silence warning
  //
  //CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][ResamplePostFFT], 1, NULL, &resample_length, NULL, 0, NULL, &exec);
  //checkErr(CL_err, "clEnqueueNDRangeKernel(cl_handles->kernels[ResamplePostFFT])");

  size_t src_off = sizeof(fft_complex) * nyqst,                     // src offset
      dst_off = sizeof(fft_complex) * (nyqst + Ninterp - nfft),  // dst offset
      size = sizeof(fft_complex) * Ninterp - (nyqst + nfft),     // size
      end = (dst_off + size) / sizeof(fft_complex);
  cl_long negative = dst_off - (src_off + size);

  CL_err = clEnqueueCopyBuffer(cl_handles->read_queues[id][idet],               // queue
                               xa_d, xa_d,                                      // src, dst
                               sizeof(fft_complex) * nyqst,                     // src offset
                               sizeof(fft_complex) * (nyqst + Ninterp - nfft),  // dst offset
                               sizeof(fft_complex) * (Ninterp - (nyqst + nfft)),// size
                               1, &event_wait_list[0],                          // wait events
                               &copy_events[0]);                                // out event
  checkErr(CL_err, "clEnqueueCopyBuffer(xa_d, xa_d)");

  CL_err = clEnqueueCopyBuffer(cl_handles->write_queues[id][idet],              // queue
                               xb_d, xb_d,                                      // src, dst
                               sizeof(fft_complex) * nyqst,                     // src offset
                               sizeof(fft_complex) * (nyqst + Ninterp - nfft),  // dst offset
                               sizeof(fft_complex) * (Ninterp - (nyqst + nfft)),// size
                               1, &event_wait_list[1],                          // wait events
                               &copy_events[1]);                                // out event
  checkErr(CL_err, "clEnqueueCopyBuffer(xb_d, xb_d)");

#ifdef _WIN32
  fft_complex pattern = { 0, 0 };
#else
  fft_complex pattern = 0;
#endif

  CL_err = clEnqueueFillBuffer(cl_handles->read_queues[id][idet],               // queue
                               xa_d,                                            // buffer
                               &pattern, sizeof(fft_complex),                   // patern and size
                               sizeof(fft_complex) * nyqst,                     // offset
                               sizeof(fft_complex) * (Ninterp - (nyqst + nfft)),// size
                               1, &copy_events[0],                              // wait events
                               &fill_events[0]);                                // out event
  checkErr(CL_err, "clEnqueueFillBuffer(xa_d, { 0, 0 })");
  
  CL_err = clEnqueueFillBuffer(cl_handles->write_queues[id][idet],              // queue
                               xb_d,                                            // buffer
                               &pattern, sizeof(fft_complex),                   // patern and size
                               sizeof(fft_complex) * nyqst,                     // offset
                               sizeof(fft_complex) * (Ninterp - (nyqst + nfft)),// size
                               1, &copy_events[1],                              // wait events
                               &fill_events[1]);                                // out event
  checkErr(CL_err, "clEnqueueFillBuffer(xb_d, { 0, 0 })");

#ifdef TESTING
  clWaitForEvents(2, fill_events);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, Ninterp, idet, "xa_fourier_resampled");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, Ninterp, idet, "xb_fourier_resampled");
#endif
}