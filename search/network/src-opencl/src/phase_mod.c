// OpenCL behavioral defines
//
// 1.2+ OpenCL headers: tells the headers not to bitch about clCreateCommandQueue being renamed to clCreateCommandQueueWithProperties
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
//
// Select API to use
#define CL_TARGET_OPENCL_VERSION 120

#include <phase_mod.h>

// Polgraw includes
#include <CL/util.h>    // checkErr
#include <log.h>        // save_numbered_real_buffer, save_numbered_complex_buffer

typedef phase_mod_real real;

cl_event phase_mod_1(const cl_int idet,
                     const cl_int id,
                     const cl_int N,
                     const double het1,
                     const signal_params_t sgnlt1,
                     const cl_mem xar,
                     const cl_mem xbr,
                     const cl_mem shft,
                     cl_mem xa,
                     cl_mem xb,
                     OpenCL_handles* cl_handles,
                     const cl_uint num_events_in_wait_list,
                     const cl_event* event_wait_list)
{
  cl_int CL_err = CL_SUCCESS;

  // Helper variable to make pointer types match. Cast to silence warning
  real real_sgnlt1 = (real)sgnlt1,
       real_het1 = (real)het1;
  size_t size_N = (size_t)N;

  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 0, sizeof(cl_mem), &xa);	  checkErr(CL_err, "clSetKernelArg(&xa)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 1, sizeof(cl_mem), &xb);	  checkErr(CL_err, "clSetKernelArg(&xb)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 2, sizeof(cl_mem), &xar);  checkErr(CL_err, "clSetKernelArg(&xar)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 3, sizeof(cl_mem), &xbr);  checkErr(CL_err, "clSetKernelArg(&xbr)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 4, sizeof(real), &het1);	  checkErr(CL_err, "clSetKernelArg(&het1)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 5, sizeof(real), &sgnlt1); checkErr(CL_err, "clSetKernelArg(&sgnlt1)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 6, sizeof(cl_mem), &shft); checkErr(CL_err, "clSetKernelArg(&shft)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod1], 7, sizeof(cl_int), &N);    checkErr(CL_err, "clSetKernelArg(&N)");

  cl_event exec;

  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][PhaseMod1], 1, NULL, &size_N, NULL, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(PhaseMod1)");

#ifdef TESTING
  CL_err = clWaitForEvents(1, &exec); checkErr(CL_err, "clWaitForEvents(PhaseMod1)");

   save_numbered_real_buffer(cl_handles->read_queues[id][idet], shft, N, idet, "pre_fft_phasemod_ifo_sig_shft", SHIFT_DOUBLE);
   save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xar, N, idet, "pre_fft_phasemod_ifo_sig_xDatma", XDATM_DOUBLE);
   save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xbr, N, idet, "pre_fft_phasemod_ifo_sig_xDatmb", XDATM_DOUBLE);
   save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa, N, idet, "pre_fft_phasemod_xa", FFT_DOUBLE);
   save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb, N, idet, "pre_fft_phasemod_xb", FFT_DOUBLE);
#endif

  return exec;
}

cl_event phase_mod_2(const cl_int idet,
                     const cl_int id,
                     const cl_int N,
                     const double het1,
                     const signal_params_t sgnlt1,
                     const cl_mem xar,
                     const cl_mem xbr,
                     const cl_mem shft,
                     cl_mem xa,
                     cl_mem xb,
                     OpenCL_handles* cl_handles,
                     const cl_uint num_events_in_wait_list,
                     const cl_event* event_wait_list)
{
  cl_int CL_err = CL_SUCCESS;

  // Helper variable to make pointer types match. Cast to silence warning
  real real_sgnlt1 = (real)sgnlt1,
       real_het1 = (real)het1;
  size_t size_N = (size_t)N;

  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 0, sizeof(cl_mem), &xa);	  checkErr(CL_err, "clSetKernelArg(&xa)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 1, sizeof(cl_mem), &xb);	  checkErr(CL_err, "clSetKernelArg(&xb)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 2, sizeof(cl_mem), &xar);  checkErr(CL_err, "clSetKernelArg(&xar)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 3, sizeof(cl_mem), &xbr);  checkErr(CL_err, "clSetKernelArg(&xbr)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 4, sizeof(real), &het1);	  checkErr(CL_err, "clSetKernelArg(&het1)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 5, sizeof(real), &sgnlt1); checkErr(CL_err, "clSetKernelArg(&sgnlt1)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 6, sizeof(cl_mem), &shft); checkErr(CL_err, "clSetKernelArg(&shft)");
  clSetKernelArg(cl_handles->kernels[id][PhaseMod2], 7, sizeof(cl_int), &N);    checkErr(CL_err, "clSetKernelArg(&N)");

  cl_event exec;

  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][PhaseMod2], 1, NULL, &size_N, NULL, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(PhaseMod1)");

#ifdef TESTING
  CL_err = clWaitForEvents(1, &exec); checkErr(CL_err, "clWaitForEvents(PhaseMod2)");

  save_numbered_real_buffer(cl_handles->read_queues[id][idet], shft, N, idet, "pre_fft_phasemod_ifo_sig_shft", SHIFT_DOUBLE);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xar, N, idet, "pre_fft_phasemod_ifo_sig_xDatma", XDATM_DOUBLE);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xbr, N, idet, "pre_fft_phasemod_ifo_sig_xDatmb", XDATM_DOUBLE);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa, N, idet, "pre_fft_phasemod_xa", FFT_DOUBLE);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb, N, idet, "pre_fft_phasemod_xb", FFT_DOUBLE);
#endif

  return exec;
}
