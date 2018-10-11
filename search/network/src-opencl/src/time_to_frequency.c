// OpenCL behavioral defines
//
// 1.2+ OpenCL headers: tells the headers not to bitch about clCreateCommandQueue being renamed to clCreateCommandQueueWithProperties
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
//
// Select API to use
#define CL_TARGET_OPENCL_VERSION 120

#include <tshift_pmod.h>

// Polgraw includes
#include <floats.h>     // tshift_pmod_real, tshift_pmod_real3
#include <CL/util.h>    // checkErr

void zero_pad(const cl_int idet,
              const cl_int id,
              const Search_settings *sett,
              cl_mem xa_d,
              cl_mem xb_d,
              OpenCL_handles* cl_handles,
              const cl_uint num_events_in_wait_list,
              const cl_event* event_wait_list,
              cl_event* zero_pad_events)
{
  cl_int CL_err = CL_SUCCESS;

#ifdef _WIN32
  fft_complex pattern = { 0, 0 };
#else
  fft_complex pattern = 0;
#endif

  // Zero pad from offset until the end
  CL_err = clEnqueueFillBuffer(cl_handles->write_queues[id][idet], xa_d, &pattern, sizeof(fft_complex), sett->N * sizeof(fft_complex), (sett->fftpad*sett->nfft - sett->N) * sizeof(fft_complex), num_events_in_wait_list, event_wait_list, &zero_pad_events[0]);
  checkErr(CL_err, "clEnqueueFillBuffer");
  CL_err = clEnqueueFillBuffer(cl_handles->write_queues[id][idet], xb_d, &pattern, sizeof(fft_complex), sett->N * sizeof(fft_complex), (sett->fftpad*sett->nfft - sett->N) * sizeof(fft_complex), num_events_in_wait_list, event_wait_list, &zero_pad_events[1]);
  checkErr(CL_err, "clEnqueueFillBuffer");

#ifdef TESTING
  clWaitForEvents(2, zero_pad_events);
  // Wasteful
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, sett->fftpad*sett->nfft/*sett->Ninterp*/, 0, "pre_fft_post_zero_xa");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, sett->fftpad*sett->nfft/*sett->Ninterp*/, 0, "pre_fft_post_zero_xb");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, sett->fftpad*sett->nfft/*sett->Ninterp*/, 1, "pre_fft_post_zero_xa");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, sett->fftpad*sett->nfft/*sett->Ninterp*/, 1, "pre_fft_post_zero_xb");
#endif
}

void time_to_frequency(const cl_int idet,
                       const cl_int id,
                       const Search_settings *sett,
                       const FFT_plans* plans,
                       cl_mem xa_d,
                       cl_mem xb_d,
                       OpenCL_handles* cl_handles,
                       const cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list,
                       cl_event* fw2_fft_events)
{
  clfftStatus CLFFT_status = CLFFT_SUCCESS;

  clfftEnqueueTransform(plans->plan[id], CLFFT_FORWARD, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &fw2_fft_events[0], &xa_d, NULL, NULL);
  checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");
  clfftEnqueueTransform(plans->plan[id], CLFFT_FORWARD, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &fw2_fft_events[1], &xb_d, NULL, NULL);
  checkErrFFT(CLFFT_status, "clfftEnqueueTransform(CLFFT_FORWARD)");

#ifdef TESTING
  clWaitForEvents(2, fw2_fft_events);
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, sett->nfftf, 0, "post_fft_phasemod_xa");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, sett->nfftf, 0, "post_fft_phasemod_xb");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, sett->nfftf, 1, "post_fft_phasemod_xa");
  save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, sett->nfftf, 1, "post_fft_phasemod_xb");
#endif
}
