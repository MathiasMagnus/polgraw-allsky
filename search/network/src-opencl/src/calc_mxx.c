#include <calc_mxx.h>

void calc_mxx(const cl_uint nifo,
              const cl_int id,
              const cl_mem* aadots_d,
              const cl_mem* bbdots_d,
              const Detector_settings* ifo,
              cl_mem maa_d,
              cl_mem mbb_d,
              OpenCL_handles* cl_handles,
              const cl_uint num_events_in_wait_list,
              const cl_event** event_wait_list,
              cl_event* mxx_fill_events,
              cl_event* axpy_events)
{
  cl_int CL_err = CL_SUCCESS;
  clblasStatus status[2] = { clblasSuccess, clblasSuccess };

  ampl_mod_real pattern = 0;

  CL_err = clEnqueueFillBuffer(cl_handles->write_queues[id][0], maa_d, &pattern, sizeof(ampl_mod_real), 0, sizeof(ampl_mod_real), 0, NULL, &mxx_fill_events[0]); checkErr(CL_err, "clEnqueueFillBuffer(maa_d)");
  CL_err = clEnqueueFillBuffer(cl_handles->write_queues[id][0], mbb_d, &pattern, sizeof(ampl_mod_real), 0, sizeof(ampl_mod_real), 0, NULL, &mxx_fill_events[1]); checkErr(CL_err, "clEnqueueFillBuffer(mbb_d)");

  cl_event* input_wait_events = (cl_event*)malloc((nifo * 2 + 2) * sizeof(cl_event));

  for (cl_uint i = 0; i < nifo; ++i)
  {
    input_wait_events[i * 2 + 0] = event_wait_list[i][0];
    input_wait_events[i * 2 + 1] = event_wait_list[i][1];
  }
  input_wait_events[nifo * 2 + 0] = mxx_fill_events[0];
  input_wait_events[nifo * 2 + 1] = mxx_fill_events[1];

#if AMPL_MOD_DOUBLE
  status[0] = clblasDaxpy(1, 1 / ifo[0].sig.sig2, aadots_d[0], 0, 1, maa_d, 0, 1, 1, &cl_handles->exec_queues[id][0], (nifo * 2 + 2), input_wait_events, &axpy_events[0]); checkErrBLAS(status[0], "clblasDaxpy()");
  status[1] = clblasDaxpy(1, 1 / ifo[0].sig.sig2, bbdots_d[0], 0, 1, mbb_d, 0, 1, 1, &cl_handles->exec_queues[id][0], (nifo * 2 + 2), input_wait_events, &axpy_events[1]); checkErrBLAS(status[1], "clblasDaxpy()");
#else
  status[0] = clblasSaxpy(1, (ampl_mod_real)(1 / ifo[0].sig.sig2), aadots_d[0], 0, 1, maa_d, 0, 1, 1, &cl_handles->exec_queues[id][0], (nifo * 2 + 2), input_wait_events, &axpy_events[0]); checkErrBLAS(status[0], "clblasDaxpy()");
  status[1] = clblasSaxpy(1, (ampl_mod_real)(1 / ifo[0].sig.sig2), bbdots_d[0], 0, 1, mbb_d, 0, 1, 1, &cl_handles->exec_queues[id][0], (nifo * 2 + 2), input_wait_events, &axpy_events[1]); checkErrBLAS(status[1], "clblasDaxpy()");
#endif

  for (cl_uint i = 1; i < nifo; ++i)
  {
#ifdef AMPL_MOD_DOUBLE
    status[0] = clblasSaxpy(1, 1 / ifo[i].sig.sig2, aadots_d[i], 0, 1, maa_d, 0, 1, 1, &cl_handles->exec_queues[id][0], 2, &axpy_events[(i - 1) * 2 + 0], &axpy_events[i * 2 + 0]); checkErrBLAS(status[0], "clblasDaxpy()");
    status[1] = clblasSaxpy(1, 1 / ifo[i].sig.sig2, bbdots_d[i], 0, 1, mbb_d, 0, 1, 1, &cl_handles->exec_queues[id][0], 2, &axpy_events[(i - 1) * 2 + 1], &axpy_events[i * 2 + 1]); checkErrBLAS(status[1], "clblasDaxpy()");
#else
    status[0] = clblasDaxpy(1, 1 / ifo[i].sig.sig2, aadots_d[i], 0, 1, maa_d, 0, 1, 1, &cl_handles->exec_queues[id][0], 2, &axpy_events[(i - 1) * 2 + 0], &axpy_events[i * 2 + 0]); checkErrBLAS(status[0], "clblasDaxpy()");
    status[1] = clblasDaxpy(1, 1 / ifo[i].sig.sig2, bbdots_d[i], 0, 1, mbb_d, 0, 1, 1, &cl_handles->exec_queues[id][0], 2, &axpy_events[(i - 1) * 2 + 1], &axpy_events[i * 2 + 1]); checkErrBLAS(status[1], "clblasDaxpy()");
#endif
  }

  // Cleanup
  for (cl_uint i = 1; i < nifo; ++i)
  {
    CL_err = clReleaseEvent(input_wait_events[i]);
    checkErr(CL_err, "clReleaseEvent(input_wait_events[i])");
  }
  free(input_wait_events);
}