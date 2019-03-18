// OpenCL behavioral defines
//
// 1.2+ OpenCL headers: tells the headers not to bitch about clCreateCommandQueue being renamed to clCreateCommandQueueWithProperties
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
//
// Select API to use
#define CL_TARGET_OPENCL_VERSION 120

#include <blas_dot.h>

// Polgraw includes
#include <CL/util.h>    // checkErrBLAS

// clBLAS includes
#include <clBLAS.h>     // clblas?dot

void blas_dot(const cl_int idet,
              const cl_int id,
              const cl_uint n,
              const cl_mem aa_d,
              const cl_mem bb_d,
              cl_mem aadot_d,
              cl_mem bbdot_d,
              BLAS_handles* blas_handles,
              OpenCL_handles* cl_handles,
              const cl_uint num_events_in_wait_list,
              const cl_event* event_wait_list,
              cl_event* blas_exec)
{
  clblasStatus status[2] = { clblasSuccess, clblasSuccess };

#if AMPL_MOD_DOUBLE
  status[0] = clblasDdot(n, aadot_d, 0, aa_d, 0, 1, aa_d, 0, 1, blas_handles->aaScratch_d[id][idet], 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[0]); checkErrBLAS(status[0], "clblasDdot()");
  status[1] = clblasDdot(n, bbdot_d, 0, bb_d, 0, 1, bb_d, 0, 1, blas_handles->bbScratch_d[id][idet], 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[1]); checkErrBLAS(status[1], "clblasDdot()");
#else
  status[0] = clblasSdot(n, aadot_d, 0, aa_d, 0, 1, aa_d, 0, 1, blas_handles->aaScratch_d[id][idet], 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[0]); checkErrBLAS(status[0], "clblasDdot()");
  status[1] = clblasSdot(n, bbdot_d, 0, bb_d, 0, 1, bb_d, 0, 1, blas_handles->bbScratch_d[id][idet], 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[1]); checkErrBLAS(status[1], "clblasDdot()");
#endif
}
