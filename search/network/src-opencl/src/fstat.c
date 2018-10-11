// OpenCL behavioral defines
//
// 1.2+ OpenCL headers: tells the headers not to bitch about clCreateCommandQueue being renamed to clCreateCommandQueueWithProperties
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
//
// Select API to use
#define CL_TARGET_OPENCL_VERSION 120

#include <fstat.h>

// Polgraw includes
#include <floats.h>     // tshift_pmod_real, tshift_pmod_real3
#include <CL/util.h>    // checkErr

cl_event compute_Fstat(const cl_int idet,
                       const cl_int id,
                       const cl_int nmin,
                       const cl_int nmax,
                       const cl_mem xa_d,
                       const cl_mem xb_d,
                       const cl_mem maa_d,
                       const cl_mem mbb_d,
                       cl_mem F_d,
                       OpenCL_handles* cl_handles,
                       const cl_uint num_events_in_wait_list,
                       const cl_event* event_wait_list)
{
    cl_int CL_err = CL_SUCCESS;

  cl_int N = nmax - nmin;
  // Helper variable to make pointer types match. Cast to silence warning
  size_t size_N = (size_t)(N),
         size_nmin = (size_t)nmin;

  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 0, sizeof(cl_mem), &xa_d);  checkErr(CL_err, "clSetKernelArg(&xa_d)");
  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 1, sizeof(cl_mem), &xb_d);  checkErr(CL_err, "clSetKernelArg(&xb_d)");
  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 2, sizeof(cl_mem), &F_d);   checkErr(CL_err, "clSetKernelArg(&F_d)");
  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 3, sizeof(cl_mem), &maa_d); checkErr(CL_err, "clSetKernelArg(&maa_d)");
  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 4, sizeof(cl_mem), &mbb_d); checkErr(CL_err, "clSetKernelArg(&mbb_d)");
  clSetKernelArg(cl_handles->kernels[id][ComputeFStat], 5, sizeof(cl_int), &N);     checkErr(CL_err, "clSetKernelArg(&N)");

  cl_event exec;

  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][ComputeFStat], 1, &size_nmin, &size_N, NULL, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(ComputeFStat)");

#ifdef TESTING
  // Wasteful
  clWaitForEvents(1, &exec);
  save_numbered_real_buffer_with_offset(cl_handles->read_queues[id][idet], F_d, nmin, nmax - nmin, 0, "Fstat");
  save_numbered_real_buffer_with_offset(cl_handles->read_queues[id][idet], F_d, nmin, nmax - nmin, 1, "Fstat");
#endif

  return exec;
}

cl_event normalize_FStat_wg_reduce(const cl_int idet,
                                   const cl_int id,
                                   const cl_int nmin,
                                   const cl_int nmax,
                                   const cl_uint nav,
                                   cl_mem F_d,
                                   OpenCL_handles* cl_handles,
                                   const cl_uint num_events_in_wait_list,
                                   const cl_event* event_wait_list)
{
  cl_int CL_err = CL_SUCCESS;
  size_t max_wgs;             // maximum supported wgs on the device (limited by register count)

  CL_err = clGetKernelWorkGroupInfo(cl_handles->kernels[id][NormalizeFStatWG],
                                    cl_handles->devs[id],
                                    CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(size_t),
                                    &max_wgs,
                                    NULL);
  checkErr(CL_err, "clGetKernelWorkGroupInfo(FStatSimple, CL_KERNEL_WORK_GROUP_SIZE)");

  clSetKernelArg(cl_handles->kernels[id][NormalizeFStatWG], 0, sizeof(cl_mem), &F_d);           checkErr(CL_err, "clSetKernelArg(&F_d)");
  clSetKernelArg(cl_handles->kernels[id][NormalizeFStatWG], 1, nav * sizeof(fstat_real), NULL); checkErr(CL_err, "clSetKernelArg(&shared)");
  clSetKernelArg(cl_handles->kernels[id][NormalizeFStatWG], 2, sizeof(cl_uint), &nav);          checkErr(CL_err, "clSetKernelArg(&nav)");

  size_t wgs = nav < max_wgs ? nav : max_wgs; // std::min
  size_t gsi = wgs * ((nmax - nmin) / nav); // gsi = multiply wgs (processing one nav bunch) with the number of navs
  size_t off = nmin;

  // Check preconditions of kernel before launch
  assert(nav <= 4096); // PRECONDITION: Assumption that nav number of doubles even fit into local memory and no multi-pass needed for fetching data from global
                       //
                       // NOTE: According to the OpenCL 1.2 spec, Table 4.3 on Device Queries, page 43: CL_DEVICE_LOCAL_MEM_SIZE must be at least 32kB for devices
                       //       of type other than CL_DEVICE_TYPE_CUSTOM. 4096 count of double even fits into 32kB
  assert(wgs <= nav);
  assert(nav % wgs == 0);
  assert(((wgs != 0) && !(wgs & (wgs - 1)))); // Method 9. @ https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
  //assert(gsi % nav == 0);

  cl_event exec;
  CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], cl_handles->kernels[id][NormalizeFStatWG], 1, &off, &gsi, &wgs, num_events_in_wait_list, event_wait_list, &exec);
  checkErr(CL_err, "clEnqueueNDRangeKernel(FStatSimple)");

#ifdef TESTING
  // Wasteful
  clWaitForEvents(1, &exec);
  save_numbered_real_buffer_with_offset(cl_handles->read_queues[id][idet], F_d, nmin, nmax - nmin, 0, "Fstat_norm");
  save_numbered_real_buffer_with_offset(cl_handles->read_queues[id][idet], F_d, nmin, nmax - nmin, 1, "Fstat_norm");
#endif

  return exec;
}
