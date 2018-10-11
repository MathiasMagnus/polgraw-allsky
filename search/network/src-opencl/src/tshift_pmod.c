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

typedef tshift_pmod_real real;
typedef tshift_pmod_real3 real3;

cl_event tshift_pmod(const cl_int idet,
                     const cl_int id,
                     const cl_int N,
                     const cl_int nfft,
                     const cl_int interpftpad,
                     const double shft1,
                     const double het0,
                     const double oms,
                     const cl_double3 ns,
                     const cl_mem xDat_d,
                     const cl_mem aa_d,
                     const cl_mem bb_d,
                     const cl_mem DetSSB_d,
                     cl_mem xa_d,
                     cl_mem xb_d,
                     cl_mem shft_d,
                     cl_mem shftf_d,
                     cl_mem tshift_d,
                     const OpenCL_handles* cl_handles,
                     const cl_uint num_events_in_wait_list,
                     const cl_event* event_wait_list)
{
    cl_int CL_err = CL_SUCCESS;

    // Helper variable to make pointer types match. Cast to silence warning
    real real_shft1 = (real)shft1,
         real_het0 = (real)het0,
         real_oms = (real)oms;
    real3 real3_ns = { (real)ns.s0,
                       (real)ns.s1,
                       (real)ns.s2,
                       (real)ns.s3 };
    size_t size_nfft = (size_t)nfft;

    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 0, sizeof(cl_int), &N);            checkErr(CL_err, "clSetKernelArg(&N)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 1, sizeof(cl_int), &nfft);         checkErr(CL_err, "clSetKernelArg(&nfft)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 2, sizeof(cl_int), &interpftpad);  checkErr(CL_err, "clSetKernelArg(&interftpad)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 3, sizeof(real), &real_shft1);     checkErr(CL_err, "clSetKernelArg(&shft1)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 4, sizeof(real), &real_het0);      checkErr(CL_err, "clSetKernelArg(&het0)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 5, sizeof(real), &real_oms);       checkErr(CL_err, "clSetKernelArg(&oms)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 6, sizeof(real3), &real3_ns);      checkErr(CL_err, "clSetKernelArg(&ns0)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 7, sizeof(cl_mem), &xDat_d);       checkErr(CL_err, "clSetKernelArg(&xDat_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 8, sizeof(cl_mem), &aa_d);         checkErr(CL_err, "clSetKernelArg(&aa_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 9, sizeof(cl_mem), &bb_d);         checkErr(CL_err, "clSetKernelArg(&bb_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 10, sizeof(cl_mem), &DetSSB_d);    checkErr(CL_err, "clSetKernelArg(&DetSSB_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 11, sizeof(cl_mem), &xa_d);        checkErr(CL_err, "clSetKernelArg(&xa_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 12, sizeof(cl_mem), &xb_d);        checkErr(CL_err, "clSetKernelArg(&xb_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 13, sizeof(cl_mem), &shft_d);      checkErr(CL_err, "clSetKernelArg(&shft_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 14, sizeof(cl_mem), &shftf_d);     checkErr(CL_err, "clSetKernelArg(&shftf_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][TShiftPMod], 15, sizeof(cl_mem), &tshift_d);    checkErr(CL_err, "clSetKernelArg(&tshift_d)");
    
    cl_event exec;

    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet],  // queue
                                    cl_handles->kernels[id][TShiftPMod],// kernel
                                    1,                                  // dimensions
                                    NULL, &size_nfft, NULL,             // offset, global/local work-size
                                    num_events_in_wait_list,            // nomen est omen
                                    event_wait_list,                    // nomen est omen
                                    &exec);                             // event out
    checkErr(CL_err, "clEnqueueNDRangeKernel(cl_handles->kernels[TShiftPMod])");

#ifdef TESTING
    clWaitForEvents(1, &exec);
    save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xa_d, nfft, idet, "xa_time");
    save_numbered_complex_buffer(cl_handles->read_queues[id][idet], xb_d, nfft, idet, "xb_time");
    save_numbered_real_buffer(cl_handles->read_queues[id][idet], shft_d, N, idet, "ifo_sig_shft");
    save_numbered_real_buffer(cl_handles->read_queues[id][idet], shftf_d, N, idet, "ifo_sig_shftf");
#endif

    return exec;
}
