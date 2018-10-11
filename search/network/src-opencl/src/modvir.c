// OpenCL behavioral defines
//
// 1.2+ OpenCL headers: tells the headers not to bitch about clCreateCommandQueue being renamed to clCreateCommandQueueWithProperties
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
//
// Select API to use
#define CL_TARGET_OPENCL_VERSION 120

#include <modvir.h>

// Polgraw includes
#include <auxi.h>       // sqr
#include <floats.h>     // modvir_real_t
#include <CL/util.h>    // checkErr

typedef modvir_real real;

cl_event modvir(const cl_int idet,
                const cl_int id,
                const cl_int Np,
                const double sinal,
                const double cosal,
                const double sindel,
                const double cosdel,
                const double cphir,
                const double sphir,
                const double omr,
                const cl_mem ifo_amod_d,
                cl_mem aa_d,
                cl_mem bb_d,
                const OpenCL_handles* cl_handles,
                const cl_uint num_events_in_wait_list,
                const cl_event* event_wait_list)
{
    cl_int CL_err = CL_SUCCESS;
    // Helper variable to make pointer types match. Cast to silence warning
    real cosalfr = (real)(cosal * (cphir) + sinal * (sphir)),
         sinalfr = (real)(sinal * (cphir) - cosal * (sphir)),
         c2d = (real)sqr(cosdel),
         c2sd = (real)(sindel * cosdel),
         real_sindel = (real)sindel,
         real_cosdel = (real)cosdel,
         real_omr = (real)omr;
    size_t size_Np = (size_t)Np;

    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 0, sizeof(cl_int), &idet);             checkErr(CL_err, "clSetKernelArg(&idet)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 1, sizeof(cl_int), &Np);               checkErr(CL_err, "clSetKernelArg(&Np)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 2, sizeof(real), &sinalfr);            checkErr(CL_err, "clSetKernelArg(&sinalfr)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 3, sizeof(real), &cosalfr);            checkErr(CL_err, "clSetKernelArg(&cosalfr)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 4, sizeof(real), &real_sindel);        checkErr(CL_err, "clSetKernelArg(&sindel)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 5, sizeof(real), &real_cosdel);        checkErr(CL_err, "clSetKernelArg(&cosdel)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 6, sizeof(real), &c2d);                checkErr(CL_err, "clSetKernelArg(&c2d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 7, sizeof(real), &c2sd);               checkErr(CL_err, "clSetKernelArg(&c2sd)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 8, sizeof(real), &real_omr);           checkErr(CL_err, "clSetKernelArg(&real_omr)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 9, sizeof(cl_mem), &ifo_amod_d);       checkErr(CL_err, "clSetKernelArg(&ifo_amod_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 10, sizeof(cl_mem), &aa_d);            checkErr(CL_err, "clSetKernelArg(&aa_d)");
    CL_err = clSetKernelArg(cl_handles->kernels[id][Modvir], 11, sizeof(cl_mem), &bb_d);            checkErr(CL_err, "clSetKernelArg(&bb_d)");

    cl_event exec;

    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][idet], // queue
                                    cl_handles->kernels[id][Modvir],   // kernel
                                    1,                                 // dimensions
                                    NULL, &size_Np, NULL,              // offset, global/local work-size
                                    num_events_in_wait_list,           // nomen est omen
                                    event_wait_list,                   // nomen est omen
                                    &exec);                            // event out
    checkErr(CL_err, "clEnqueueNDRangeKernel(cl_handles->kernels[Modvir])");

#ifdef TESTING
    clWaitForEvents(1, &exec);
    save_numbered_real_buffer(cl_handles->read_queues[id][idet], aa_d, Np, idet, "ifo_sig_aa");
    save_numbered_real_buffer(cl_handles->read_queues[id][idet], bb_d, Np, idet, "ifo_sig_bb");
#endif

    return exec;
}
