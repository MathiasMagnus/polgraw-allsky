// Polgraw includes
#include <CL/util.h>        // checkErr
#include <spline_z.h>
#include <auxi.h>
#include <struct.h>

// Standard C includes
#include <stdlib.h>          // calloc, free
#include <string.h>          // memcpy


/// <summary>Initialize the spline matrices.</summary>
/// <remarks>PCI Should replace it with kernels that initialize on the device.</remarks>
///
void init_spline_matrices(OpenCL_handles* cl_handles,
                          cl_mem* cu_d,  // buffer of complex_devt
                          cl_mem* cu_dl, // buffer of complex_devt
                          cl_mem* cu_du, // buffer of complex_devt
                          cl_mem* cu_B,  // buffer of complex_devt
                          cl_int N)
{
    cl_int CL_err = CL_SUCCESS;
    N -= 1; // N is number of intervals here

    complex_devt *d, *du, *dl;

    d = (complex_devt*)calloc(N - 1, sizeof(complex_devt));
    du = (complex_devt*)calloc(N - 1, sizeof(complex_devt));
    dl = (complex_devt*)calloc(N - 1, sizeof(complex_devt));

    for (int i = 0; i<N - 2; i++)
    {
        dl[i + 1].s[0] = 1;
        du[i].s[0] = 1;
        d[i].s[0] = 4;

        dl[i].s[1] = 0;
        du[i].s[1] = 0;
        d[i].s[1] = 0;
    }

    // dl[0] is 0 and du[N-2]=0
    dl[0].s[0] = 0;
    du[N - 2].s[0] = 0;
    d[N - 2].s[0] = 4;

    dl[N - 2].s[1] = 0;
    du[N - 2].s[1] = 0;
    d[N - 2].s[1] = 0;

    // copy to gpu
    *cu_d = clCreateBuffer(cl_handles->ctx,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           (N - 1) * sizeof(complex_devt),
                           d,
                           &CL_err);
    checkErr(CL_err, "clCreateBuffer(cu_d)");

    *cu_dl = clCreateBuffer(cl_handles->ctx,
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            (N - 1) * sizeof(complex_devt),
                            d,
                            &CL_err);
    checkErr(CL_err, "clCreateBuffer(cu_dl)");

    *cu_du = clCreateBuffer(cl_handles->ctx,
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            (N - 1) * sizeof(complex_devt),
                            d,
                            &CL_err);
    checkErr(CL_err, "clCreateBuffer(cu_du)");

    // allocate B (or z) vector
    *cu_B = clCreateBuffer(cl_handles->ctx,
                           CL_MEM_READ_WRITE,
                           (N + 1) * sizeof(complex_devt),
                           NULL,
                           &CL_err);
    checkErr(CL_err, "clCreateBuffer(cu_B)");

    //clean up
    free(d);
    free(du);
    free(dl);
}

void gpu_interp(cl_mem cu_y,                // buffer of complex_t
                cl_int N,
                cl_mem cu_new_x,            // buffer of real_t
                cl_mem cu_new_y,            // buffer of complex_t
                cl_int new_N,
                cl_mem cu_d,                // buffer of complex_t
                cl_mem cu_dl,               // buffer of complex_t
                cl_mem cu_du,               // buffer of complex_t
                cl_mem cu_B,                // buffer of complex_t
                OpenCL_handles* cl_handles) // handles to OpenCL resources
{/*
    N-=1; // N is number of intervals here
  
    // allocate and compute vector B=z (replaced on gtsv)
    // z has size N+1 (i=0..N), but we solve only for (i=1..N-1)
    // (z[0] = z[N] = 0) because of `natural conditions` of spline
#ifdef _WIN32
    complex_t pattern = {(real_t)0, (real_t)0};
#else
    complex_t pattern = 0;
#endif
    cl_event fill_event;

    clEnqueueFillBuffer(cl_handles->write_queues[0], cu_B, &pattern, sizeof(complex_t), 0, (N + 1) * sizeof(complex_t), 0, NULL, &fill_event);

    clWaitForEvents(1, &fill_event);
    clReleaseEvent(fill_event);

    computeB_gpu(cu_y, cu_B, N, cl_handles); // TODO almost certainly wrong indexing. (Checked, seems legit, though very convoluted)

    tridiagMul_gpu(cu_dl,
                   cu_d,
                   cu_du,
                   cu_B,
                   N + 1,
                   cl_handles); // TODO almost certainly wrong indexing of cu_B + use persistent tmp buffer
    
    interpolate_gpu(cu_new_x,
                    cu_new_y,
                    cu_B,
                    cu_y,
                    N,
                    new_N,
                    cl_handles);*/
}

void spline_interpolate_cpu(const cl_int idet,
                            const cl_int id,
                            const size_t arr_len,
	                        const size_t N,
	                        const int interpftpad,
	                        const real_t sig2,
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
  cl_int CL_err;
  void *xa, *xb, *shftf, *xDatma, *xDatmb;

  xa = clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
			              xa_d,
			              CL_FALSE,
			              CL_MAP_READ,
			              0,
			              arr_len * sizeof(complex_t),
	                      num_events_in_wait_list,
	                      event_wait_list,
			              &spline_map_events[0],
			              &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(fft_arr->xa_d)");

  xb = clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
			              xb_d,
	                      CL_FALSE,
			              CL_MAP_READ,
			              0,
			              arr_len * sizeof(complex_t),
	                      num_events_in_wait_list,
	                      event_wait_list,
	                      &spline_map_events[1],
			              &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(fft_arr->xb_d)");

  shftf = clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
			                 shftf_d,
	                         CL_FALSE,
			                 CL_MAP_READ,
			                 0,
			                 N * sizeof(real_t),
	                         num_events_in_wait_list,
	                         event_wait_list,
	                         &spline_map_events[2],
			                 &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(ifo[n].sig.shftf_d)");

  xDatma = clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
                              xDatma_d,
	                          CL_FALSE,
                              CL_MAP_WRITE,
                              0,
                              N * sizeof(complex_devt),
	                          num_events_in_wait_list,
	                          event_wait_list,
	                          &spline_map_events[3],
                              &CL_err);
  checkErr(CL_err, "clEnqueueMapBuffer(ifo[n].sig.xDatma_d)");

  xDatmb = clEnqueueMapBuffer(cl_handles->read_queues[id][idet],
                              xDatmb_d,
	                          CL_FALSE,
                              CL_MAP_WRITE,
                              0,
                              N * sizeof(complex_devt),
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
  save_numbered_complex_buffer(cl_handles->exec_queues[0], xDatma_d, N, n, "ifo_sig_xDatma");
  save_numbered_complex_buffer(cl_handles->exec_queues[0], xDatmb_d, N, n, "ifo_sig_xDatmb");
#endif

  blas_scale(idet, id, N, 1. / sig2,
             xDatma_d,
             xDatmb_d,
             blas_handles,
             cl_handles,
             2, spline_unmap_events + 3, // it is enough to wait on the last 2 of the unmaps: xDatma, xDatmb
             spline_blas_events);
#ifdef TESTING
  save_numbered_complex_buffer(cl_handles->exec_queues[0], ifo[n].sig.xDatma_d, N, n, "rescaled_ifo_sig_xDatma");
  save_numbered_complex_buffer(cl_handles->exec_queues[0], ifo[n].sig.xDatmb_d, N, n, "rescaled_ifo_sig_xDatmb");
#endif
}

void blas_scale(const cl_int idet,
                const cl_int id,
                const size_t n,
                const real_t a,
                cl_mem xa_d,
                cl_mem xb_d,
                BLAS_handles* blas_handles,
                OpenCL_handles* cl_handles,
                const cl_uint num_events_in_wait_list,
                const cl_event* event_wait_list,
                cl_event* blas_exec)
{
  clblasStatus status[2];

#ifdef COMP_FLOAT
  status[0] = clblasSscal(n * 2, a, xa_d, 0, 1, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[0]); checkErrBLAS(status[0], "clblasSscal(xa_d)");
  status[1] = clblasSscal(n * 2, a, xb_d, 0, 1, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[1]); checkErrBLAS(status[1], "clblasSscal(xb_d)");
#else
  status[0] = clblasDscal(n * 2, a, xa_d, 0, 1, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[0]); checkErrBLAS(status[0], "clblasDscal(xa_d)");
  status[1] = clblasDscal(n * 2, a, xb_d, 0, 1, 1, &cl_handles->exec_queues[id][idet], num_events_in_wait_list, event_wait_list, &blas_exec[1]); checkErrBLAS(status[1], "clblasDscal(xb_d)");
#endif // COMP_FLOAT
}

void computeB_gpu(cl_mem y,
                  cl_mem B,
                  cl_int N,
                  OpenCL_handles* cl_handles)
{
//    cl_int CL_err = CL_SUCCESS;
//
//    clSetKernelArg(cl_handles->kernels[id][ComputeB], 0, sizeof(cl_mem), &y);
//    clSetKernelArg(cl_handles->kernels[id][ComputeB], 1, sizeof(cl_mem), &B);
//    clSetKernelArg(cl_handles->kernels[id][ComputeB], 2, sizeof(cl_int), &N);
//
//    cl_event exec;
//    size_t size_N = (size_t)(N - 1); // Helper variable to make pointer types match. Cast to silence warning
//                                     // Subtract 
//    
//    // TODO: introduce offsets to preserve leading and trailing zeroes
//    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[id][0], cl_handles->kernels[id][ComputeB], 1, NULL, &size_N, NULL, 0, NULL, &exec);
//
//    clWaitForEvents(1, &exec);
//
//    clReleaseEvent(exec);
}

void tridiagMul_gpu(cl_mem dl,
                    cl_mem d,
                    cl_mem du,
                    cl_mem x,
                    cl_int length,
                    OpenCL_handles* cl_handles)
{
//    cl_int CL_err = CL_SUCCESS;
//
//    cl_mem y = clCreateBuffer(cl_handles->ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, length * sizeof(complex_t), NULL, &CL_err);
//
//    clSetKernelArg(cl_handles->kernels[TriDiagMul], 0, sizeof(cl_mem), &dl);
//    clSetKernelArg(cl_handles->kernels[TriDiagMul], 1, sizeof(cl_mem), &d);
//    clSetKernelArg(cl_handles->kernels[TriDiagMul], 2, sizeof(cl_mem), &du);
//    clSetKernelArg(cl_handles->kernels[TriDiagMul], 3, sizeof(cl_mem), &x);
//    clSetKernelArg(cl_handles->kernels[TriDiagMul], 4, sizeof(cl_mem), &y);
//
//    cl_event events[2];
//    size_t size_length = (size_t)length; // Helper variable to make pointer types match. Cast to silence warning
//
//    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[0], cl_handles->kernels[TriDiagMul], 1, NULL, &size_length, NULL, 0, NULL, &events[0]);
//    CL_err = clEnqueueCopyBuffer(cl_handles->write_queues[0], y, x, 0, 0, length * sizeof(complex_t), 1, &events[0], &events[1]);
//
//    clWaitForEvents(2, events);
//
//    for (size_t i = 0 ; i < 2; ++i) clReleaseEvent(events[i]);
//    clReleaseMemObject(y);
}

void interpolate_gpu(cl_mem new_x,
                     cl_mem new_y,
                     cl_mem z,
                     cl_mem y,
                     cl_int N,
                     cl_int new_N,
                     OpenCL_handles* cl_handles)
{
//    cl_int CL_err = CL_SUCCESS;
//
//    clSetKernelArg(cl_handles->kernels[Interpolate], 0, sizeof(cl_mem), &new_x);
//    clSetKernelArg(cl_handles->kernels[Interpolate], 1, sizeof(cl_mem), &new_y);
//    clSetKernelArg(cl_handles->kernels[Interpolate], 2, sizeof(cl_mem), &z);
//    clSetKernelArg(cl_handles->kernels[Interpolate], 3, sizeof(cl_mem), &y);
//    clSetKernelArg(cl_handles->kernels[Interpolate], 4, sizeof(cl_int), &N);
//    clSetKernelArg(cl_handles->kernels[Interpolate], 5, sizeof(cl_int), &new_N);
//
//    cl_event exec;
//    size_t size_new_N = (size_t)new_N; // Helper variable to make pointer types match. Cast to silence warning
//    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[0], cl_handles->kernels[Interpolate], 1, NULL, &size_new_N, NULL, 0, NULL, &exec);
//
//    clWaitForEvents(1, &exec);
//
//    clReleaseEvent(exec);
}
