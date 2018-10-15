// C behavioral defines
//
// When available, opt-in for secure CRT functions
#ifdef __STDC_LIB_EXT1__
#define __STDC_WANT_LIB_EXT1__ 1
#endif
// OpenCL behavioral defines
//
// 1.2+ OpenCL headers: tells the headers not to bitch about clCreateCommandQueue being renamed to clCreateCommandQueueWithProperties
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
//
// Select API to use
#define CL_TARGET_OPENCL_VERSION 120

#include <log.h>

// Polgraw includes
#include <CL/util.h>    // checkErr

// Standard C includes
#include <stdio.h>      // fopen
#include <string.h>     // strcpy, strcat

/// <summary>Prints the largest absolute value of a host side complex array.</summary>
///
void print_complex_min_max(void* arr, size_t N, const char* msg, _Bool dp)
{
  if (dp)
    print_complex_min_max_dp(arr, N, msg);
  else
    print_complex_min_max_sp(arr, N, msg);
}

/// <summary>Prints the first 'n' values of a host side real array.</summary>
///
void print_real_array(void* arr, size_t count, const char* msg, _Bool dp)
{
#ifdef _WIN32
  int bytes = printf_s("%s:\n\n", msg);
  size_t i;
  if (dp)
    for (i = 0; i < count; ++i) bytes = printf_s("\t%f\n", ((double*)arr)[i]);
  else
    for (i = 0; i < count; ++i) bytes = printf_s("\t%f\n", ((float*)arr)[i]);
  bytes = printf_s("\n");
#else
  printf("%s:\n\n", msg);
  size_t i;
  if (dp)
    for (i = 0; i < count; ++i) printf("\t%f\n", ((double*)arr)[i]);
  else
    for (i = 0; i < count; ++i) printf("\t%f\n", ((float*)arr)[i]);
  printf("\n");
#endif
  fflush(NULL);
}

/// <summary>Prints the first 'n' values of a host side complex array.</summary>
///
void print_complex_array(void* arr, size_t count, const char* msg, _Bool dp)
{
  if (dp)
    print_complex_array_dp(arr, count, msg);
  else
    print_complex_array_sp(arr, count, msg);
}

/// <summary>Saves values of a host side real array to disk.</summary>
///
void save_real_array(void* arr, size_t count, const char* name, _Bool dp)
{
  char filename[1024];

  strcpy(filename, "cl_");
  strcat(filename, name);
  strcat(filename, ".dat");

  FILE* fc = fopen(filename, "w");
  if (fc == NULL) perror("Failed to open output file.");

  size_t i;
  if (dp)
    for (i = (size_t)0; i < count; ++i)
    {
      //fprintf(fc, "%zu %e\n", i, ((double*)arr)[i]); // GnuPlot friendly
      fprintf(fc, "%e\n", ((double*)arr)[i]); // STL friendly
    }
  else
    for (i = (size_t)0; i < count; ++i)
    {
      //fprintf(fc, "%zu %e\n", i, ((float*)arr)[i]); // GnuPlot friendly
      fprintf(fc, "%e\n", ((float*)arr)[i]); // STL friendly
    }

  int close = fclose(fc);
  if (close == EOF) perror("Failed to close output file.");
}

/// <summary>Saves values of a host side complex array to disk.</summary>
///
void save_complex_array(void* arr, size_t count, const char* name, _Bool dp)
{
    if (dp)
        save_complex_array_dp(arr, count, name);
    else
        save_complex_array_sp(arr, count, name);
}

/// <summary>Saves values of a device side real array to disk.</summary>
///
void save_real_buffer(cl_command_queue queue, cl_mem buf, int count, const char* name, _Bool dp)
{
    save_real_buffer_with_offset(queue, buf, 0, count, name, dp);
}

void save_real_buffer_with_offset(cl_command_queue queue, cl_mem buf, int off, int count, const char* name, _Bool dp)
{
  cl_int CL_err;
  cl_event map, unmap;

  void* temp =
    clEnqueueMapBuffer(queue,
                       buf,
                       CL_TRUE,
                       CL_MAP_READ,
                       off * (dp ? sizeof(double) : sizeof(float)),
                       count * (dp ? sizeof(double) : sizeof(float)),
                       0, NULL,
                       &map,
                       &CL_err);
  checkErr(CL_err, "clEnqueueMapBufffer");

  CL_err = clWaitForEvents(1, &map);
  checkErr(CL_err, "clWaitForEvents");

  save_real_array(temp, count, name, dp);

  CL_err = clEnqueueUnmapMemObject(queue, buf, temp, 0, NULL, &unmap);
  checkErr(CL_err, "clEnqueueUnmapMemObject");

  clReleaseEvent(map);
  clReleaseEvent(unmap);
}

/// <summary>Saves values of a device side complex array to disk.</summary>
///
void save_complex_buffer(cl_command_queue queue, cl_mem buf, int count, const char* name, _Bool dp)
{
    cl_int CL_err;
    cl_event map, unmap;

    void* temp =
        clEnqueueMapBuffer(queue,
            buf,
            CL_TRUE,
            CL_MAP_READ,
            0, count * (dp ? sizeof(complex_double) : sizeof(complex_float)),
            0, NULL,
            &map,
            &CL_err);
    checkErr(CL_err, "clEnqueueMapBufffer");

    CL_err = clWaitForEvents(1, &map);
    checkErr(CL_err, "clWaitForEvents");

    save_complex_array(temp, count, name, dp);

    CL_err = clEnqueueUnmapMemObject(queue, buf, temp, 0, NULL, &unmap);
    checkErr(CL_err, "clEnqueueUnmapMemObject");

    clReleaseEvent(map);
    clReleaseEvent(unmap);
}

/// <summary>Saves values of a device side real array to disk.</summary>
///
void save_numbered_real_buffer(cl_command_queue queue, cl_mem buf, int count, size_t n, const char* name, _Bool dp)
{
    save_numbered_real_buffer_with_offset(queue, buf, 0, count, n, name, dp);
}

void save_numbered_real_buffer_with_offset(cl_command_queue queue, cl_mem buf, int off, int count, size_t n, const char* name, _Bool dp)
{
    char num_filename[1024],
         num_str[10];

    sprintf(num_str, "%zu", n);

    strcpy(num_filename, name);
    strcat(num_filename, ".");
    strcat(num_filename, num_str);

    save_real_buffer_with_offset(queue, buf, off, count, num_filename, dp);
}

/// <summary>Saves values of a device side complex array to disk.</summary>
///
void save_numbered_complex_buffer(cl_command_queue queue, cl_mem buf, int count, size_t n, const char* name, _Bool dp)
{
    char num_filename[1024],
        num_str[10];

    sprintf(num_str, "%zu", n);

    strcpy(num_filename, name);
    strcat(num_filename, ".");
    strcat(num_filename, num_str);

    save_complex_buffer(queue, buf, count, num_filename, dp);
}