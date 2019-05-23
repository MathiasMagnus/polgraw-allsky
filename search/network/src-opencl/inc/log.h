#pragma once

// OpenCL includes
#include <CL/cl.h>      // cl_command_queue, cl_mem

// Standard C includes
//#include <stdbool.h>    // _Bool

/// <summary>Prints the largest absolute value of a host side complex array.</summary>
///
void print_complex_min_max(void* arr, size_t N, const char* msg, _Bool dp);

/// <summary>Prints the first 'count' values of a host side real array.</summary>
///
void print_real_array(void* arr, size_t count, const char* msg, _Bool dp);

/// <summary>Prints the first 'count' values of a host side complex array.</summary>
///
void print_complex_array(void* arr, size_t count, const char* msg, _Bool dp);

/// <summary>Prints the first 'count' values of a device side real array.</summary>
///
void print_real_buffer(cl_command_queue queue, cl_mem buf, int count, const char* msg, _Bool dp);

/// <summary>Prints the first 'count' values of a device side complex array.</summary>
///
void print_complex_buffer(cl_command_queue queue, cl_mem buf, int count, const char* msg, _Bool dp);

/// <summary>Saves values of a host side real array to disk.</summary>
///
void save_real_array(void* arr, size_t count, const char* name, _Bool dp);

/// <summary>Saves values of a host side complex array to disk.</summary>
///
void save_complex_array(void* arr, size_t count, const char* name, _Bool dp);

/// <summary>Saves values of a device side real array to disk.</summary>
///
void save_real_buffer(cl_command_queue queue, cl_mem buf, int count, const char* name, _Bool dp);

/// <summary>Saves values of a device side real array to disk.</summary>
///
void save_real_buffer_with_offset(cl_command_queue queue, cl_mem buf, int off, int count, const char* name, _Bool dp);

/// <summary>Saves values of a device side complex array to disk.</summary>
///
void save_complex_buffer(cl_command_queue queue, cl_mem buf, int count, const char* name, _Bool dp);

/// <summary>Saves values of a device side real array to disk.</summary>
///
void save_numbered_real_buffer(cl_command_queue queue, cl_mem buf, int count, size_t n, const char* name, _Bool dp);

/// <summary>Saves values of a device side complex array to disk.</summary>
///
void save_numbered_complex_buffer(cl_command_queue queue, cl_mem buf, int count, size_t n, const char* name, _Bool dp);

/// <summary>Saves values of a device side real array to disk.</summary>
///
void save_numbered_real_buffer_with_offset(cl_command_queue queue, cl_mem buf, int off, int count, size_t n, const char* name, _Bool dp);

///////////////////////////////////////////////
//                                           //
//               implementation              //
//                                           //
///////////////////////////////////////////////

#include <complex.h>
#ifdef _WIN32
#define complex_float _Fcomplex
#define complex_double _Dcomplex
#else
#define complex_float complex float
#define complex_double complex double
#endif

void print_complex_min_max_sp(complex_float* arr, size_t N, const char* msg);
void print_complex_min_max_dp(complex_double* arr, size_t N, const char* msg);

void print_complex_array_sp(complex_float* arr, size_t count, const char* msg);
void print_complex_array_dp(complex_double* arr, size_t count, const char* msg);

void save_real_array_sp(complex_float* arr, size_t count, const char* name);
void save_real_array_dp(complex_double* arr, size_t count, const char* name);

void save_complex_array_sp(complex_float* arr, size_t count, const char* name);
void save_complex_array_dp(complex_double* arr, size_t count, const char* name);
