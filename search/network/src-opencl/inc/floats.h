#pragma once

// OpenCL includes
#include <CL/cl.h>      // cl_float, cl_double

// Standard C includes
#include <complex.h>    // _Dcomplex

typedef double modvir_real_in;
typedef double modvir_real;
typedef double modvir_real_out;

//changing computations in spindown loop to single-precision arithmetic
#ifdef COMP_FLOAT //if single-precision
#define CLFFT_TRANSFORM_PRECISION CLFFT_SINGLE
#define CLFFT_TRANSFORM_LAYOUT CLFFT_REAL
#define COMPLEX_TYPE cufftComplex
#define FLOAT_TYPE float
#define HOST_COMPLEX_TYPE complex float
#else //if double-precision
#define CLFFT_TRANSFORM_PRECISION CLFFT_DOUBLE
#define CLFFT_TRANSFORM_LAYOUT CLFFT_COMPLEX_INTERLEAVED
typedef cl_double real_t;
typedef cl_double2 complex_devt;
typedef cl_double3 real3_t;
#define FLOAT_TYPE cl_double
#define COMPLEX_TYPE cl_double2
#ifdef _WIN32
typedef _Dcomplex complex_t;
#define HOST_COMPLEX_TYPE _Dcomplex

complex_t cbuild(const real_t real, const real_t imag);

complex_t cmulcc(const complex_t lhs, const complex_t rhs);
complex_t cmulcr(const complex_t lhs, const real_t rhs);
complex_t cmulrc(const real_t lhs, const complex_t rhs);

complex_t cdivcc(const complex_t lhs, const complex_t rhs);
complex_t cdivcr(const complex_t lhs, const real_t rhs);
complex_t cdivrc(const real_t lhs, const complex_t rhs);

complex_t caddcc(const complex_t lhs, const complex_t rhs);
complex_t caddcr(const complex_t lhs, const real_t rhs);
complex_t caddrc(const real_t lhs, const complex_t rhs);

complex_t csubcc(const complex_t lhs, const complex_t rhs);
complex_t csubcr(const complex_t lhs, const real_t rhs);
complex_t csubrc(const real_t lhs, const complex_t rhs);

#else
typedef complex double complex_t;
#define HOST_COMPLEX_TYPE complex double
#endif // WIN32
#endif
