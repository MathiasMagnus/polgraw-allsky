#ifndef __FLOATS_HCL__
#define __FLOATS_HCL__

// Polgraw includes
#include <../inc/precision.h>


#ifdef COMP_FLOAT
typedef float real_t;
typedef float2 complex_t;
#else
typedef double real_t;
typedef double2 complex_t;
typedef double3 real3_t;
#endif // COMP_FLOAT

real_t creal(const complex_t val) { return val.x; }
real_t cimag(const complex_t val) { return val.y; }

complex_t cbuild(const real_t real, const real_t imag) { return (complex_t)(real, imag); }

complex_t cmulcc(const complex_t lhs, const complex_t rhs) { return cbuild(creal(lhs)*creal(rhs) - cimag(lhs)*cimag(rhs),
                                                                           creal(lhs)*cimag(rhs) + cimag(lhs)*creal(rhs)); }


#endif // __FLOATS_HCL__
