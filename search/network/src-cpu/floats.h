#ifndef __FLOATS_H__
#define __FLOATS_H__

// Standard C includes
#include <complex.h>    // _Dcomplex

//changing computations in spindown loop to single-precision arithmetic
#undef COMP_FLOAT

#ifdef _WIN32

#ifdef COMP_FLOAT

typedef float real_t;
typedef _Fcomplex complex_t;
#define FLOAT_TYPE float
#define COMPLEX_TYPE _Fcomplex

complex_t cmulcc(const complex_t lhs, const complex_t rhs);
complex_t cmulcr(const complex_t lhs, const real_t rhs);

#else

typedef double real_t;
typedef _Dcomplex complex_t;
#define FLOAT_TYPE double
#define COMPLEX_TYPE _Dcomplex

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

//complex_t cpown(const complex_t base, const unsigned int exponent)
//{
//    complex_t res = base;
//
//    for(int i = 0; i < exponent - 1; ++i)
//        res = cmulcc(base, res);
//
//    return res;
//}
#endif // COMP_FLOAT

#else

#ifdef COMP_FLOAT
typedef float real_t;
typedef complex float complex_t;
#define FLOAT_TYPE float
#define COMPLEX_TYPE complex float
#else
typedef double real_t;
typedef complex double complex_t;
#define FLOAT_TYPE double
#define COMPLEX_TYPE complex double
#endif // COMP_FLOAT

#endif // _WIN32


#endif
