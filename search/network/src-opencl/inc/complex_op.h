#ifndef __COMPLEX_OP_H__
#define __COMPLEX_OP_H__

// Standard C includes
#include <complex.h>      // _Fcomplex, _Dcomplex

// Requires external specification of COMP_DOUBLE

#if COMP_DOUBLE
#define real_t double
  #ifdef _WIN32
  #define complex_t _Dcomplex
  #else
  #define complex_t complex double
  #endif
#else
#define real_t float
  #ifdef _WIN32
  #define complex_t _Fcomplex
  #else
  #define complex_t complex float
  #endif
#endif

#if COMP_DOUBLE

complex_t cbuild(const real_t real, const real_t imag) { return _Cbuild(real, imag); }

complex_t cmulcc(const complex_t lhs, const complex_t rhs) { return _Cmulcc(lhs, rhs); }
complex_t cmulcr(const complex_t lhs, const real_t rhs) { return _Cmulcr(lhs, rhs); }
complex_t cmulrc(const real_t lhs, const complex_t rhs) { return _Cmulcr(rhs, lhs); }

#else

complex_t cbuild(const real_t real, const real_t imag) { return _FCbuild(real, imag); }

complex_t cmulcc(const complex_t lhs, const complex_t rhs) { return _FCmulcc(lhs, rhs); }
complex_t cmulcr(const complex_t lhs, const real_t rhs) { return _FCmulcr(lhs, rhs); }
complex_t cmulrc(const real_t lhs, const complex_t rhs) { return _FCmulcr(rhs, lhs); }

#endif

complex_t cdivcc(const complex_t lhs, const complex_t rhs)
{
  real_t denom = (fcreal(rhs)*fcreal(rhs) + fcimag(rhs)*fcimag(rhs));
  return cbuild((fcreal(lhs)*fcreal(rhs) + fcimag(lhs)*fcimag(rhs)) / denom,
                (fcimag(lhs)*fcreal(rhs) - fcreal(lhs)*fcimag(rhs)) / denom);
}
complex_t cdivcr(const complex_t lhs, const real_t rhs)
{
  return cbuild(fcreal(lhs) / rhs,
                fcimag(lhs) / rhs);
}
complex_t cdivrc(const real_t lhs, const complex_t rhs)
{
  real_t denom = (fcreal(rhs)*fcreal(rhs) + fcimag(rhs)*fcimag(rhs));
  return cbuild((lhs*fcreal(rhs)) / denom,
                (-lhs * fcimag(rhs)) / denom);
}

complex_t caddcc(const complex_t lhs, const complex_t rhs) { return cbuild(fcreal(lhs) + fcreal(rhs), fcimag(lhs) + fcimag(rhs)); }
complex_t caddcr(const complex_t lhs, const real_t rhs) { return cbuild(fcreal(lhs) + rhs, fcimag(lhs)); }
complex_t caddrc(const real_t lhs, const complex_t rhs) { return cbuild(lhs + fcreal(rhs), fcimag(rhs)); }

complex_t csubcc(const complex_t lhs, const complex_t rhs) { return cbuild(fcreal(lhs) - fcreal(rhs), fcimag(lhs) - fcimag(rhs)); }
complex_t csubcr(const complex_t lhs, const real_t rhs) { return cbuild(fcreal(lhs) - rhs, fcimag(lhs)); }
complex_t csubrc(const real_t lhs, const complex_t rhs) { return cbuild(lhs - fcreal(rhs), fcimag(rhs)); }

#undef real_t
#undef complex_t

#endif // __COMPLEX_OP_H__
