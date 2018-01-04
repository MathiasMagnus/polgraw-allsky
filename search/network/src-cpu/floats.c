// Polgraw includes
#include <floats.h>


#ifdef _WIN32

#ifdef COMP_FLOAT

complex_t cmulcc(const complex_t lhs, const complex_t rhs) { return _FCmulcc(lhs, rhs); }
complex_t cmulcr(const complex_t lhs, const real_t rhs) { return _FCmulcr(lhs, rhs); }

#else

complex_t cbuild(const real_t real, const real_t imag) { return _Cbuild(real, imag); }

complex_t cmulcc(const complex_t lhs, const complex_t rhs) { return _Cmulcc(lhs, rhs); }
complex_t cmulcr(const complex_t lhs, const real_t rhs) { return _Cmulcr(lhs, rhs); }
complex_t cmulrc(const real_t lhs, const complex_t rhs) { return _Cmulcr(rhs, lhs); }

complex_t cdivcc(const complex_t lhs, const complex_t rhs)
{
    real_t denom = (creal(rhs)*creal(rhs) + cimag(rhs)*cimag(rhs));
    return cbuild((creal(lhs)*creal(rhs) + cimag(lhs)*cimag(rhs)) / denom,
                  (cimag(lhs)*creal(rhs) - creal(lhs)*cimag(rhs)) / denom);
}
complex_t cdivcr(const complex_t lhs, const real_t rhs)
{
    return cbuild(creal(lhs) / rhs,
                  cimag(lhs) / rhs);
}
complex_t cdivrc(const real_t lhs, const complex_t rhs)
{
    real_t denom = (creal(rhs)*creal(rhs) + cimag(rhs)*cimag(rhs));
    return cbuild((lhs*creal(rhs)) / denom,
                  (-lhs*cimag(rhs)) / denom);
}

complex_t caddcc(const complex_t lhs, const complex_t rhs) { return cbuild(creal(lhs) + creal(rhs), cimag(lhs) + cimag(rhs)); }
complex_t caddcr(const complex_t lhs, const real_t rhs) { return cbuild(creal(lhs) + rhs, cimag(lhs)); }
complex_t caddrc(const real_t lhs, const complex_t rhs) { return cbuild(lhs + creal(rhs), cimag(rhs)); }

complex_t csubcc(const complex_t lhs, const complex_t rhs) { return cbuild(creal(lhs) - creal(rhs), cimag(lhs) - cimag(rhs)); }
complex_t csubcr(const complex_t lhs, const real_t rhs) { return cbuild(creal(lhs) - rhs, cimag(lhs)); }
complex_t csubrc(const real_t lhs, const complex_t rhs) { return cbuild(lhs - creal(rhs), cimag(rhs)); }

#endif

#endif