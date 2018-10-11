#ifndef __COMPLEX_OP_H_CL__
#define __COMPLEX_OP_H_CL__

// Requires external specification of COMP_DOUBLE

#if COMP_DOUBLE
#define real_t double
#define complex_t double2
#else
#define real_t float
#define complex_t float2
#endif


real_t creal(const complex_t complex) { return complex.x; }
real_t cimag(const complex_t complex) { return complex.y; }

complex_t cbuild(const real_t real, const real_t imag) { return (complex_t)(real, imag); }

complex_t cmulcc(const complex_t lhs, const complex_t rhs) { return cbuild(lhs.x * rhs.x - lhs.y * rhs.y,
                                                                           lhs.y * rhs.x + lhs.x * rhs.y); }
complex_t cmulcr(const complex_t lhs, const real_t rhs) { return cbuild(lhs.x * rhs, lhs.y * rhs); }
complex_t cmulrc(const real_t lhs, const complex_t rhs) { return cbuild(lhs * rhs.x, lhs * rhs.y); }

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
        (-lhs * cimag(rhs)) / denom);
}

complex_t caddcc(const complex_t lhs, const complex_t rhs) { return cbuild(creal(lhs) + creal(rhs), cimag(lhs) + cimag(rhs)); }
complex_t caddcr(const complex_t lhs, const real_t rhs) { return cbuild(creal(lhs) + rhs, cimag(lhs)); }
complex_t caddrc(const real_t lhs, const complex_t rhs) { return cbuild(lhs + creal(rhs), cimag(rhs)); }

complex_t csubcc(const complex_t lhs, const complex_t rhs) { return cbuild(creal(lhs) - creal(rhs), cimag(lhs) - cimag(rhs)); }
complex_t csubcr(const complex_t lhs, const real_t rhs) { return cbuild(creal(lhs) - rhs, cimag(lhs)); }
complex_t csubrc(const real_t lhs, const complex_t rhs) { return cbuild(lhs - creal(rhs), cimag(rhs)); }

#undef real_t
#undef complex_t

#endif // __COMPLEX_OP_H_CL__
