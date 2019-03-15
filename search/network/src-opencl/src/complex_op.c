#ifdef _MSC_VER
#include <complex_op.h>

_Dcomplex cbuild(const double real, const double imag) { return _Cbuild(real, imag); }
//double creal(_Dcomplex cplx) { return cplx; }
//double cimag(_Dcomplex cplx) { return cplx; }

_Dcomplex cmulcc(const _Dcomplex lhs, const _Dcomplex rhs) { return _Cmulcc(lhs, rhs); }
_Dcomplex cmulcr(const _Dcomplex lhs, const double rhs) { return _Cmulcr(lhs, rhs); }
_Dcomplex cmulrc(const double lhs, const _Dcomplex rhs) { return _Cmulcr(rhs, lhs); }

_Dcomplex cdivcc(const _Dcomplex lhs, const _Dcomplex rhs)
{
    double denom = (creal(rhs)*creal(rhs) + cimag(rhs)*cimag(rhs));
    return cbuild((creal(lhs)*creal(rhs) + cimag(lhs)*cimag(rhs)) / denom,
                  (cimag(lhs)*creal(rhs) - creal(lhs)*cimag(rhs)) / denom);
}
_Dcomplex cdivcr(const _Dcomplex lhs, const double rhs)
{
    return cbuild(creal(lhs) / rhs,
                  cimag(lhs) / rhs);
}
_Dcomplex cdivrc(const double lhs, const _Dcomplex rhs)
{
    double denom = (creal(rhs)*creal(rhs) + cimag(rhs)*cimag(rhs));
    return cbuild((lhs*creal(rhs)) / denom,
                  (-lhs * cimag(rhs)) / denom);
}

_Dcomplex caddcc(const _Dcomplex lhs, const _Dcomplex rhs) { return cbuild(creal(lhs) + creal(rhs), cimag(lhs) + cimag(rhs)); }
_Dcomplex caddcr(const _Dcomplex lhs, const double rhs) { return cbuild(creal(lhs) + rhs, cimag(lhs)); }
_Dcomplex caddrc(const double lhs, const _Dcomplex rhs) { return cbuild(lhs + creal(rhs), cimag(rhs)); }

_Dcomplex csubcc(const _Dcomplex lhs, const _Dcomplex rhs) { return cbuild(creal(lhs) - creal(rhs), cimag(lhs) - cimag(rhs)); }
_Dcomplex csubcr(const _Dcomplex lhs, const double rhs) { return cbuild(creal(lhs) - rhs, cimag(lhs)); }
_Dcomplex csubrc(const double lhs, const _Dcomplex rhs) { return cbuild(lhs - creal(rhs), cimag(rhs)); }




_Fcomplex fcbuild(const float real, const float imag) { return _FCbuild(real, imag); }

_Fcomplex fcmulcc(const _Fcomplex lhs, const _Fcomplex rhs) { return _FCmulcc(lhs, rhs); }
_Fcomplex fcmulcr(const _Fcomplex lhs, const float rhs) { return _FCmulcr(lhs, rhs); }
_Fcomplex fcmulrc(const float lhs, const _Fcomplex rhs) { return _FCmulcr(rhs, lhs); }

_Fcomplex fcdivcc(const _Fcomplex lhs, const _Fcomplex rhs)
{
    float denom = (crealf(rhs)*crealf(rhs) + cimagf(rhs)*cimagf(rhs));
    return fcbuild((crealf(lhs)*crealf(rhs) + cimagf(lhs)*cimagf(rhs)) / denom,
                   (cimagf(lhs)*crealf(rhs) - crealf(lhs)*cimagf(rhs)) / denom);
}
_Fcomplex fcdivcr(const _Fcomplex lhs, const float rhs)
{
    return fcbuild(crealf(lhs) / rhs,
                   cimagf(lhs) / rhs);
}
_Fcomplex fcdivrc(const float lhs, const _Fcomplex rhs)
{
    float denom = (crealf(rhs)*crealf(rhs) + cimagf(rhs)*cimagf(rhs));
    return fcbuild((lhs*crealf(rhs)) / denom,
                   (-lhs * cimagf(rhs)) / denom);
}

_Fcomplex fcaddcc(const _Fcomplex lhs, const _Fcomplex rhs) { return fcbuild(crealf(lhs) + crealf(rhs), cimagf(lhs) + cimagf(rhs)); }
_Fcomplex fcaddcr(const _Fcomplex lhs, const float rhs) { return fcbuild(crealf(lhs) + rhs, cimagf(lhs)); }
_Fcomplex fcaddrc(const float lhs, const _Fcomplex rhs) { return fcbuild(lhs + crealf(rhs), cimagf(rhs)); }

_Fcomplex fcsubcc(const _Fcomplex lhs, const _Fcomplex rhs) { return fcbuild(crealf(lhs) - crealf(rhs), cimagf(lhs) - cimagf(rhs)); }
_Fcomplex fcsubcr(const _Fcomplex lhs, const float rhs) { return fcbuild(crealf(lhs) - rhs, cimagf(lhs)); }
_Fcomplex fcsubrc(const float lhs, const _Fcomplex rhs) { return fcbuild(lhs - crealf(rhs), cimagf(rhs)); }
#endif

