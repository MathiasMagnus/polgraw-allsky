#ifndef __COMPLEX_OP_H__
#define __COMPLEX_OP_H__

// Standard C includes
#define _CRT_USE_C_COMPLEX_H
#include <complex.h>      // _Fcomplex, _Dcomplex


_Dcomplex cbuild(const double real, const double imag);
//double creal(_Dcomplex cplx);
//double cimag(_Dcomplex cplx);

_Dcomplex cmulcc(const _Dcomplex lhs, const _Dcomplex rhs);
_Dcomplex cmulcr(const _Dcomplex lhs, const double rhs);
_Dcomplex cmulrc(const double lhs, const _Dcomplex rhs);

_Dcomplex cdivcc(const _Dcomplex lhs, const _Dcomplex rhs);
_Dcomplex cdivcr(const _Dcomplex lhs, const double rhs);
_Dcomplex cdivrc(const double lhs, const _Dcomplex rhs);

_Dcomplex caddcc(const _Dcomplex lhs, const _Dcomplex rhs);
_Dcomplex caddcr(const _Dcomplex lhs, const double rhs);
_Dcomplex caddrc(const double lhs, const _Dcomplex rhs);

_Dcomplex csubcc(const _Dcomplex lhs, const _Dcomplex rhs);
_Dcomplex csubcr(const _Dcomplex lhs, const double rhs);
_Dcomplex csubrc(const double lhs, const _Dcomplex rhs);




_Fcomplex fcbuild(const float real, const float imag);
//float creal(_Fcomplex cplx);
//float cimag(_Fcomplex cplx);

_Fcomplex fcmulcc(const _Fcomplex lhs, const _Fcomplex rhs);
_Fcomplex fcmulcr(const _Fcomplex lhs, const float rhs);
_Fcomplex fcmulrc(const float lhs, const _Fcomplex rhs);

_Fcomplex fcdivcc(const _Fcomplex lhs, const _Fcomplex rhs);
_Fcomplex fcdivcr(const _Fcomplex lhs, const float rhs);
_Fcomplex fcdivrc(const float lhs, const _Fcomplex rhs);

_Fcomplex fcaddcc(const _Fcomplex lhs, const _Fcomplex rhs);
_Fcomplex fcaddcr(const _Fcomplex lhs, const float rhs);
_Fcomplex fcaddrc(const float lhs, const _Fcomplex rhs);

_Fcomplex fcsubcc(const _Fcomplex lhs, const _Fcomplex rhs);
_Fcomplex fcsubcr(const _Fcomplex lhs, const float rhs);
_Fcomplex fcsubrc(const float lhs, const _Fcomplex rhs);


#endif // __COMPLEX_OP_H__
