#ifndef __COMPLEX_OP_H_CL__
#define __COMPLEX_OP_H_CL__


double2 cbuild(const double real, const double imag);

double creal(const double2 comp);
double cimag(const double2 comp);

double2 cmulcc(const double2 lhs, const double2 rhs);
double2 cmulcr(const double2 lhs, const double rhs);
double2 cmulrc(const double lhs, const double2 rhs);

double2 cdivcc(const double2 lhs, const double2 rhs);
double2 cdivcr(const double2 lhs, const double rhs);
double2 cdivrc(const double lhs, const double2 rhs);




float2 fcbuild(const float real, const float imag);

float fcreal(const float2 comp);
float fcimag(const float2 comp);

float2 fcmulcc(const float2 lhs, const float2 rhs);
float2 fcmulcr(const float2 lhs, const float rhs);
float2 fcmulrc(const float lhs, const float2 rhs);

float2 fcdivcc(const float2 lhs, const float2 rhs);
float2 fcdivcr(const float2 lhs, const float rhs);
float2 fcdivrc(const float lhs, const float2 rhs);

#endif // __COMPLEX_OP_H_CL__
