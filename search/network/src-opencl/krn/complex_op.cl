#include <complex_op.h.cl>

double2 cbuild(const double real, const double imag) { return (double2)(real, imag); }

double creal(const double2 comp) { return comp.x; }
double cimag(const double2 comp) { return comp.y; }

double2 cmulcc(const double2 lhs, const double2 rhs) { return cbuild(lhs.x * rhs.x - lhs.y * rhs.y,
                                                                     lhs.y * rhs.x + lhs.x * rhs.y); }
double2 cmulcr(const double2 lhs, const double rhs) { return cbuild(lhs.x * rhs, lhs.y * rhs); }
double2 cmulrc(const double lhs, const double2 rhs) { return cbuild(lhs * rhs.x, lhs * rhs.y); }

double2 cdivcc(const double2 lhs, const double2 rhs)
{
    double denom = (creal(rhs)*creal(rhs) + cimag(rhs)*cimag(rhs));
    return cbuild((creal(lhs)*creal(rhs) + cimag(lhs)*cimag(rhs)) / denom,
                  (cimag(lhs)*creal(rhs) - creal(lhs)*cimag(rhs)) / denom);
}
double2 cdivcr(const double2 lhs, const double rhs)
{
    return cbuild(creal(lhs) / rhs,
                  cimag(lhs) / rhs);
}
double2 cdivrc(const double lhs, const double2 rhs)
{
    double denom = (creal(rhs)*creal(rhs) + cimag(rhs)*cimag(rhs));
    return cbuild((lhs*creal(rhs)) / denom,
                  (-lhs * cimag(rhs)) / denom);
}





float2 fcbuild(const float real, const float imag) { return (float2)(real, imag); }

float fcreal(const float2 comp) { return comp.x; }
float fcimag(const float2 comp) { return comp.y; }

float2 fcmulcc(const float2 lhs, const float2 rhs) { return fcbuild(lhs.x * rhs.x - lhs.y * rhs.y,
                                                                   lhs.y * rhs.x + lhs.x * rhs.y); }
float2 fcmulcr(const float2 lhs, const float rhs) { return fcbuild(lhs.x * rhs, lhs.y * rhs); }
float2 fcmulrc(const float lhs, const float2 rhs) { return fcbuild(lhs * rhs.x, lhs * rhs.y); }

float2 fcdivcc(const float2 lhs, const float2 rhs)
{
    float denom = (fcreal(rhs)*fcreal(rhs) + fcimag(rhs)*fcimag(rhs));
    return fcbuild((fcreal(lhs)*fcreal(rhs) + fcimag(lhs)*fcimag(rhs)) / denom,
                  (fcimag(lhs)*fcreal(rhs) - fcreal(lhs)*fcimag(rhs)) / denom);
}
float2 fcdivcr(const float2 lhs, const float rhs)
{
    return fcbuild(fcreal(lhs) / rhs,
                  fcimag(lhs) / rhs);
}
float2 fcdivrc(const float lhs, const float2 rhs)
{
    float denom = (fcreal(rhs)*fcreal(rhs) + fcimag(rhs)*fcimag(rhs));
    return fcbuild((lhs*fcreal(rhs)) / denom,
                   (-lhs * fcimag(rhs)) / denom);
}
