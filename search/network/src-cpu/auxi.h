#ifndef __AUXI_H__
#define __AUXI_H__

// Polgraw includes
#include <floats.h>

#include <complex.h>
#include <stddef.h>

#define sqr(x) ((x)*(x))
#define TOSTRA(x) #x
#define TOSTR(x) TOSTRA(x)

#define TINY 1.0e-20
#define NINTERP 3			 /* degree of the interpolation polynomial */
							 /* Do not change this value!!! */
#define NAVFSTAT 4096
//#define round(x) floor((x)+0.5)


void lin2ast(double be1, double be2, int pm, double sepsm, double cepsm,	\
         double *sinal, double *cosal, double *sindel, double *cosdel);

void ast2lin(real_t alfa, real_t delta, double epsm, double *be);
 
void spline(complex_t *, int, complex_t *);
complex_t splint (complex_t *, complex_t *, int, double);
void splintpad (complex_t *, real_t *, int, int, complex_t*);
double var (double *, int);

void gridr (double *, int *, int *, int *, double, double);
double FStat (double *, int, int, int);

int ludcmp (double *, int, int *, double *);
int lubksb (double *, int, int *, double *);

// gridopt 
int invm (const double *, int, double *);
double det (const double *, int);

// for qsorting the lines 
int compared2c (const void *, const void *);

/// <summary>Prints the largest absolute value of a host side complex array.</summary>
///
void print_complex_min_max(complex_t* arr, size_t N, const char* msg);

/// <summary>Prints the first 'n' values of a host side real array.</summary>
///
void print_real_array(real_t* arr, size_t count, const char* msg);

/// <summary>Prints the first 'n' values of a host side complex array.</summary>
///
void print_complex_array(complex_t* arr, size_t count, const char* msg);

/// <summary>Saves values of a host side real array to disk.</summary>
///
void save_real_array(real_t* arr, size_t count, const char* filename);

/// <summary>Saves values of a host side complex array to disk.</summary>
///
void save_complex_array(complex_t* arr, size_t count, const char* filename);

#endif
