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

/// <summary>Change linear (grid) coordinates to real coordinates</summary>
/// <remarks>lin2ast described in Phys. Rev. D 82, 022005 (2010) (arXiv:1003.0844)</remarks>
///
void lin2ast(real_t be1, real_t be2, int pm, real_t sepsm, real_t cepsm,
	         real_t *sinal, real_t *cosal, real_t *sindel, real_t *cosdel);

/// <summary>Returns the variance (square of the standard deviation) of a given vector <c>x</c> of length <c>n</c></summary>
///
double var (double *, int);

/// <summary>Establish the grid range in which the search will be performed with the use of the M matrix from grid.bin</summary>
/// 
void gridr(double *M,
           int *spndr,
           int *nr,
           int *mr,
           double oms,
           double Smax);

/// <summary>LU decomposition of a given real matrix <c>a[0..n-1][0..n-1]</c>.</summary>
/// <param name="a">An array containing elements of matrix <c>a</c> (changed on exit).</param>
/// <param name="n">Number of rows and columns of <c>a</c>.</param>
/// <param name="indx">Output row permutation effected by the partial pivoting.</param>
/// <param name="d">Output +-1 depending on whether the number of rows interchanged was even or odd, respectively.</param>
/// <returns>0 on success, 1 on error.</returns>
///
int ludcmp(double *a,
           int n,
           int *indx,
           double *d);

int lubksb (double *, int, int *, double *);

// gridopt 
int invm (const double *, int, double *);
double det (const double *, int);

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

/// <summary>Saves values of a device side real array to disk.</summary>
///
void save_real_buffer(cl_command_queue queue, cl_mem buf, int count, const char* filename);

/// <summary>Saves values of a device side complex array to disk.</summary>
///
void save_complex_buffer(cl_command_queue queue, cl_mem buf, int count, const char* filename);

#endif
