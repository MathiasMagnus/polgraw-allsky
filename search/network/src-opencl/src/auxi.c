// C behavioral defines
//
// MSVC: macro to include constants, such as M_PI (include before math.h)
#define _USE_MATH_DEFINES

// Polgraw includes
#include <auxi.h>

// Standard C includes
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <complex.h>
#include <float.h>


/// <summary>Change linear (grid) coordinates to real coordinates</summary>
/// <remarks>lin2ast described in Phys. Rev. D 82, 022005 (2010) (arXiv:1003.0844)</remarks>
///
void lin2ast(real_t be1, real_t be2, int pm, real_t sepsm, real_t cepsm,
             real_t *sinal, real_t *cosal, real_t *sindel, real_t *cosdel)
{

  *sindel = be1*sepsm-(2*pm-3)*sqrt(1.-sqr(be1)-sqr(be2))*cepsm;
  *cosdel = sqrt(1.-sqr(*sindel));
  *sinal = (be1-sepsm*(*sindel))/(cepsm*(*cosdel));
  *cosal = be2/(*cosdel);
  
} // lin2ast


/// <summary>Returns the variance (square of the standard deviation) of a given vector <c>x</c> of length <c>n</c></summary>
///
double var (double *x, int n)
{
  int i;
  double mean=0., variance=0.;

  for (i=0; i<n; i++)
    mean += x[i];
  mean /= n;
  for (i=0; i<n; i++)
    variance += sqr (x[i]-mean);
  variance /= (n-1);
  return variance;
} /* var() */


/// <summary>Establish the grid range in which the search will be performed with the use of the M matrix from grid.bin</summary>
/// 
void gridr(double *M,
           int *spndr,
           int *nr,
           int *mr,
           double oms,
           double Smax)
{
    double cof, Mp[16], smx[64], d, Ob;
    int i, j, indx[4];

    // Grid range //

    // input:
    // *M - pointer to the array that generates the grid
    // *spndr - pointer to the range of spindowns in grid units
    // i.e., integer numbers
    // *nr and *mr - pointers to the range of sky positions
    // in grid units i.e., integer numbers

    // from settings() :
    // maximal value of the spindown:
    // Smax = 2.*M_PI*(fpo+B)*dt*dt/(2.*tau_min)
    //
    // oms equals 2.*M_PI*fpo*dt

    Ob = M_PI;
    cof = oms + Ob;

    //Mp - macierz transponowana
    for (i = 0; i<4; i++)
        for (j = 0; j<4; j++)
            Mp[4 * i + j] = M[4 * j + i];

    ludcmp(Mp, 4, indx, &d);

    for (i = 0; i<8; i++)
    {
        smx[8 * i + 2] = cof;
        smx[8 * i + 6] = -cof;
    }
    for (i = 0; i<4; i++)
    {
        smx[16 * i + 3] = smx[16 * i + 7] = cof;
        smx[16 * i + 11] = smx[16 * i + 15] = -cof;
    }
    for (i = 0; i<8; i++)
    {
        smx[4 * i] = Ob;
        smx[4 * i + 32] = -Ob;
    }
    for (i = 0; i<2; i++)
        for (j = 0; j<4; j++)
        {
            smx[32 * i + 4 * j + 1] = -Smax;
            smx[32 * i + 4 * j + 17] = 0.;
        }
    for (i = 0; i<16; i++)
        lubksb(Mp, 4, indx, smx + 4 * i);

    spndr[0] = nr[0] = mr[0] = 16384;
    spndr[1] = nr[1] = mr[1] = -16384;

    // Explicit casts from floor/ceil silences warning (programmer states intent)
    for (i = 0; i<16; i++)
    {
        if (floor(smx[4 * i + 1]) < spndr[0])
            spndr[0] = (int)floor(smx[4 * i + 1]);
        if (ceil(smx[4 * i + 1]) > spndr[1])
            spndr[1] = (int)ceil(smx[4 * i + 1]);

        if (floor(smx[4 * i + 2]) < nr[0])
            nr[0] = (int)floor(smx[4 * i + 2]);
        if (ceil(smx[4 * i + 2]) > nr[1])
            nr[1] = (int)ceil(smx[4 * i + 2]);

        if (floor(smx[4 * i + 3]) < mr[0])
            mr[0] = (int)floor(smx[4 * i + 3]);
        if (ceil(smx[4 * i + 3]) > mr[1])
            mr[1] = (int)ceil(smx[4 * i + 3]);
    }

} // gridr()

double FStat (double *F, int nfft, int nav, int indx) {
  /* FStat Smoothed F-statistic */

  // input:
  // *F - pointer to the value of F statistic
  // nfft - the length of the FFT data
  // nav	- length of the block (nfft/nav is the number of blocks)
  // indx - block index

  int i, j;
  double mu, *fr, pxout=0.;

  indx /= nav;
  fr = F;
  for (j=0; j<nfft/nav; j++) {
    mu = 0.;
    for (i=0; i<nav; i++)
      mu += *fr++;
    mu /= 2.*nav;
    if (j == indx)
      pxout = mu;
    fr -= nav;
    for (i=0; i<nav; i++)
      *fr++ /= mu;
  } /* for j */
  return pxout;
} /* FStat() */

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
           double *d)
{
    int i, imax = -1, j, k;
    double big, dum, sum, temp;
    double *vv;

    vv = (double*)calloc(n, sizeof(double));
    *d = 1.0;

    for (i = 0; i<n; i++)
    {
        big = 0.0;
        for (j = 0; j<n; j++)
            if ((temp = fabs(a[n*i + j])) > big)
                big = temp;
        if (big == 0.0)
            return 1;
        vv[i] = 1.0 / big;
    }
    for (j = 0; j<n; j++)
    {
        for (i = 0; i<j; i++)
        {
            sum = a[n*i + j];
            for (k = 0; k<i; k++)
                sum -= a[n*i + k] * a[n*k + j];
            a[n*i + j] = sum;
        }
        big = 0.0;
        for (i = j; i<n; i++)
        {
            sum = a[n*i + j];
            for (k = 0; k<j; k++)
                sum -= a[n*i + k] * a[n*k + j];
            a[n*i + j] = sum;
            if ((dum = vv[i] * fabs(sum)) >= big)
            {
                big = dum;
                imax = i;
            }
        }
        if (j != imax)
        {
            for (k = 0; k<n; k++)
            {
                dum = a[n*imax + k];
                a[n*imax + k] = a[n*j + k];
                a[n*j + k] = dum;
            }
            *d = -(*d);
            vv[imax] = vv[j];
        }
        indx[j] = imax;
        if (a[n*j + j] == 0.0)
            a[n*j + j] = TINY;
        if (j != n)
        {
            dum = 1.0 / (a[n*j + j]);
            for (i = j + 1; i<n; i++)
                a[n*i + j] *= dum;
        }
    }

    free(vv);
    return 0;

} /* ludcmp() */

int lubksb (double *a, int n, int *indx, double *b)
/* Solves the set of n linear equations A X=B.
   Input:
   a[0..n-1][0..n-1] - LU decomposition af a matrix A,
   determined by ludcmp()
   n				- number of rows and columns of a
   indx[0..n-1]		- permutation vector returned by ludcmp
   b[0..n-1]			 - right-hand side vector B
   (changed on exit)
   Output:
   b[0..n-1]			- solution vector X
*/
{
  int i, ii=-1, ip, j;
  double sum;

  for (i=0; i<n; i++) {
    ip = indx[i];
    sum = b[ip];
    b[ip] = b[i];
    if (ii>=0)
      for (j=ii; j<=i-1; j++)
	sum -= a[n*i+j]*b[j];
    else if (sum)
      ii = i;
    b[i] = sum;
  }
  for (i=n-1; i>=0; i--) {
    sum = b[i];
    for (j=i+1; j<n; j++)
      sum -= a[n*i+j]*b[j];
    b[i] = sum/a[n*i+i];
  }
  return 0;
} /* lubksb() */


int invm (const double *a, int N, double *y)
     /* Inverse of a real matrix a[0..N-1][0..N-1].
	Input:
		a[0..N-1][0..N-1] - given matrix (saved on exit)
		N	      - number of rows and columns of a
        Output:
		y[0..N-1][0..N-1] - inverse of a
     */
{
  double d, *col, *al;
  int i, j, *indx;

  al = (double *) calloc (sqr(N), sizeof (double));
  indx = (int *) calloc (N, sizeof (int));
  col = (double *) calloc (N, sizeof (double));
  for (i=0; i<sqr(N); i++)
    al[i] = a[i];
  if (ludcmp (al, N, indx, &d))
    return 1;
  for (j=0; j<N; j++) {
    for (i=0; i<N; i++)
      col[i] = 0.0;
    col[j] = 1.0;
    lubksb (al, N, indx, col);
    for (i=0; i<N; i++)
      y[N*i+j] = col[i];
  }
  free (col);
  free (indx);
  free (al);
  return 0;
} /* invm() */


double det (const double *a, int N)
     /* determinant of a real matrix a[0..N-1][0..N-1] */
{
  double d, *al;;
  int j, *indx;

  al = (double *) calloc (sqr(N), sizeof (double));
  indx = (int *) calloc (N, sizeof (int));
  for (j=0; j<sqr(N); j++)
    al[j] = a[j];
  ludcmp (al, N, indx, &d);
  for (j=0; j<N; j++)
    d *= al[N*j+j];
  free (indx);
  free (al);
  return d;
} // det()

/// <summary>Dump array to disk</summary>
///
void dumparr (const real_t* arr, const size_t length, const char* filename)
{
  FILE* fc = fopen(filename, "w");
  if (fc == NULL) perror("Failed to open output file.");
  
  size_t count = fwrite((void *)(arr), sizeof(real_t), length, fc);
  if (count < length) perror("Failed to write output file.");
  
  int close = fclose(fc);
  if (close == EOF) perror("Failed to close output file.");
}