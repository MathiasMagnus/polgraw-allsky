#include <log.h>

#define COMP_DOUBLE 1
#ifdef _MSC_VER
#include <complex_op.h>
#endif

// Standard C includes
#include <float.h>      // DBL_MAX
#include <stdio.h>      // fopen
#include <math.h>       // fcabs
#include <string.h>     // strcpy, strcat

void print_complex_min_max_dp(complex_double* arr, size_t N, const char* msg)
{
  double min = DBL_MAX,
         max = 0;
  for (size_t i = 0; i < N; ++i)
  {
    if (cabs(arr[i]) < min) min = cabs(arr[i]);
    if (cabs(arr[i]) > max) max = cabs(arr[i]);
  }
  printf("%s\tMin: %f\nMax: %f\n", msg, min, max);
}

void print_complex_array_dp(complex_double* arr, size_t count, const char* msg)
{
#ifdef _WIN32
  int bytes = printf_s("%s:\n\n", msg);
  size_t i;
  for (i = 0; i < count; ++i) { bytes = printf_s("\t{%f,%f}\n", creal(arr[i]), cimag(arr[i])); }
  bytes = printf_s("\n");
#else
  printf("%s:\n\n", msg);
  size_t i;
  for (i = 0; i < count; ++i) { printf("\t{%f,%f}\n", creal(arr[i]), cimag(arr[i])); }
  printf("\n");
#endif
  fflush(NULL);
}

void save_complex_array_dp(complex_double* arr, size_t count, const char* name)
{
  char filename[1024];

  strcpy(filename, "cl_");
  strcat(filename, name);
  strcat(filename, ".dat");

  FILE* fc = fopen(filename, "w");
  if (fc == NULL) perror("Failed to open output file.");

  size_t i;
  for (i = (size_t)0; i < count; ++i)
      //fprintf(fc, "%lf %lf\n", creal(arr[i]), cimag(arr[i]));
      //fprintf(fc, "%zu %e + i %e\n", i, creal(arr[i]), cimag(arr[i]));
      //fprintf(fc, "(%e,%e)\n", creal(arr[i]), cimag(arr[i]));
      fprintf(fc, "%e %e\n", creal(arr[i]), cimag(arr[i]));

  int close = fclose(fc);
  if (close == EOF) perror("Failed to close output file.");
}
