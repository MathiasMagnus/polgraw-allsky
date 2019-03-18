#include <log.h>

#define COMP_DOUBLE 0
#ifdef _MSC_VER
#include <complex_op.h>
#endif

// Standard C includes
#include <float.h>      // FLT_MAX
#include <stdio.h>      // fopen
#include <math.h>       // fcabs
#include <string.h>     // strcpy, strcat

void print_complex_min_max_sp(complex_float* arr, size_t N, const char* msg)
{
  float min = FLT_MAX,
        max = 0;
  for (size_t i = 0; i < N; ++i)
  {
    if (cabsf(arr[i]) < min) min = cabsf(arr[i]);
    if (cabsf(arr[i]) > max) max = cabsf(arr[i]);
  }
  printf("%s\tMin: %f\nMax: %f\n", msg, min, max);
}

void print_complex_array_sp(complex_float* arr, size_t count, const char* msg)
{
#ifdef _WIN32
  int bytes = printf_s("%s:\n\n", msg);
  size_t i;
  for (i = 0; i < count; ++i) { bytes = printf_s("\t{%f,%f}\n", crealf(arr[i]), cimagf(arr[i])); }
  bytes = printf_s("\n");
#else
  printf("%s:\n\n", msg);
  size_t i;
  for (i = 0; i < count; ++i) { printf("\t{%f,%f}\n", crealf(arr[i]), cimagf(arr[i])); }
  printf("\n");
#endif
  fflush(NULL);
}

void save_complex_array_sp(complex_float* arr, size_t count, const char* name)
{
  char filename[1024];

  strcpy(filename, "cl_");
  strcat(filename, name);
  strcat(filename, ".dat");

  FILE* fc = fopen(filename, "w");
  if (fc == NULL) perror("Failed to open output file.");

  size_t i;
  for (i = (size_t)0; i < count; ++i)
      //fprintf(fc, "%lf %lf\n", crealf(arr[i]), cimagf(arr[i]));
      //fprintf(fc, "%zu %e + i %e\n", i, crealf(arr[i]), cimagf(arr[i]));
      //fprintf(fc, "(%e,%e)\n", crealf(arr[i]), cimagf(arr[i]));
      fprintf(fc, "%e %e\n", crealf(arr[i]), cimagf(arr[i]));

  int close = fclose(fc);
  if (close == EOF) perror("Failed to close output file.");
}
