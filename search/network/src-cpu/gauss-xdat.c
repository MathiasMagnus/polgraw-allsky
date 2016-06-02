#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h>

// ./gauss-xdat 86164 3 1 ../testdata/001/H1/xdatc_001_0666.bin

int main(int argc, char **argv) { 

  int i, N; 
  double *x, amp, sigma; 

  N = atoi(argv[1]); 
  amp = atof(argv[2]); 
  sigma = atof(argv[3]);  
 
  x = (double *)calloc(N, sizeof(double));   

	FILE *dataout;  
  dataout = fopen(argv[4], "wb"); 

  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
 
  // Generate normal distribution (around 0, 
  // with amplitude amp and sigma)
  for(i=0; i<N; i++) 
    x[i] = amp*gsl_ran_gaussian(r, sigma);
 
  gsl_rng_free(r); 

  fwrite(x, sizeof(*x), N, dataout);  
  fclose(dataout); 
    
	return 0; 

}

