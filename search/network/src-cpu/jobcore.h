#ifndef __JOBCORE_H__
#define __JOBCORE_H__

// Polgraw includes
#include <struct.h>     // Search_settings, Command_line_opts, Search_range, FFT_plans, FFT_arrays, Aux_arrays
#include <floats.h>     // FLOAT_TYPE, HOST_COMPLEX_TYPE

void search(
	    Search_settings *sett,
	    Command_line_opts *opts,
	    Search_range *s_range,
	    FFTW_plans *plans,
	    FFTW_arrays *fftw_arr,
	    Aux_arrays *aux,
	    int *Fnum,
	    double *F);

/* Main job function
 * The output is stored in single or double precision 
 * (FLOAT_TYPE defined in struct.h)  
 */ 

int job_core(
	     int pm,                   // hemisphere
	     int mm,                   // grid 'sky position'
	     int nn,                   // other grid 'sky position'
	     Search_settings *sett,    // search settings
	     Command_line_opts *opts,  // cmd opts
	     Search_range *s_range,    // range for searching
	     FFTW_plans *plans,        // plans for fftw
	     FFTW_arrays *fftw_arr,    // arrays for fftw
	     Aux_arrays *aux,          // auxiliary arrays
	     double *F,                // F-statistics array
	     int *sgnlc,               // current number of candidates
	     FLOAT_TYPE *sgnlv,        // candidate array
	     int *FNum);               // candidate signal number

#endif
