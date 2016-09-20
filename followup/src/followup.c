/* Ver. 1.0
Main followup programm. Required files
(all should be in directory 'data' - given 
as argument) :
'candidates.coi' - File with candidates from coincidences
'list.txt'
Ver. 2.0
Functions glue and neigh added!
Ver. 3.0
Mesh adaptive direct search (MADS) added (to find real
maximum)
Ver. 4.0
Simplex added
Ver. 5.0
Function neigh moved to followup.c
All vectors/arrays declarated using yeppp! library 
 = FASTER!

MS
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
//#include <fcntl.h>
#include <getopt.h>
#include <gsl/gsl_linalg.h>
#include <time.h>
#include <dirent.h>
#include <omp.h>

#include "auxi.h"
#include "settings.h"
#include "struct.h"
#include "init.h"
//#include "timer.h"

//#include "glue.h"
//#include "neigh.h"

#include <assert.h>
#if defined(SLEEF)
//#include "sleef-2.80/purec/sleef.h"
#include <sleefsimd.h>
#elif defined(YEPPP)
#include <yepLibrary.h>
#include <yepCore.h>
#include <yepMath.h>
#endif

// Default output and data directories

#ifndef PREFIX
#define PREFIX ./FSTAT_OUT
#endif

#ifndef DTAPREFIX
#define DTAPREFIX .
#endif

#define ZEPS 1e-10

//Function neigh takes candidate parameters and number of bins (as arguments) and creates grid around it.
Yep64f** neigh(double *m, double *perc, int b){ 
//	double **array;
	int rows, cols = 4;
	rows = pow((b+1),4);
	int k;
// Allocation of memory for martix
#ifdef YEPPP
    	yepLibrary_Init();
	Yep64f **array = (Yep64f**)malloc(rows*sizeof(Yep64f));
	for (k=0; k < rows; k++) array[k] = (Yep64f*)calloc(rows,sizeof(Yep64f));
    	enum YepStatus status;
#else
  	array = (double **)malloc(rows*sizeof(double *));
  	for (k=0; k < rows; k++) array[k] = (double *)calloc(cols, sizeof(double));
#endif
	double beg[4];
	double width[4];
	int i1, i2, i3, i4, j, i;
	for(j = 0; j < 4; j++) {
		width[j] = 2*perc[j]*m[j]/b;
	}
	i = 0;
	beg[0] = m[0]*(1 - perc[0]);
	for(i1 = 0; i1 < (b + 1); i1++){
		beg[1] = m[1]*(1 - perc[1]);
		for(i2 = 0; i2 < (b + 1); i2++){
			beg[2] = m[2]*(1 - perc[2]);
			for(i3 = 0; i3 < (b + 1); i3++){
				beg[3] = m[3]*(1 - perc[3]);
				for(i4 = 0; i4 < (b + 1); i4++){
					for(j = 0; j < 4; j++) {
						array[i][j] = beg[j];
					}
					beg[3] = beg[3] + width[3];
					i++;
				}
				beg[2] = beg[2] + width[2];
			}
			beg[1] = beg[1] + width[1];
		}
		beg[0] = beg[0] + width[0];
	}
	return array;

}

// Allocation of memory for martix with given number of rows and columns
double** matrix(int rows, int cols) {

  	int k;
	double **m;
  	m = (double **)malloc(rows*sizeof(double *));
  
  	for (k=0; k < rows; k++)
    		m[k] = (double *)calloc(cols, sizeof(double));
  
  	return m;
}

// Allocation of memory for vector and martix with given number of rows and columns
double * alloc_vector(int cols){
	return (double *) malloc(sizeof(double) * cols);
}
void free_vector(double * vector, int cols){
	free(vector);
}

double ** alloc_matrix(int rows, int cols){
	int i;
	double ** matrix = (double **) malloc(sizeof(double *) * rows);
	for (i = 0; i < rows; i++) matrix[i] = alloc_vector(cols);
	return matrix;
}
void free_matrix(double ** matrix, int rows, int cols){
	int i;
	for (i = 0; i < rows; i++) free_vector(matrix[i], cols);
	free(matrix);
}

// Function computes F-statistics in given point
double* Fstatnet(Search_settings *sett, double *sgnlo, double *nSource){

	int i = 0, n = 0; 
	double aatemp, bbtemp, aa = 0., bb = 0.;
	complex double exph, xasum, xbsum;	


#ifdef YEPPP
	int VLEN = sett->N;
	yepLibrary_Init();
//Yep64f *x = (Yep64f*)calloc(ARRAY_SIZE, sizeof(Yep64f));
	Yep64f *_sph = (Yep64f*)malloc(sizeof(Yep64f)*VLEN);
	Yep64f *_cph = (Yep64f*)malloc(sizeof(Yep64f)*VLEN);
	Yep64f *phase = (Yep64f*)malloc(sizeof(Yep64f)*VLEN); 
	Yep64f *fstat_out = (Yep64f*)malloc(sizeof(Yep64f)*11); 
	enum YepStatus status;
#endif

//From jobcore.c, line 237 
//Loop for each detector 1

	xasum = 0 - I * 0;
	xbsum = 0 - I * 0;
  	for(n=0; n < sett->nifo; ++n) { 

// Calculate detector positions with respect to baricenter
// Copied from jobcore.c, line 248

//shft & phase loop
    		for(i=0; i < sett->N; ++i) {

      			ifo[n].sig.shft[i] = nSource[0]*ifo[n].sig.DetSSB[i*3]
		         	+ nSource[1]*ifo[n].sig.DetSSB[i*3+1]
		         	+ nSource[2]*ifo[n].sig.DetSSB[i*3+2];
    
// Phase modulation function
// Copied from jobcore.c, line 265

			phase[i] = sgnlo[0]*(double)(i + ifo[n].sig.shft[i]) 
				+ (sgnlo[1]*i*i) + ((sett->oms 
				+ 2.*sgnlo[1]*i)*ifo[n].sig.shft[i]);
		} //shft & phase			

//Sin & Cos calculations using Yeppp!

		status = yepMath_Cos_V64f_V64f(phase, _cph, VLEN);
		assert(status == YepStatusOk);
		status = yepMath_Sin_V64f_V64f(phase, _sph, VLEN);
		assert(status == YepStatusOk);

// Matched filter 
// Copied from jobcore.c, line 276 and 337

		for (i = 0; i < sett->N; ++i){

	      		exph = _cph[i] - I * _sph[i];

	      		ifo[n].sig.xDatma[i] = ifo[n].sig.xDat[i]*ifo[n].sig.aa[i]*exph;
	      		ifo[n].sig.xDatmb[i] = ifo[n].sig.xDat[i]*ifo[n].sig.bb[i]*exph;
	
		}
	} //End of detector loop 1

//Loop for each detector 2

  	for(n=0; n < sett->nifo; ++n) { 
		aatemp = 0.;
		bbtemp = 0.;
		for(i = 0; i < sett->N; i++){

			aatemp += sqr(ifo[n].sig.aa[i]);
			bbtemp += sqr(ifo[n].sig.bb[i]);
		}
		for(i=0; i < sett->N; ++i) {
			ifo[n].sig.xDatma[i] /= ifo[n].sig.sig2;
			ifo[n].sig.xDatmb[i] /= ifo[n].sig.sig2;

			xasum += ifo[n].sig.xDatma[i];
			xbsum += ifo[n].sig.xDatmb[i];
		}
		aa += aatemp/ifo[n].sig.sig2; 
		bb += bbtemp/ifo[n].sig.sig2;   

	}// End of detector loop 2

// F - statistic
	fstat_out[5] = - ((( sqr(creal(xasum)) + sqr(cimag(xasum)))/aa)
			+ ((sqr(creal(xbsum)) + sqr(cimag(xasum)))/bb));

// Amplitude estimates
	fstat_out[0] = 2*creal(xasum)/aa;
	fstat_out[1] = 2*creal(xbsum)/bb;
	fstat_out[2] = -2*cimag(xasum)/aa;
	fstat_out[3] = -2*cimag(xbsum)/bb;

// Signal-to-noise ratio
	fstat_out[4] = sqrt(2*(-fstat_out[5]-2));

	fstat_out[6] = sgnlo[0];
	fstat_out[7] = sgnlo[1];
	fstat_out[8] = sgnlo[2];
	fstat_out[9] = sgnlo[3];

// Signal-to-noise ratio from estimated amplitudes (for h0 = 1)

	fstat_out[10] = sqrt(sqr(2*creal(xasum)) + sqr(2*creal(xbsum)) + sqr(2*cimag(xasum)) + sqr(2*cimag(xbsum))); 	

	free(_sph);
	free(_cph);
	free(phase);

	return fstat_out;
}



//mesh adaptive direct search (MADS) maximum search declaration
//double
Yep64f* MADS(Search_settings *sett, Aux_arrays *aux, double* in, double *start, double delta, double *pc, int bins){

	int i, j, k, l, m, n, o, r, a = 0;
  	double sinalt, cosalt, sindelt, cosdelt;
	double param[4]; 			//initial size of mesh
	double smallest = 100;
	for(r = 0; r < 4;r++){
		param[r] = pc[r]/bins;
	}
	for(r = 0; r < 4; r++){
		if(param[r] < smallest) smallest = param[r];
	}
#ifdef YEPPP
    yepLibrary_Init();
    Yep64f *p = (Yep64f*)malloc(sizeof(Yep64f)*4);
    Yep64f *out = (Yep64f*)malloc(sizeof(Yep64f)*11);
    Yep64f *nSource = (Yep64f*)malloc(sizeof(Yep64f)*3); 
    Yep64f *extr = (Yep64f*)malloc(sizeof(Yep64f)*11); 
    Yep64f *res = (Yep64f*)malloc(sizeof(Yep64f)*11);
    enum YepStatus status;

#endif
//	puts("MADS");

	for(i = 0; i < 11; i++) extr[i] = in[i];
	while(smallest >= delta){  	//when to end computations
		k = 0;
		for(j = 0; j < 8; j++){
			for(i = 0; i < 4; i++) p[i] = in[6+i];
			if(j < 4){
				p[k] = in[6+k] - start[k]*(param[k] - delta);
				k++;
			}
			else { 
				k--;
				p[k] = in[6+k] + start[k]*(param[k] - delta);
			}
			sinalt = sin(p[3]);
			cosalt = cos(p[3]);
			sindelt = sin(p[2]);
			cosdelt = cos(p[2]);

			nSource[0] = cosalt*cosdelt;
			nSource[1] = sinalt*cosdelt;
			nSource[2] = sindelt;

			for (o = 0; o < sett->nifo; ++o){
				modvir(sinalt, cosalt, sindelt, cosdelt, 
			   		sett->N, &ifo[o], aux);  
			}
			res = Fstatnet(sett, p, nSource); //Fstat function for mesh points

			if (res[5] < extr[5]){
				for (l = 0; l < 11; l++){ 
					extr[l] = res[l];
				}
			}

		} //j
		if (extr[5] == in[5]){
			smallest = smallest - delta;
			for(r = 0; r < 4; r++){
				param[r] = param[r] - delta;
			}
		}
		else{
			for(m = 0; m < 4; m++) p[m] = extr[6+m];
			for(m = 0; m < 11; m++) in[m] = extr[m];
		}
	} // while
	for (n= 0; n < 11; n++){
		out[n] = extr[n];
	}
	return out;
}

// Few functions for Nelder-Mead (simplex) algorithm  

double ** make_simplex(double * point, int dim, double *pc2){
	int i, j;
	double ** simplex = alloc_matrix(dim + 1, dim);
	for (i = 0; i < dim + 1; i++){
		for (j = 0; j < dim; j++){
			simplex[i][j] = point[j];
		}
	}
	for (i = 0; i < dim; i++){
		simplex[i][i] = (1 + pc2[i])*simplex[i][i];
	}
	return simplex;
}

void evaluate_simplex(double ** simplex, int dim, double ** fx, Search_settings *sett, Aux_arrays *aux, double *nS){
	double sinalt, cosalt, sindelt, cosdelt;
	double *out;
	int i, o, j;
	for (i = 0; i < dim + 1; i++){
			sinalt = sin(simplex[i][3]);
			cosalt = cos(simplex[i][3]);
			sindelt = sin(simplex[i][2]);
			cosdelt = cos(simplex[i][2]);

			nS[0] = cosalt*cosdelt;
			nS[1] = sinalt*cosdelt;
			nS[2] = sindelt;

			for (o = 0; o < sett->nifo; ++o){
				modvir(sinalt, cosalt, sindelt, cosdelt, 
			   		sett->N, &ifo[o], aux);  
			}
			out = Fstatnet(sett, simplex[i], nS);
			for (j = 0; j < 11; j++) fx[i][j] = out[j];
	}
}

int* simplex_extremes(double ** fx, int dim){
	int i;
	int ihi, ilo, inhi;
	static int ihe[3];

	if (fx[0][5] > fx[1][5]){ 
		ihi = 0; ilo = inhi = 1; 
	}
	else { 
		ihi = 1; 
		ilo = inhi = 0; 
	}
	for (i = 2; i < dim + 1; i++){
		if (fx[i][5] <= fx[ilo][5]) {
			ilo = i;
		}		
		else if (fx[i][5] > fx[ihi][5]){ 
			inhi = ihi; 
			ihi = i; 
		}
		else if (fx[i][5] > fx[inhi][5]){
			inhi = i;
		}
	}
	ihe[0] = ihi;
	ihe[1] = ilo;
	ihe[2] = inhi;

	return ihe;
}

void simplex_bearings(double ** simplex, int dim, double * midpoint, double * line, int ihi){
	int i, j;
	for (j = 0; j < dim; j++) midpoint[j] = 0.0;
	for (i = 0; i < dim + 1; i++){
		if (i != ihi){
			for (j = 0; j < dim; j++) midpoint[j] += simplex[i][j];
		}
	}
	for (j = 0; j < dim; j++){
		midpoint[j] /= dim;
		line[j] = simplex[ihi][j] - midpoint[j];
	}
}
int update_simplex(double ** simplex, int dim, double  fmax, double ** fx, int ihi, double * midpoint, double * line, double scale, Search_settings *sett, Aux_arrays *aux, double *nS){
	int i, o, j, update = 0; 
	double * next = alloc_vector(dim);
	double * fx2;
	double sinalt, cosalt, sindelt, cosdelt;
	for (i = 0; i < dim; i++) next[i] = midpoint[i] + scale * line[i];

	sinalt = sin(next[3]);
	cosalt = cos(next[3]);
	sindelt = sin(next[2]);
	cosdelt = cos(next[2]);

	nS[0] = cosalt*cosdelt;
	nS[1] = sinalt*cosdelt;
	nS[2] = sindelt;

	for (o = 0; o < sett->nifo; ++o){
		modvir(sinalt, cosalt, sindelt, cosdelt, 
	   		sett->N, &ifo[o], aux);  
	}
	fx2 = Fstatnet(sett, next, nS);
	if (fx2[5] < fmax){
		for (i = 0; i < dim; i++) simplex[ihi][i] = next[i];
		for (j = 0; j < 11; j++) fx[ihi][j] = fx2[j];
		update = 1;
	}
	free_vector(next, dim);
	return update;
}

void contract_simplex(double ** simplex, int dim, double ** fx, int ilo, int ihi, Search_settings *sett, Aux_arrays *aux, double *nS){
  	double sinalt, cosalt, sindelt, cosdelt;
	double * fx3;
	int i, j, k, o;
	for (i = 0; i < dim + 1; i++){
		if (i != ilo){
			for (j = 0; j < dim; j++) simplex[i][j] = (simplex[ilo][j]+simplex[i][j])*0.5;
			sinalt = sin(simplex[i][3]);
			cosalt = cos(simplex[i][3]);
			sindelt = sin(simplex[i][2]);
			cosdelt = cos(simplex[i][2]);

			nS[0] = cosalt*cosdelt;
			nS[1] = sinalt*cosdelt;
			nS[2] = sindelt;

			for (o = 0; o < sett->nifo; ++o){
				modvir(sinalt, cosalt, sindelt, cosdelt, 
			   		sett->N, &ifo[o], aux);  
			}

			fx3 = Fstatnet(sett, simplex[i], nS);
			for (k = 0; k < 11; k++) fx[i][k] = fx3[k];
		}
	}
}

int check_tol(double fmax, double fmin, double ftol){
	double delta = fabs(fmax - fmin);
	double accuracy = (fabs(fmax) + fabs(fmin)) * ftol;
	return (delta < (accuracy + ZEPS));
}

double * amoeba(Search_settings *sett, Aux_arrays *aux, double *point, double *nS, double *res_max, int dim, double tol, double *pc2){
	int ihi, ilo, inhi;
// ihi = ih[0], ilo = ih[1], inhi = ih[2];
	int *ih;
	int j, i;
	static double NM_out[11];
	double ** fx = alloc_matrix(dim + 1, 11);
	double * midpoint = alloc_vector(dim);
	double * line = alloc_vector(dim);
	double ** simplex = make_simplex(point, dim, pc2);
	evaluate_simplex(simplex, dim, fx, sett, aux, nS);
	while (true)
	{
		ih = simplex_extremes(fx, dim);
		ihi = ih[0];
		ilo = ih[1]; 
		inhi = ih[2];
		simplex_bearings(simplex, dim, midpoint, line, ihi);

		if(check_tol(fx[ihi][5], fx[ilo][5], tol)) break;
		update_simplex(simplex, dim, fx[ihi][5], fx, ihi, midpoint, line, -1.0, sett, aux, nS);

		if (fx[ihi][5] < fx[ilo][5]){
			update_simplex(simplex, dim, fx[ihi][5], fx, ihi, midpoint, line, -2.0, sett, aux, nS);
		}
		else if (fx[ihi][5] > fx[inhi][5]){
			if (!update_simplex(simplex, dim, fx[ihi][5], fx, ihi, midpoint, line, 0.5, sett, aux, nS)){
				contract_simplex(simplex, dim, fx, ilo, ihi, sett, aux, nS);
			}
		}
	}
	for (j = 0; j < dim; j++) point[j] = simplex[ilo][j];
	for (j = 0; j < 11; j++) NM_out[j] = fx[ilo][j];
/*	free_matrix(fx, dim + 1, 10);
	free_vector(midpoint, dim);
	free_vector(line, dim);
	free_matrix(simplex, dim + 1, dim);
*/
	free(fx);
	free(midpoint);
	free(line);
	free(simplex);
	return NM_out;
}

// Main programm
int main (int argc, char *argv[]) {

	Search_settings sett;	
	Command_line_opts opts;
  	Search_range s_range; 
  	Aux_arrays aux_arr;
  	double *F; 			// F-statistic array
  	int i, j, r, c, a, b, g; 	
	int d, o, m, k;
	int bins = 5, ROW, dim = 4;	// neighbourhood of point will be divide into defined number of bins
	double pc[4];			// % define neighbourhood around each parameter for initial grid
	double pc2[4];			// % define neighbourhood around each parameter for direct maximum search (MADS & Simplex)
	double tol = 1e-10;
//	double delta = 1e-5;		// initial step in MADS function
//	double *results;		// Vector with results from Fstatnet function
//	double *maximum;		// True maximum of Fstat
//	double results_max[11];	
	double s1, s2, s3, s4;
	double sgnlo[4]; 		 
	double **arr;		//  arr[ROW][COL], arrg[ROW][COL];
	double nSource[3];
  	double sinalt, cosalt, sindelt, cosdelt;
	double F_min;
	char path[512];
	double x, y;
	ROW = pow((bins+1),4);

#ifdef YEPPP
    yepLibrary_Init();
    Yep64f *results_max = (Yep64f*)malloc(sizeof(Yep64f)*11); 
    Yep64f *results_first = (Yep64f*)malloc(sizeof(Yep64f)*11);
    Yep64f *results = (Yep64f*)malloc(sizeof(Yep64f)*11);
    Yep64f *maximum = (Yep64f*)malloc(sizeof(Yep64f)*11);
//    Yep64f *sgnlo = (Yep64f*)malloc(sizeof(Yep64f)*4);  
//    Yep64f *nSource = (Yep64f*)malloc(sizeof(Yep64f)*3); 
    Yep64f *mean = (Yep64f*)malloc(sizeof(Yep64f)*4); 

    enum YepStatus status;

#endif

	pc[0] = 0.03;
	pc[1] = 0.03;
	pc[2] = 0.03;
	pc[3] = 0.03;

	for (i = 0; i < 4; i++){
		pc2[i] = 2*pc[i]/bins;
	}
// Time tests
	double tdiff;
	clock_t tstart, tend;

// Command line options 
	handle_opts(&sett, &opts, argc, argv); 
// Output data handling
/*  struct stat buffer;

  if (stat(opts.prefix, &buffer) == -1) {
    if (errno == ENOENT) {
      // Output directory apparently does not exist, try to create one
      if(mkdir(opts.prefix, S_IRWXU | S_IRGRP | S_IXGRP 
          | S_IROTH	| S_IXOTH) == -1) {
	      perror (opts.prefix);
	      return 1;
      }
    } else { // can't access output directory
      perror (opts.prefix);
      return 1;
    }
  }
*/
	sprintf(path, "%s/candidates.coi", opts.dtaprefix);

//Glue function
	if(strlen(opts.glue)) {
		glue(&opts);

		sprintf(opts.dtaprefix, "./data_total");
		sprintf(opts.dtaprefix, "%s/followup_total_data", opts.prefix); 
		opts.ident = 000;
	}	
	FILE *coi;
	int z;
	if ((coi = fopen(path, "r")) != NULL) {
//		while(!feof(coi)) {

/*			if(!fread(&w, sizeof(unsigned short int), 1, coi)) { break; } 
		  	fread(&mean, sizeof(float), 5, coi);
		  	fread(&fra, sizeof(unsigned short int), w, coi); 
		  	fread(&ops, sizeof(int), w, coi);

			if((fread(&mean, sizeof(float), 4, coi)) == 4){
*/
			while(fscanf(coi, "%le %le %le %le", &mean[0], &mean[1], &mean[2], &mean[3]) == 4){
//Time test
//			tstart = clock();
				arr = matrix(ROW, 4);

//Function neighbourhood - generating grid around point
				arr = neigh(mean, pc, bins);
// Output data handling
/*  				struct stat buffer;

  				if (stat(opts.prefix, &buffer) == -1) {
    					if (errno == ENOENT) {
// Output directory apparently does not exist, try to create one
      						if(mkdir(opts.prefix, S_IRWXU | S_IRGRP | S_IXGRP 
          						| S_IROTH	| S_IXOTH) == -1) {
	      						perror (opts.prefix);
	      						return 1;
      						}
    					} 
					else { // can't access output directory
			      			perror (opts.prefix);
			      			return 1;
			    		}
  				}
*/
// Grid data
  				if(strlen(opts.addsig)) { 

					read_grid(&sett, &opts);
				}
// Search settings
  				search_settings(&sett); 
// Detector network settings
  				detectors_settings(&sett, &opts); 
// Array initialization
  				init_arrays(&sett, &opts, &aux_arr, &F);
// Amplitude modulation functions for each detector  
				for(i=0; i<sett.nifo; i++) rogcvir(&ifo[i]); 
// Adding signal from file
  				if(strlen(opts.addsig)) { 

    					add_signal(&sett, &opts, &aux_arr, &s_range);
  				}

// Setting number of using threads (not required)
//omp_set_num_threads(2);

				results_max[5] = 0.;
// Main loop - over all parameters + parallelisation
//#pragma omp parallel default(shared) private(d, m, o, i, sgnlo, sinalt, cosalt, sindelt, cosdelt, nSource, results, maximum)
//{
//#pragma omp for  
				for (d = 0; d < ROW; ++d){

					for (m = 0; m < 4; m++){
						sgnlo[m] = arr[d][m];
					}
 
//for (m = 0; m < 4; m++) sgnlo[m] = mean[m]; 

					sinalt = sin(sgnlo[3]);
					cosalt = cos(sgnlo[3]);
					sindelt = sin(sgnlo[2]);
					cosdelt = cos(sgnlo[2]);

					nSource[0] = cosalt*cosdelt;
					nSource[1] = sinalt*cosdelt;
					nSource[2] = sindelt;

					for (o = 0; o < sett.nifo; ++o){
						modvir(sinalt, cosalt, sindelt, cosdelt, 
					   		sett.N, &ifo[o], &aux_arr);  
					}

// F-statistic in given point
//#pragma omp critical
					results = Fstatnet(&sett, sgnlo, nSource);
//					printf("%le %le %le %le %le %le\n", results[6], results[7], results[8], results[9], -results[5], results[4]);
//#pragma omp critical
					if(results[5] < results_max[5]){
						for (i = 0; i < 11; i++){
							results_max[i] = results[i];
						}
					}

// Maximum search using simplex algorithm
					if(opts.simplex_flag){
//						puts("Simplex");
						maximum = amoeba(&sett, &aux_arr, sgnlo, nSource, results, dim, tol, pc2);
// Maximum value in points searching
//#pragma omp critical
						if(maximum[5] < results_max[5]){
							for (i = 0; i < 11; i++){
								results_max[g] = maximum[g];
							}
						}

					} //simplex
				} // d - main outside loop
//} //pragma

				for(g = 0; g < 11; g++) results_first[g] = results_max[g];

// Maximum search using MADS algorithm
  				if(opts.mads_flag) {
//					puts("MADS");
					maximum = MADS(&sett, &aux_arr, results_max, mean, tol, pc2, bins);
				}

//Time test
//				tend = clock();
//				tdiff = (tend - tstart)/(double)CLOCKS_PER_SEC;
				printf("Maximum: %le %le %le %le %le %le\n", results_max[6], results_max[7], results_max[8], results_max[9], -results_max[5], results_max[4]);



			} // while fread coi
//		}
	} //if coi
	else {
		
		perror (path);
		return 1;
	}

// Output information
/*	puts("**********************************************************************");
	printf("***	Maximum value of F-statistic for grid is : (-)%.8le	***\n", -results_first[5]);
	printf("Sgnlo: %.8le %.8le %.8le %.8le\n", results_first[6], results_first[7], results_first[8], results_first[9]);
	printf("Amplitudes: %.8le %.8le %.8le %.8le\n", results_first[0], results_first[1], results_first[2], results_first[3]);
	printf("Signal-to-noise ratio: %.8le\n", results_first[4]); 
	printf("Signal-to-noise ratio from estimated amplitudes (for h0 = 1): %.8le\n", results_first[10]);
	puts("**********************************************************************");
if((opts.mads_flag)||(opts.simplex_flag)){
	printf("***	True maximum is : (-)%.8le				***\n", -maximum[5]);
	printf("Sgnlo for true maximum: %.8le %.8le %.8le %.8le\n", maximum[6], maximum[7], maximum[8], maximum[9]);
	printf("Amplitudes for true maximum: %.8le %.8le %.8le %.8le\n", maximum[0], maximum[1], maximum[2], maximum[3]);
	printf("Signal-to-noise ratio for true maximum: %.8le\n", maximum[4]); 
	printf("Signal-to-noise ratio from estimated amplitudes (for h0 = 1) for true maximum: %.8le\n", maximum[10]);
	puts("**********************************************************************");
}*/
// Cleanup & memory free 
//  	cleanup(&sett, &opts, &aux_arr, F);
	

	return 0;

}

//old test
//time LD_LIBRARY_PATH=lib/yeppp-1.0.0/binaries/linux/x86_64 ./followup -data ./data -ident 001 -band 100 -fpo 199.21875 
// new test: 
//time LD_LIBRARY_PATH=/home/msieniawska/tests/polgraw-allsky/search/network/src-cpu/lib/yeppp-1.0.0/binaries/linux/x86_64/ ./followup -data /home/msieniawska/tests/polgraw-allsky/followup/src/testdata/ -output /home/msieniawska/tests/polgraw-allsky/followup/src/output -label J0000+1902 -band 1902
//test for basic testdata:
//time LD_LIBRARY_PATH=/home/msieniawska/tests/polgraw-allsky/search/network/src-cpu/lib/yeppp-1.0.0/binaries/linux/x86_64/ ./followup -data /home/msieniawska/tests/polgraw-allsky/followup/src/testdata/ -output /home/msieniawska/tests/polgraw-allsky/followup/src/output -band 100 -ident 10 -fpo 199.21875
//time LD_LIBRARY_PATH=/home/msieniawska/tests/polgraw-allsky/search/network/src-cpu/lib/yeppp-1.0.0/binaries/linux/x86_64/ ./followup -data /home/msieniawska/tests/bigdogdata/mdc_025/ -output /home/polgraw-allsky/followup/src/output -band 100 -ident 10 -fpo 199.21875
//
//time LD_LIBRARY_PATH=/work/psk/msieniawska/test_followup/1/polgraw-allsky/search/network/src-cpu/lib/yeppp-1.0.0/binaries/linux/x86_64/ ./followup -data /work/psk/msieniawska/test_followup/data/ -output /work/psk/msieniawska/test_followup/output -band 103 -label 103_10 -fpo 103.0
//
//time LD_LIBRARY_PATH=/home/msieniawska/tests/bin_test/1/polgraw-allsky/search/network/src-cpu/lib/yeppp-1.0.0/binaries/linux/x86_64/ ./followup -data /home/msieniawska/tests/bin_test/data -output /home/msieniawska/tests/bin_test/output1 -fpo 124.9453125 -label 103_10 -fpo 103.0 -dt 2.0>& out1.txt
//
//time LD_LIBRARY_PATH=/home/msieniawska/tests/gluetest/polgraw-allsky/search/network/src-cpu/lib/yeppp-1.0.0/binaries/linux/x86_64/ ./followup -data /home/msieniawska/tests/gluetest/d1/followup_total_data -output /home/msieniawska/tests/gluetest/output1 -fpo 124.9453125 -label 103_10 -ident 000 -dt 2.0

//time LD_LIBRARY_PATH=/home/msieniawska/tests/addsig/polgraw-allsky/search/network/src-cpu/lib/yeppp-1.0.0/binaries/linux/x86_64 ./followup -data /home/msieniawska/tests/addsig/data -output . -ident 001 -band 0666 -dt 2 -addsig sigfile001 --nocheckpoint
//time LD_LIBRARY_PATH=/home/msieniawska/tests/addsig/polgraw-allsky/search/network/src-cpu/lib/yeppp-1.0.0/binaries/linux/x86_64 ./followup -data /home/msieniawska/tests/addsig/data -output . -ident 001 -band 0666 -dt 2 -addsig /home/msieniawska/tests/addsig/data/sigfile001 --nocheckpoint

//time LD_LIBRARY_PATH=/home/msieniawska/tests/addsig/polgraw-allsky/search/network/src-cpu/lib/yeppp-1.0.0/binaries/linux/x86_64 ./gwsearch-cpu -data /home/msieniawska/tests/addsig/data -output . -ident 031 -band 0666 -dt 2 -addsig /home/msieniawska/tests/addsig/data/sig1 --nocheckpoint >searchtest.txt

//time LD_LIBRARY_PATH=/home/msieniawska/tests/addsig/polgraw-allsky/test/search/network/src-cpu/lib/yeppp-1.0.0/binaries/linux/x86_64 ./followup -data /home/msieniawska/tests/addsig/data -output . -ident 031 -band 0666 -dt 2 -addsig /home/msieniawska/tests/addsig/data/sig1 -usedet H1 --simplex

