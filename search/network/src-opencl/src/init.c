// C behavioral defines
//
// MSVC: macro to include constants, such as M_PI (include before math.h)
#define _USE_MATH_DEFINES

// OpenCL behavioral defines
//
// 1.2+ OpenCL headers: tells the headers not to bitch about clCreateCommandQueue being renamed to clCreateCommandQueueWithProperties
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
//
// Select API to use
#define CL_TARGET_OPENCL_VERSION 120

// Polgraw includes
#include <init.h>       // all function declarations
#include <struct.h>     // Search_settings, Command_line_opts, OpenCL_handles, ...
#include <settings.h>
#include <auxi.h>
#include <spline_z.h>
#include <CL/util.h>    // checkErr

// OpenCL includes
#include <CL/cl.h>
#include <locations.h>

// OpenMP includes
#include <omp.h>

// Posix includes
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef _WIN32
#include <direct.h>
#include <dirent.h>
#include <getopt.h>
#else
#include <unistd.h>
#include <dirent.h>
#include <getopt.h>
#endif // WIN32

// Standard C includes
#include <stdio.h>
#include <stdlib.h>     // EXIT_FAILURE
#include <math.h>
#include <complex.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <limits.h>     // UINT_MAX


void handle_opts(Search_settings* sett,
                 OpenCL_settings* cl_sett,
                 Command_line_opts* opts,
                 int argc, 
                 char* argv[]) {
  
  opts->hemi=0;
  opts->wd=NULL;

  // Default F-statistic threshold 
  opts->trl=20;
    
  strcpy (opts->prefix, TOSTR(PREFIX));
  strcpy (opts->dtaprefix, TOSTR(DTAPREFIX));

  opts->label[0]  = '\0';
  opts->range[0]  = '\0';
  opts->getrange[0] = '\0';
  opts->usedet[0]   = '\0';
  opts->addsig[0] = '\0';
    
  // Initial value of starting frequency set to a negative quantity. 
  // If this is not changed by the command line value, fpo is calculated 
  // from the band number b (fpo = fpo = fstart + 0.96875*b/(2dt))
  sett->fpo = -1;

  // Default initial value of the data sampling time 
  sett->dt = 0.5; 

  opts->help_flag=0;
  opts->white_flag=0;
  opts->s0_flag=0;
  opts->checkp_flag=0;

  opts->dev[0] = '\0';
  opts->plat = 0;

  static int help_flag=0, white_flag=0, s0_flag=0, checkp_flag=1;

  // Reading arguments 

  while (1) {
    static struct option long_options[] = {
      {"help", no_argument, &help_flag, 1},
      {"whitenoise", no_argument, &white_flag, 1},
      {"nospindown", no_argument, &s0_flag, 1},
      {"nocheckpoint", no_argument, &checkp_flag, 0},
      // frame number
      {"ident", required_argument, 0, 'i'},
      // frequency band number
      {"band", required_argument, 0, 'b'},
      // output directory
      {"output", required_argument, 0, 'o'},
      // input data directory
      {"data", required_argument, 0, 'd'},
      // non-standard label for naming files
      {"label", required_argument, 0, 'l'},
      // narrower grid range parameter file
      {"range", required_argument, 0, 'r'},
      // write full grid range to file
      {"getrange", required_argument, 0, 'g'},
      // change directory parameter
      {"cwd", required_argument, 0, 'c'},
      // interpolation method
      {"threshold", required_argument, 0, 't'},
      // hemisphere
      {"hemisphere", required_argument, 0, 'h'},
      // fpo value
      {"fpo", required_argument, 0, 'p'},
      // add signal parameters
      {"addsig", required_argument, 0, 'x'},
      // which detectors to use
      {"usedet", required_argument, 0, 'u'}, 
      // data sampling time 
      {"dt", required_argument, 0, 's'},
      // which device type to use
      {"device", required_argument, 0, 'D'}, 
      // which platform id to use
      {"platform", required_argument, 0, 'P'},
      {0, 0, 0, 0}
    };

    if (help_flag) {

      printf("polgraw-allsky periodic GWs: search for candidate signals with the F-statistic\n");
      printf("Usage: ./search -[switch1] <value1> -[switch2] <value2> ...\n") ;
      printf("Switches are:\n\n");
      printf("-d, -data         Data directory (default is .)\n");
      printf("-o, -output       Output directory (default is ./candidates)\n");
      printf("-i, -ident        Frame number\n");
      printf("-b, -band         Band number\n");
      printf("-l, -label        Custom label for the input and output files\n");
      printf("-r, -range        Use file with grid range or pulsar position\n");
      printf("-g, -getrange     Write grid ranges & exit (ignore -r)\n");
      printf("-c, -cwd          Change to directory <dir>\n");
      printf("-t, -threshold    Threshold for the F-statistic (default is 20)\n");
      printf("-h, -hemisphere   Hemisphere (default is 0 - does both)\n");
      printf("-p, -fpo          Reference band frequency fpo value\n");
      printf("-s, -dt           data sampling time dt (default value: 0.5)\n");
      printf("-u, -usedet       Use only detectors from string (default is use all available)\n");
      printf("-x, -addsig       Add signal with parameters from <file>\n\n");
      printf("-D, -device       OpenCL device type to use [cpu|gpu|acc] (default maps to CL_DEVICE_TYPE_DEFAULT)\n\n");
      printf("-P, -platform     OpenCL platform id to use (default is 0)\n\n");


      printf("Also:\n\n");
      printf("--whitenoise      White Gaussian noise assumed\n");
      printf("--nospindown      Spindowns neglected\n");
      printf("--nocheckpoint    State file won't be created (no checkpointing)\n");
      printf("--help            This help\n");

      exit(EXIT_SUCCESS);
    }

    int option_index = 0;
    int c = getopt_long_only(argc, argv, "i:b:o:d:l:r:g:c:t:h:p:x:s:u:D:P:", 
                 long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 'i':
      opts->ident = atoi (optarg);
      break;
    case 't':
      opts->trl = atof(optarg);
      break;
    case 'h':
      opts->hemi = (int)atof(optarg); // WARNING: Why atof when assigning to int?
      break;
    case 'b':
      opts->band = atoi(optarg);
      break;
    case 'o':
      strcpy(opts->prefix, optarg);
      break;
    case 'd':
      strcpy(opts->dtaprefix, optarg);
      break;
    case 'l':
      opts->label[0] = '_';
      strcpy(1+opts->label, optarg);
      break;
    case 'r':
      strcpy(opts->range, optarg);
      break;
    case 'g':
      strcpy(opts->getrange, optarg);
      break;
    case 'c':
      opts->wd = (char *) malloc (1+strlen(optarg));
      strcpy(opts->wd, optarg);
      break;
    case 'p':
      sett->fpo = atof(optarg);
      break;
    case 'x':
      strcpy(opts->addsig, optarg);
      break;
    case 's':
      sett->dt = atof(optarg);
      break;
    case 'u':
      strcpy(opts->usedet, optarg);
      break;
    case 'D':
      strcpy(opts->dev, optarg);
      break;
    case 'P':
      opts->plat = atoi(optarg);
      break;

    case '?':
      break;
    default:
      break ;
    } // switch c
  } // while 1

  opts->white_flag = white_flag;
  opts->s0_flag = s0_flag;
  opts->checkp_flag = checkp_flag;	
    
  printf("Input data directory is %s\n", opts->dtaprefix);
  printf("Output directory is %s\n", opts->prefix);
  printf("Frame and band numbers are %d and %d\n", opts->ident, opts->band);

  // Starting band frequency:
  // fpo_val is optionally read from the command line
  // Its initial value is set to -1
  if(!(sett->fpo >= 0))

    // The usual definition (multiplying the offset by B=1/(2dt))
    // !!! in RDC_O1 the fstart equals 10, not 100 like in VSR1 !!! 
    // 
    sett->fpo = 10. + 0.96875*opts->band*(0.5/sett->dt);

  printf("The reference frequency fpo is %f\n", sett->fpo);
  printf("The data sampling time dt is  %f\n", sett->dt); 

  if (opts->white_flag)
    printf ("Assuming white Gaussian noise\n");

  // For legacy: FFT is now the only option 
  printf ("Using fftinterp=FFT (FFT interpolation by zero-padding)\n");

  if(opts->trl!=20)
    printf ("Threshold for the F-statistic is %lf\n", opts->trl);
  if(opts->hemi)
    printf ("Search for hemisphere %d\n", opts->hemi);
  if (opts->s0_flag)
    printf ("Assuming s_1 = 0.\n");
  if (strlen(opts->label))
    printf ("Using '%s' as data label\n", opts->label);

  if(strlen(opts->getrange)){
    printf ("Writing full grid ranges to '%s'\n", opts->getrange);
    if(strlen(opts->range)) {
      opts->range[0] = '\0';
      printf ("     WARNING! -r option will be ignored...\n");
    }
  }

  if (strlen(opts->range))
    printf ("Obtaining grid range from '%s'\n", opts->range);

  if (strlen(opts->addsig))
    printf ("Adding signal from '%s'\n", opts->addsig);
  if (opts->wd) {
    printf ("Changing working directory to %s\n", opts->wd);
#ifdef _WIN32
    if (_chdir(opts->wd)) { perror(opts->wd); abort(); }
#else
    if (chdir(opts->wd)) { perror (opts->wd); abort (); }
#endif
  }

  if (strlen(opts->dev))
  {
    if (!strncmp(opts->dev, "cpu", 3))
      cl_sett->dev_type = CL_DEVICE_TYPE_CPU;
    if (!strncmp(opts->dev, "gpu", 3))
      cl_sett->dev_type = CL_DEVICE_TYPE_GPU;
    if (!strncmp(opts->dev, "acc", 3))
      cl_sett->dev_type = CL_DEVICE_TYPE_ACCELERATOR;
  }
  else
  {
    cl_sett->dev_type = CL_DEVICE_TYPE_DEFAULT;
  }

  cl_sett->plat_id = opts->plat;

} // end of command line options handling 

void init_opencl(OpenCL_handles* cl_handles,
                 OpenCL_settings* cl_sett)
{
  cl_handles->plat = select_platform(cl_sett->plat_id);

  cl_handles->devs = select_devices(cl_handles->plat,
                                    cl_sett->dev_type,
                                    &cl_handles->dev_count);

  cl_handles->ctx = create_standard_context(cl_handles->devs,
                                            cl_handles->dev_count);

  cl_handles->write_queues = create_command_queue_set(cl_handles->ctx);
  cl_handles->exec_queues  = create_command_queue_set(cl_handles->ctx);
  cl_handles->read_queues  = create_command_queue_set(cl_handles->ctx);

  {
    char** sources = load_kernel_sources();

    cl_handles->prog = build_program_with_sources(cl_handles->ctx, sources);

    free_kernel_sources(sources);
  }

  cl_handles->kernels = create_kernels(cl_handles->prog,
                                       cl_handles->dev_count);
}

cl_platform_id select_platform(cl_uint plat_id)
{
    cl_int CL_err = CL_SUCCESS;
    cl_platform_id result = NULL;
    cl_uint numPlatforms = 0;

    CL_err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(CL_err, "clGetPlatformIDs(numPlatforms)");

    if (plat_id > (numPlatforms - 1))
    {
        perror("Platform of the specified index does not exist.");
        exit(-1);
    }
    else
    {
        cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
        CL_err = clGetPlatformIDs(numPlatforms, platforms, NULL);
        checkErr(CL_err, "clGetPlatformIDs(platforms)");

        result = platforms[plat_id];

        char pbuf[100];
        CL_err = clGetPlatformInfo( result, CL_PLATFORM_VENDOR, sizeof( pbuf ), pbuf, NULL );
            checkErr( CL_err, "clGetPlatformInfo(CL_PLATFORM_VENDOR)" );
#ifdef WIN32
            printf_s( "Selected OpenCL platform vendor:\n\t%s\n", pbuf ); // TODO: don't throw away error code.
#else
            printf( "Selected OpenCL platform vendor:\n\t%s\n", pbuf );
#endif
        
        free(platforms);
    }

    return result;
}

cl_device_id* select_devices(cl_platform_id platform,
                             cl_device_type dev_type,
                             cl_uint* count)
{
    cl_int CL_err = CL_SUCCESS;
    cl_device_id* result = NULL;

    CL_err = clGetDeviceIDs(platform, dev_type, 0, 0, count);
    checkErr(CL_err, "clGetDeviceIDs(numDevices)");

    if (*count == 0u)
    {
        perror("No devices of the specified type are found on the specified platform.");
        exit(-1);
    }
    else
    {
        // Original
        result = (cl_device_id*)malloc(*count * sizeof(cl_device_id));
        CL_err = clGetDeviceIDs(platform, dev_type, *count, result, 0);
        checkErr(CL_err, "clGetDeviceIDs(devices)");

        // Forced multi-device
        //result = (cl_device_id*)malloc(2 * sizeof(cl_device_id));
        //CL_err = clGetDeviceIDs(platform, dev_type, 1, &result[0], 0);
        //checkErr(CL_err, "clGetDeviceIDs(devices)");
        //CL_err = clGetDeviceIDs(platform, dev_type, 1, &result[1], 0);
        //checkErr(CL_err, "clGetDeviceIDs(devices)");
        //*count = 2;

#ifdef WIN32
            printf_s( "Selected OpenCL device(s):\n" ); // TODO: don't throw away error code.
#else
            printf( "Selected OpenCL device(s):\n" );
#endif
        cl_uint i;
        for(i = 0; i < *count; ++i)
        {
            char pbuf[100];
            CL_err = clGetDeviceInfo( result[i], CL_DEVICE_NAME, sizeof( pbuf ), pbuf, NULL );
            checkErr( CL_err, "clGetDeviceInfo(CL_DEVICE_NAME)" );
#ifdef WIN32
            printf_s( "\t%s\n", pbuf ); // TODO: don't throw away error code.
#else
            printf( "\t%s\n", pbuf );
#endif
        }
    }

    return result;
}

cl_context create_standard_context(cl_device_id* devices, cl_uint count)
{
    cl_int CL_err = CL_SUCCESS;
    cl_context result = NULL;
    cl_platform_id platform = NULL;

    CL_err = clGetDeviceInfo(devices[0],
                             CL_DEVICE_PLATFORM,
                             sizeof(cl_platform_id),
                             &platform,
                             NULL);
    checkErr(CL_err, "clGetDeviceInfo(CL_DEVICE_PLATFORM)");

    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

    result = clCreateContext(cps, count, devices, NULL, NULL, &CL_err);
    checkErr(CL_err, "clCreateContext()");

    return result;
}

cl_command_queue** create_command_queue_set(cl_context context)
{
    cl_int CL_err = CL_SUCCESS;

    cl_uint count = 0;
    CL_err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &count, NULL);
    checkErr(CL_err, "clGetContextInfo(CL_CONTEXT_NUM_DEVICES)");

	cl_device_id* devices = (cl_device_id*)malloc(count * sizeof(cl_device_id));

    CL_err = clGetContextInfo(context, CL_CONTEXT_DEVICES, count * sizeof(cl_device_id), devices, NULL);
    checkErr(CL_err, "clGetContextInfo(CL_CONTEXT_DEVICES)");

	cl_command_queue** result = (cl_command_queue**)malloc(count * sizeof(cl_command_queue*));

    for (cl_uint i = 0; i < count; ++i)
    {
		result[i] = (cl_command_queue*)malloc(MAX_DETECTORS * sizeof(cl_command_queue));

		for (cl_uint j = 0; j < MAX_DETECTORS; ++j)
		{
			result[i][j] = clCreateCommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &CL_err);
			checkErr(CL_err, "clCreateCommandQueue()");
		}

        CL_err = clReleaseDevice(devices[i]);
        checkErr(CL_err, "clReleaseDevice()");
    }

    free(devices);

    return result;
}

char* load_file(const char* filename)
{
    long int size = 0;
    size_t res = 0;
    char* src = NULL;
    FILE* file = NULL;
#ifdef _WIN32
    errno_t err = 0;

    err = fopen_s(&file, filename, "rb");
    if (err)
    {
        printf_s("Failed to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
#else
    file = fopen(filename, "rb");

    if (!file)
    {
        printf("Failed to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
#endif

    if (fseek(file, 0, SEEK_END))
    {
        fclose(file);
        return NULL;
    }

    size = ftell(file);
    if (size == 0)
    {
        fclose(file);
        return NULL;
    }

    rewind(file);

    src = (char *)calloc(size + 1, sizeof(char));
    if (!src)
    {
        src = NULL;
        fclose(file);
        return src;
    }

    res = fread(src, 1u, sizeof(char) * size, file);
    if (res != sizeof(char) * size)
    {
        fclose(file);
        free(src);

        return src;
    }

    src[size] = '\0'; // NULL terminated
    fclose(file);

    return src;
}

char** load_kernel_sources()
{
    char** result = (char**)malloc(kernel_path_count * sizeof(char*));

    for (int i = 0; i < kernel_path_count; ++i)
        result[i] = load_file(kernel_paths[i]);

    return result;
}

void free_kernel_sources(char** sources)
{
    for (int i = 0; i < kernel_path_count; ++i)
        free(sources[i]);

    free(sources);
}

cl_program build_program_with_sources(cl_context context,
                                      const char** sources)
{
    cl_int CL_err = CL_SUCCESS;
    cl_program result = NULL;

    cl_uint numDevices = 0;
    cl_device_id* devices = NULL;

    size_t* lengths = (size_t*)malloc(kernel_path_count * sizeof(size_t));
    for(int i = 0 ; i < kernel_path_count; ++i)
#ifdef _WIN32
      lengths[i] = strnlen_s(sources[i], UINT_MAX);
#else
      lengths[i] = strnlen(sources[i]);
#endif

    cl_uint uint_kernel_path_count = (cl_uint)kernel_path_count;
    result = clCreateProgramWithSource(context, uint_kernel_path_count, sources, lengths, &CL_err);
    checkErr(CL_err, "clCreateProgramWithSource()");

    CL_err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);
    checkErr(CL_err, "clGetContextInfo(CL_CONTEXT_NUM_DEVICES)");
    devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
    CL_err = clGetContextInfo(context, CL_CONTEXT_DEVICES, numDevices * sizeof(cl_device_id), devices, NULL);
    checkErr(CL_err, "clGetContextInfo(CL_CONTEXT_DEVICES)");

    char build_params[1024];
    strcpy(build_params, " -I");
    strcat(build_params, kernel_inc_path );
    strcat(build_params, " -cl-opt-disable" );
    strcat(build_params, " -cl-std=CL1.2" );
    strcat(build_params, " -Werror" ); // Warnings will be treated like errors, this is useful for debug
    CL_err = clBuildProgram(result, numDevices, devices, build_params, NULL, NULL);

    if (CL_err != CL_SUCCESS)
    {
        size_t len = 0;
        char *buffer;

        CL_err = clGetProgramBuildInfo(result, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        checkErr(CL_err, "clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG)");

        buffer = calloc(len, sizeof(char));

        clGetProgramBuildInfo(result, devices[0], CL_PROGRAM_BUILD_LOG, len, buffer, NULL);

        fprintf(stderr, "%s\n", buffer);

        free(buffer);

        checkErr(-111, "clBuildProgram");
    }

    free(lengths);

    return result;
}

cl_kernel** create_kernels(cl_program program,
	                       cl_uint count)
{
    cl_kernel** result = (cl_kernel**)malloc(count * sizeof(cl_kernel*));

	for (cl_uint i = 0; i < count; ++i)
	{
		result[i] = (cl_kernel*)malloc(kernel_count * sizeof(cl_kernel));

		for (cl_uint j = 0; j < kernel_count; ++j)
			result[i][j] = obtain_kernel(program, j);
	}

    return result;
}

const char* obtain_kernel_name(cl_uint i)
{
    const char* result = NULL;

    switch (i)
    {
    case Modvir:
        result = "modvir";
        break;
    case TShiftPMod:
        result = "tshift_pmod";
        break;
    case ResamplePostFFT:
        result = "resample_postfft";
        break;
    //case ComputeB:
    //    result = "computeB";
    //    break;
    //case TriDiagMul:
    //    result = "tridiagMul";
    //    break;
    //case Interpolate:
    //    result = "interpolate";
    //    break;
    case PhaseMod1:
        result = "phase_mod_1";
        break;
    case PhaseMod2:
        result = "phase_mod_2";
        break;
    case ComputeFStat:
        result = "compute_Fstat";
        break;
    case NormalizeFStatWG:
        result = "normalize_Fstat_wg_reduce";
        break;
    default:
        perror("Unkown kernel index");
        exit(-1);
        break;
    }

    return result;
}

cl_kernel obtain_kernel(cl_program program, cl_uint i)
{
    cl_int CL_err = CL_SUCCESS;
    cl_kernel result = NULL;

    result = clCreateKernel(program, obtain_kernel_name(i), &CL_err);
    checkErr(CL_err, "clCreateKernel()");

    return result;
}

void read_grid(Search_settings *sett,
               Command_line_opts *opts)
{
    sett->M = (double *)calloc(16, sizeof(double));

    FILE *data;
    char filename[512];

    // In case when -usedet option is used for one detector
    // i.e. opts->usedet has a length of 2 (e.g. H1 or V1), 
    // read grid.bin from this detector subdirectory 
    // (see detectors_settings() in settings.c for details) 
    if (strlen(opts->usedet) == 2)
        sprintf(filename, "%s/%03d/%s/grid.bin", opts->dtaprefix, opts->ident, opts->usedet);
    else
        sprintf(filename, "%s/%03d/grid.bin", opts->dtaprefix, opts->ident);

    if ((data = fopen(filename, "rb")) != NULL)
    {
        printf("Using grid file from %s\n", filename);
        fread((void *)&sett->fftpad, sizeof(int), 1, data);
        printf("Using fftpad from the grid file: %d\n", sett->fftpad);

        // M: vector of 16 components consisting of 4 rows
        // of 4x4 grid-generating matrix
        fread((void *)sett->M, sizeof(double), 16, data);
        fclose(data);
    }
    else
    {
        perror(filename);
        exit(EXIT_FAILURE);
    }

} // end of read grid 

void init_arrays(Detector_settings* ifo,
                 Search_settings* sett,
                 OpenCL_handles* cl_handles,
                 Command_line_opts* opts,
                 Aux_arrays *aux_arr,
	             FFT_arrays* fft_arr)
{
	init_ifo_arrays(sett, cl_handles, opts, ifo);

	init_aux_arrays(sett, ifo, cl_handles, aux_arr);

	init_fft_arrays(sett, cl_handles, fft_arr);

} // end of init arrays

void init_ifo_arrays(Search_settings* sett,
                     OpenCL_handles* cl_handles,
                     Command_line_opts* opts,
                     Detector_settings* ifo)
{
  // Allocates and initializes to zero the data, detector ephemeris
  // and the F-statistic arrays
  size_t status;

  for (int i = 0; i < sett->nifo; i++)
  {
    ifo[i].sig.xDat = (double*)calloc(sett->N, sizeof(double));

    // Input time-domain data handling
    // 
    // The file name ifo[i].xdatname is constructed 
    // in settings.c, while looking for the detector 
    // subdirectories
	FILE *data;
    if ((data = fopen(ifo[i].xdatname, "rb")) != NULL)
    {
		status = fread((void *)(ifo[i].sig.xDat),
                              sizeof(double),
                              sett->N,
                              data);
        fclose(data);
    }
    else
    {
      perror(ifo[i].xdatname);
      exit(EXIT_FAILURE);
    }

    cl_int CL_err = CL_SUCCESS;
    // Potentially wasteful conversion from storage to computation type
    {
      xDat_real* tmp = (xDat_real*)malloc(sett->N * sizeof(xDat_real));
      for (int ii = 0 ; ii < sett->N ; ++ii) tmp[ii] = (xDat_real)ifo[i].sig.xDat[ii]; // Cast silences warning

      ifo[i].sig.xDat_d = clCreateBuffer(cl_handles->ctx,
                                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sett->N * sizeof(xDat_real),
                                         tmp,
                                         &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.xDat_d)");
      free(tmp);
    }

    int Nzeros = 0;
    // Checking for null values in the data
    for (int j = 0; j < sett->N; j++)
      if (!ifo[i].sig.xDat[j]) Nzeros++;

    ifo[i].sig.Nzeros = Nzeros;

    // factor N/(N - Nzeros) to account for null values in the data
    ifo[i].sig.crf0 = (double)sett->N / (sett->N - ifo[i].sig.Nzeros);

    // Estimation of the variance for each detector 
    ifo[i].sig.sig2 = (ifo[i].sig.crf0)*var(ifo[i].sig.xDat, sett->N);

    ifo[i].sig.DetSSB = (DetSSB_real3*)calloc(sett->N, sizeof(DetSSB_real3));

    // Ephemeris file handling
    char filename[512];

    sprintf(filename,
            "%s/%03d/%s/DetSSB.bin",
            opts->dtaprefix,
            opts->ident,
            ifo[i].name);

    if ((data = fopen(filename, "rb")) != NULL)
    {
      // Detector position w.r.t Solar System Baricenter
      // for every datapoint
      for (int j = 0; j < sett->N; ++j)
      {
        double tmp[3];
        status = fread((void *)(&tmp),
                       sizeof(double),
                       3,
                       data);

        ifo[i].sig.DetSSB[j].s[0] = (DetSSB_real)tmp[0];
        ifo[i].sig.DetSSB[j].s[1] = (DetSSB_real)tmp[1];
        ifo[i].sig.DetSSB[j].s[2] = (DetSSB_real)tmp[2];
        ifo[i].sig.DetSSB[j].s[3] = (DetSSB_real)0;
      }

      // Deterministic phase defining the position of the Earth
      // in its diurnal motion at t=0 
      status = fread((void *)(&ifo[i].sig.phir),
                     sizeof(double),
                     1,
                     data);

      // Earth's axis inclination to the ecliptic at t=0
      status = fread((void *)(&ifo[i].sig.epsm),
                     sizeof(double),
                     1,
                     data);
      fclose(data);

      printf("Using %s as detector %s ephemerids...\n", filename, ifo[i].name);

    }
    else
    {
      perror(filename);
      return;
    }

    ifo[i].sig.DetSSB_d = clCreateBuffer(cl_handles->ctx,
                                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sett->N * sizeof(DetSSB_real3),
                                         ifo[i].sig.DetSSB,
                                         &CL_err);
    checkErr(CL_err, "clCreateBuffer(ifo[i].sig.DetSSB_d)");

    // sincos 
    ifo[i].sig.sphir = sin(ifo[i].sig.phir);
    ifo[i].sig.cphir = cos(ifo[i].sig.phir);
    ifo[i].sig.sepsm = sin(ifo[i].sig.epsm);
    ifo[i].sig.cepsm = cos(ifo[i].sig.epsm);

    sett->sepsm = ifo[i].sig.sepsm;
    sett->cepsm = ifo[i].sig.cepsm;

	ifo[i].sig.xDatma_d = (cl_mem*)malloc(cl_handles->dev_count * sizeof(cl_mem));
	ifo[i].sig.xDatmb_d = (cl_mem*)malloc(cl_handles->dev_count * sizeof(cl_mem));
	ifo[i].sig.aa_d =     (cl_mem*)malloc(cl_handles->dev_count * sizeof(cl_mem));
	ifo[i].sig.bb_d =     (cl_mem*)malloc(cl_handles->dev_count * sizeof(cl_mem));
	ifo[i].sig.shft_d =   (cl_mem*)malloc(cl_handles->dev_count * sizeof(cl_mem));
	ifo[i].sig.shftf_d =  (cl_mem*)malloc(cl_handles->dev_count * sizeof(cl_mem));

	for (cl_uint j = 0; j < cl_handles->dev_count; ++j)
	{
      ifo[i].sig.xDatma_d[j] = clCreateBuffer(cl_handles->ctx,
                                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                              sett->N * sizeof(xDatm_complex),
                                              NULL,
                                              &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.xDatma_d)");
	  
      ifo[i].sig.xDatmb_d[j] = clCreateBuffer(cl_handles->ctx,
                                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                              sett->N * sizeof(xDatm_complex),
                                              NULL,
                                              &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.xDatmb_d)");
	  
      ifo[i].sig.aa_d[j] = clCreateBuffer(cl_handles->ctx,
                                          CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                          sett->N * sizeof(ampl_mod_real),
                                          NULL,
                                          &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.aa_d)");
	  
      ifo[i].sig.bb_d[j] = clCreateBuffer(cl_handles->ctx,
                                          CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                          sett->N * sizeof(ampl_mod_real),
                                          NULL,
                                          &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.bb_d)");
	  
      ifo[i].sig.shft_d[j] = clCreateBuffer(cl_handles->ctx,
                                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                            sett->N * sizeof(shift_real),
                                            NULL,
                                            &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.shft_d)");
	  
      ifo[i].sig.shftf_d[j] = clCreateBuffer(cl_handles->ctx,
                                             CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                             sett->N * sizeof(shift_real),
                                             NULL,
                                             &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.shftf_d)");
	}
  } // end loop for detectors 

  // Check if the ephemerids have the same epsm parameter
  for (int i = 1; i<sett->nifo; i++)
  {
    if (!(ifo[i - 1].sig.sepsm == ifo[i].sig.sepsm))
    {
      printf("The parameter epsm (DetSSB.bin) differs for detectors %s and %s. Aborting...\n",
             ifo[i - 1].name,
             ifo[i].name);
      exit(EXIT_FAILURE);
    }

  }

  // if all is well with epsm, take the first value 
  sett->sepsm = ifo[0].sig.sepsm;
  sett->cepsm = ifo[0].sig.cepsm;
}

void init_fft_arrays(Search_settings* sett,
                     OpenCL_handles* cl_handles,
                     FFT_arrays* fft_arr)
{
  fft_arr->arr_len =
    (sett->fftpad*sett->nfft > sett->Ninterp ?
       sett->fftpad*sett->nfft :
       sett->Ninterp);

  fft_arr->xa_d = (cl_mem**)malloc(cl_handles->dev_count * sizeof(cl_mem*));
  fft_arr->xb_d = (cl_mem**)malloc(cl_handles->dev_count * sizeof(cl_mem*));

  for (cl_uint id = 0; id < cl_handles->dev_count; ++id)
  {
    fft_arr->xa_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem));
    fft_arr->xb_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem));

	for (int n = 0; n < sett->nifo; ++n)
    {
      cl_int CL_err = CL_SUCCESS;
      fft_arr->xa_d[id][n] =
        clCreateBuffer(cl_handles->ctx,
                       CL_MEM_READ_WRITE,
                       fft_arr->arr_len * sizeof(fft_complex),
                       NULL,
                       &CL_err);
      checkErr(CL_err, "clCreateBuffer(fft_arr->xa_d)");
    
      fft_arr->xb_d[id][n] =
        clCreateBuffer(cl_handles->ctx,
                       CL_MEM_READ_WRITE,
                       fft_arr->arr_len * sizeof(fft_complex),
                       NULL,
                       &CL_err);
      checkErr(CL_err, "clCreateBuffer(fft_arr->xb_d)");
    }
  }
}

void init_aux_arrays(Search_settings* sett,
                     Detector_settings* ifo,
                     OpenCL_handles* cl_handles,
                     Aux_arrays* aux_arr)
{
  cl_int CL_err = CL_SUCCESS;

  aux_arr->ifo_amod_d =
    clCreateBuffer(cl_handles->ctx,
                   CL_MEM_READ_ONLY,
                   sett->nifo * sizeof(Ampl_mod_coeff),
                   NULL,
                   &CL_err);
  checkErr(CL_err, "clCreateBuffer(aux_arr->ifo_amod_d)");
  
  aux_arr->tshift_d = (cl_mem**)malloc(cl_handles->dev_count * sizeof(cl_mem*));
  aux_arr->aadots_d = (cl_mem**)malloc(cl_handles->dev_count * sizeof(cl_mem*));
  aux_arr->bbdots_d = (cl_mem**)malloc(cl_handles->dev_count * sizeof(cl_mem*));
  aux_arr->maa_d = (cl_mem*)malloc(cl_handles->dev_count * sizeof(cl_mem));
  aux_arr->mbb_d = (cl_mem*)malloc(cl_handles->dev_count * sizeof(cl_mem));

  aux_arr->F_d = (cl_mem*)malloc(cl_handles->dev_count * sizeof(cl_mem));

  for (cl_uint id = 0; id < cl_handles->dev_count; ++id)
  {
    aux_arr->tshift_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem));
    aux_arr->aadots_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem));
    aux_arr->bbdots_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem));

    for (int n = 0; n < sett->nifo; ++n)
    {
      aux_arr->tshift_d[id][n] =
        clCreateBuffer(cl_handles->ctx,
                       CL_MEM_READ_WRITE,
                       sett->N * sizeof(shift_real),
                       NULL,
                       &CL_err);
      checkErr(CL_err, "clCreateBuffer(aux_arr->tshift_d)");

      aux_arr->aadots_d[id][n] =
        clCreateBuffer(cl_handles->ctx,
                       CL_MEM_READ_ONLY,
                       sizeof(ampl_mod_real),
                       NULL,
                       &CL_err);
      checkErr(CL_err, "clCreateBuffer(aux_arr->aadots_d)");

      aux_arr->bbdots_d[id][n] =
        clCreateBuffer(cl_handles->ctx,
                       CL_MEM_READ_ONLY,
                       sizeof(ampl_mod_real),
                       NULL,
                       &CL_err);
      checkErr(CL_err, "clCreateBuffer(aux_arr->bbdots_d)");
    }

    aux_arr->maa_d[id] =
      clCreateBuffer(cl_handles->ctx,
                     CL_MEM_READ_ONLY,
                     sizeof(ampl_mod_real),
                     NULL,
                     &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->maa_d)");

    aux_arr->mbb_d[id] =
      clCreateBuffer(cl_handles->ctx,
                     CL_MEM_READ_ONLY,
                     sizeof(ampl_mod_real),
                     NULL,
                     &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->mbb_d)");

    aux_arr->F_d[id] =
      clCreateBuffer(cl_handles->ctx,
                     CL_MEM_READ_WRITE,
                     2 * sett->nfft * sizeof(fstat_real),
                     NULL,
                     &CL_err);
    checkErr(CL_err, "clCreateBuffer(F_d)");
  }

  Ampl_mod_coeff* tmp =
      clEnqueueMapBuffer(cl_handles->exec_queues[0][0],
          aux_arr->ifo_amod_d,
          CL_TRUE,
          CL_MAP_WRITE_INVALIDATE_REGION,
          0,
          sett->nifo * sizeof(Ampl_mod_coeff),
          0,
          NULL,
          NULL,
          &CL_err);

  for (size_t i = 0; i < sett->nifo; ++i) tmp[i] = ifo[i].amod;

  cl_event unmap_event;
  clEnqueueUnmapMemObject(cl_handles->exec_queues[0][0], aux_arr->ifo_amod_d, tmp, 0, NULL, &unmap_event);

  clWaitForEvents(1, &unmap_event);

  // OpenCL cleanup
  clReleaseEvent(unmap_event);
}

void set_search_range(Search_settings *sett,
                      Command_line_opts *opts,
                      Search_range *s_range)
{
    // Hemispheres (with respect to the ecliptic)
    if (opts->hemi)
    {
        s_range->pmr[0] = opts->hemi;
        s_range->pmr[1] = opts->hemi;
    }
    else
    {
        s_range->pmr[0] = 1;
        s_range->pmr[1] = 2;
    }

    // If the parameter range is invoked, the search is performed
    // within the range of grid parameters from an ascii file
    // ("-r range_file" from the command line)
    FILE *data;

    if (strlen(opts->range))
    {
        if ((data = fopen(opts->range, "rb")) != NULL)
        {
            int aqq = fscanf(data, "%d %d %d %d %d %d %d %d",
                             s_range->spndr, 1 + s_range->spndr, s_range->nr,
                             1 + s_range->nr, s_range->mr, 1 + s_range->mr,
                             s_range->pmr, 1 + s_range->pmr);

            if (aqq != 8)
            {
                printf("Error when reading range file!\n");
                exit(EXIT_FAILURE);
            }

            fclose(data);
        }
        else
        {
            perror(opts->range);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        // Establish the grid range in which the search will be performed
        // with the use of the M matrix from grid.bin
        gridr(sett->M,
              s_range->spndr,
              s_range->nr,
              s_range->mr,
              sett->oms,
              sett->Smax);

        if (strlen(opts->getrange))
        {
            FILE *data;
            if ((data = fopen(opts->getrange, "w")) != NULL)
            {
                fprintf(data, "%d %d\n%d %d\n%d %d\n%d %d\n",
                        s_range->spndr[0], s_range->spndr[1],
                        s_range->nr[0], s_range->nr[1],
                        s_range->mr[0], s_range->mr[1],
                        s_range->pmr[0], s_range->pmr[1]);

                printf("Wrote input data grid ranges to %s\n", opts->getrange);
                fclose(data);
                //exit(EXIT_SUCCESS);
            }
            else
            {
                printf("Can't open %s file for writing\n", opts->getrange);
                exit(EXIT_FAILURE);
            }
        }
    }

    printf("set_search_range() - the grid ranges are maximally this:\n");
    printf("(spndr, nr, mr, pmr pairs): %d %d %d %d %d %d %d %d\n",
           s_range->spndr[0], s_range->spndr[1], s_range->nr[0], s_range->nr[1],
           s_range->mr[0], s_range->mr[1], s_range->pmr[0], s_range->pmr[1]);

    printf("Smin: %le, -Smax: %le\n", sett->Smin, sett->Smax);

} // end of set search range

void init_blas(Search_settings* sett,
               OpenCL_handles* cl_handles,
               BLAS_handles* blas_handles)
{
  clblasStatus status = clblasSetup();
  checkErrBLAS(status, "clblasSetup()");

  blas_handles->aaScratch_d = (cl_mem**)malloc(cl_handles->dev_count * sizeof(cl_mem));
  blas_handles->bbScratch_d = (cl_mem**)malloc(cl_handles->dev_count * sizeof(cl_mem));

  for (cl_uint id = 0; id < cl_handles->dev_count; ++id)
  {
    blas_handles->aaScratch_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem*));
    blas_handles->bbScratch_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem*));
    
    for (int idet = 0; idet < sett->nifo; ++idet)
    {
      cl_int CL_err = CL_SUCCESS;
      blas_handles->aaScratch_d[id][idet] =
        clCreateBuffer(cl_handles->ctx,
                       CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       sett->N * sizeof(ampl_mod_real),
                       NULL,
                       &CL_err);
	  checkErr(CL_err, "clCreateBuffer(blas_handles->aaScratch_d)");

      blas_handles->bbScratch_d[id][idet] =
        clCreateBuffer(cl_handles->ctx,
                       CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       sett->N * sizeof(ampl_mod_real),
                       NULL,
                       &CL_err);
	  checkErr(CL_err, "clCreateBuffer(blas_handles->bbScratch_d)");
    }
  }
}

void init_fft(Search_settings* sett,
              OpenCL_handles* cl_handles,
              FFT_plans* plans)
{
	size_t nfftf_size = (size_t)sett->nfftf,
           nfft_size = (size_t)sett->nfft,
		   Ninterp_size = (size_t)sett->Ninterp;

    plans->plan =   (clfftPlanHandle*)malloc(cl_handles->dev_count * sizeof(clfftPlanHandle));
    plans->pl_int = (clfftPlanHandle*)malloc(cl_handles->dev_count * sizeof(clfftPlanHandle));
    plans->pl_inv = (clfftPlanHandle*)malloc(cl_handles->dev_count * sizeof(clfftPlanHandle));

	clfftSetupData fftSetup;
	clfftStatus CLFFT_status = clfftSetup(&fftSetup);
	checkErrFFT(CLFFT_status, "clffftSetup");

    for (cl_uint id = 0; id < cl_handles->dev_count; ++id)
    {
        // Phasemod FFT
        CLFFT_status = clfftCreateDefaultPlan(&plans->plan[id], cl_handles->ctx, CLFFT_1D, &nfftf_size);
        checkErrFFT(CLFFT_status, "clCreateDefaultPlan");

        clfftPrecision clfft_precision = sizeof(fft_complex) == 8 ? CLFFT_SINGLE : CLFFT_DOUBLE;

        CLFFT_status = clfftSetPlanPrecision(plans->plan[id], clfft_precision);
        checkErrFFT(CLFFT_status, "clfftSetPlanPrecision()");
        CLFFT_status = clfftSetLayout(plans->plan[id], CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
        checkErrFFT(CLFFT_status, "clfftSetLayout(CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED)");
        CLFFT_status = clfftSetResultLocation(plans->plan[id], CLFFT_INPLACE);
        checkErrFFT(CLFFT_status, "clfftSetResultLocation(CLFFT_INPLACE)");

        CLFFT_status = clfftBakePlan(plans->plan[id],
                                     1,
                                     &cl_handles->exec_queues[id][0],
                                     NULL,
                                     NULL);
        checkErrFFT(CLFFT_status, "clfftBakePlan(plans->pl_int)");

        // Interpolation FFT
        CLFFT_status = clfftCreateDefaultPlan(&plans->pl_int[id], cl_handles->ctx, CLFFT_1D, &nfft_size);
        checkErrFFT(CLFFT_status, "clCreateDefaultPlan");

        CLFFT_status = clfftSetPlanPrecision(plans->pl_int[id], clfft_precision);
        checkErrFFT(CLFFT_status, "clfftSetPlanPrecision(CLFFT_SINGLE)");
        CLFFT_status = clfftSetLayout(plans->pl_int[id], CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
        checkErrFFT(CLFFT_status, "clfftSetLayout(CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED)");
        CLFFT_status = clfftSetResultLocation(plans->pl_int[id], CLFFT_INPLACE);
        checkErrFFT(CLFFT_status, "clfftSetResultLocation(CLFFT_INPLACE)");

        CLFFT_status = clfftBakePlan(plans->pl_int[id],
                                     1,
                                     &cl_handles->exec_queues[id][0],
                                     NULL,
                                     NULL);
        checkErrFFT(CLFFT_status, "clfftBakePlan(plans->pl_int)");

        // Inverse FFT
        CLFFT_status = clfftCreateDefaultPlan(&plans->pl_inv[id], cl_handles->ctx, CLFFT_1D, &Ninterp_size);
        checkErrFFT(CLFFT_status, "clCreateDefaultPlan");

        CLFFT_status = clfftSetPlanPrecision(plans->pl_inv[id], clfft_precision);
        checkErrFFT(CLFFT_status, "clfftSetPlanPrecision(CLFFT_SINGLE)");
        CLFFT_status = clfftSetLayout(plans->pl_inv[id], CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
        checkErrFFT(CLFFT_status, "clfftSetLayout(CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED)");
        CLFFT_status = clfftSetResultLocation(plans->pl_inv[id], CLFFT_INPLACE);
        checkErrFFT(CLFFT_status, "clfftSetResultLocation(CLFFT_INPLACE)");
        CLFFT_status = clfftSetPlanScale(plans->pl_inv[id], CLFFT_BACKWARD, (cl_float)((double)sett->interpftpad / sett->Ninterp));
        checkErrFFT(CLFFT_status, "clfftSetResultLocation(CLFFT_INPLACE)");

        CLFFT_status = clfftBakePlan(plans->pl_inv[id],
                                     1,
                                     &cl_handles->exec_queues[id][0],
                                     NULL,
                                     NULL);
        checkErrFFT(CLFFT_status, "clfftBakePlan(plans->pl_int)");
    }

} // plan_fft

void init_openmp(cl_uint count)
{
	omp_set_num_threads(count);
}

void read_checkpoints(Command_line_opts *opts,
                      Search_range *s_range, 
                      int *FNum) {

  if(opts->checkp_flag) {

    // filename of checkpoint state file, depending on the hemisphere
    if(opts->hemi)
      sprintf(opts->qname, "state_%03d_%04d%s_%d.dat",  
              opts->ident, opts->band, opts->label, opts->hemi);
    else
      sprintf(opts->qname, "state_%03d_%04d%s.dat", 
              opts->ident, opts->band, opts->label);

    FILE *state;
    if((state = fopen(opts->qname, "rb")) != NULL) {

      // Scan the state file to get last recorded parameters
      if((fscanf(state, "%d %d %d %d %d",
                 &s_range->pst,
                 &s_range->mst,
                 &s_range->nst,
                 &s_range->sst, FNum)) == EOF)
      {

        // This means that state file is empty (=end of the calculations)
        fprintf(stderr, "State file empty: nothing to do...\n");
        fclose(state);
        return;
      }

    fclose(state);

      // No state file - start from the beginning
    } else {
      s_range->pst = s_range->pmr[0];
      s_range->mst = s_range->mr[0];
      s_range->nst = s_range->nr[0];
      s_range->sst = s_range->spndr[0];
      *FNum = 0;
    } // if state

  } else {
    s_range->pst = s_range->pmr[0];
    s_range->mst = s_range->mr[0];
    s_range->nst = s_range->nr[0];
    s_range->sst = s_range->spndr[0];
    *FNum = 0;
  } // if checkp_flag

} // end reading checkpoints

void cleanup(Detector_settings* ifo,
             Search_settings *sett,
             Command_line_opts *opts,
             Search_range *s_range,
             OpenCL_handles* cl_handles,
             BLAS_handles* blas_handles,
             FFT_plans *plans,
             FFT_arrays *fft_arr,
             Aux_arrays *aux)
{
    cleanup_fft(sett, cl_handles, plans);

    cleanup_blas(sett, cl_handles, blas_handles);

    cleanup_arrays(ifo, sett, cl_handles, opts, aux, fft_arr);

    cleanup_opencl(cl_handles);

	//for (int i = 0; i<sett->nifo; i++)
    //{
    //    free(ifo[i].sig.xDat);
    //    free(ifo[i].sig.DetSSB);
    //
    //    clReleaseMemObject(ifo[i].sig.xDatma_d);
    //    clReleaseMemObject(ifo[i].sig.xDatmb_d);
    //
    //    clReleaseMemObject(ifo[i].sig.aa_d);
    //    clReleaseMemObject(ifo[i].sig.bb_d);
    //
    //    clReleaseMemObject(ifo[i].sig.shft_d);
    //    clReleaseMemObject(ifo[i].sig.shftf_d);
    //}
    //
    //clReleaseMemObject(aux->cosmodf_d);
    //clReleaseMemObject(aux->sinmodf_d);
    //
    //clReleaseMemObject(F_d);
    //
    //clReleaseMemObject(fft_arr->xa_d);
    //
    //free(sett->M);
    //
    //clfftDestroyPlan(&plans->plan);
    //clfftDestroyPlan(&plans->pl_int);
    //clfftDestroyPlan(&plans->pl_inv);
    //
	//clblasTeardown();

} // end of cleanup & memory free

void cleanup_arrays(Detector_settings* ifo,
                    Search_settings* sett,
                    OpenCL_handles* cl_handles,
                    Command_line_opts* opts,
                    Aux_arrays* aux_arr,
                    FFT_arrays* fft_arr)
{
	cleanup_fft_arrays(fft_arr, sett->nifo, cl_handles->dev_count);
}

void cleanup_fft_arrays(FFT_arrays* fft_arr,
                        int nifo,
                        cl_uint count)
{
    for (cl_uint id = 0; id < count; ++id)
    {
      for (int n = 0; n < nifo; ++n)
      {
        cl_int CL_err = CL_SUCCESS;
        
        CL_err = clReleaseMemObject(fft_arr->xa_d[id][n]); checkErr(CL_err, "clReleaseMemObject(fft_arr->xa_d[i][j])");
        CL_err = clReleaseMemObject(fft_arr->xb_d[id][n]); checkErr(CL_err, "clReleaseMemObject(fft_arr->xb_d[i][j])");
    }

    free(fft_arr->xa_d[id]);
    free(fft_arr->xb_d[id]);
  }

  free(fft_arr->xa_d);
  free(fft_arr->xb_d);
}

void cleanup_opencl(OpenCL_handles* cl_handles)
{
	cleanup_kernels(cl_handles->kernels, cl_handles->dev_count);

	cleanup_program(cl_handles->prog);

	cleanup_command_queue_set(cl_handles->read_queues, cl_handles->dev_count);
	cleanup_command_queue_set(cl_handles->exec_queues, cl_handles->dev_count);
	cleanup_command_queue_set(cl_handles->write_queues, cl_handles->dev_count);

	cleanup_context(cl_handles->ctx);

	cleanup_devices(cl_handles->devs, cl_handles->dev_count);
}

void cleanup_devices(cl_device_id* devices,
                     cl_uint count)
{
	for (cl_uint i = 0; i < count; ++i)
	{
		cl_int CL_err = clReleaseDevice(devices[i]);
		checkErr(CL_err, "clReleaseDevice(devices[i])");
	}

	free(devices);
}

void cleanup_context(cl_context ctx)
{
	cl_int CL_err = CL_SUCCESS;

	CL_err = clReleaseContext(ctx);
	checkErr(CL_err, "clReleaseContext(ctx)");
}

void cleanup_command_queue_set(cl_command_queue** queues,
                               size_t count)
{
	for (cl_uint i = 0; i < count; ++i)
	{
		for (cl_uint j = 0; j < MAX_DETECTORS; ++j)
		{
			cl_int CL_err = clReleaseCommandQueue(queues[i][j]);
			checkErr(CL_err, "clReleaseCommandQueue(queues[i][j])");
		}

		free(queues[i]);
	}

	free(queues);
}

void cleanup_program(cl_program prog)
{
	cl_int CL_err = clReleaseProgram(prog);
	checkErr(CL_err, "clReleaseProgram(prog);");
}

void cleanup_kernels(cl_kernel** kernels,
                     cl_uint count)
{
	for (cl_uint i = 0; i < count; ++i)
	{
		for (cl_uint j = 0; j < kernel_count; ++j)
		{
			cl_int CL_err = clReleaseKernel(kernels[i][j]);
			checkErr(CL_err, "clReleaseKernel(kernels[i][j])");
		}

		free(kernels[i]);
	}

	free(kernels);
}

void cleanup_blas(Search_settings* sett,
	              OpenCL_handles* cl_handles,
	              BLAS_handles* blas_handles)
{
    cl_int CL_err = CL_SUCCESS;

    for (cl_uint id = 0; id < cl_handles->dev_count; ++id)
    {
        for (int idet = 0; idet < sett->nifo; ++idet)
        {
            CL_err = clReleaseMemObject(blas_handles->aaScratch_d[id][idet]); checkErr(CL_err, "clReleaseMemObject(blas_handles->aaScratch_d)");
            CL_err = clReleaseMemObject(blas_handles->bbScratch_d[id][idet]); checkErr(CL_err, "clReleaseMemObject(blas_handles->bbScratch_d)");
        }

        free(blas_handles->aaScratch_d[id]);
        free(blas_handles->bbScratch_d[id]);
    }

    free(blas_handles->aaScratch_d);
    free(blas_handles->bbScratch_d);

	clblasTeardown();
}

void cleanup_fft(Search_settings* sett,
	             OpenCL_handles* cl_handles,
	             FFT_plans* plans)
{
	clfftStatus status = CLFFT_SUCCESS;
    for (cl_uint id = 0; id < cl_handles->dev_count; ++id)
    {
        status = clfftDestroyPlan(&plans->plan[id]);   checkErrFFT(status, "clfftDestroyPlan(plans->plan)");
        status = clfftDestroyPlan(&plans->pl_int[id]); checkErrFFT(status, "clfftDestroyPlan(plans->pl_int)");
        status = clfftDestroyPlan(&plans->pl_inv[id]); checkErrFFT(status, "clfftDestroyPlan(plans->pl_inv)");
    }
	clfftTeardown();
}

// Command line options handling: coincidences //

/*void handle_opts_coinc(Search_settings *sett,
                       Command_line_opts_coinc *opts,
                       int argc, 
                       char* argv[])
{

  opts->wd=NULL;

  strcpy (opts->prefix, TOSTR(PREFIX));
  strcpy (opts->dtaprefix, TOSTR(DTAPREFIX));

  // Default initial value of the data sampling time 
  sett->dt = 0.5;

  opts->help_flag=0;
  static int help_flag=0;

  // Default value of the minimal number of coincidences 
  opts->mincoin=3; 

  // Default value of the narrow-down parameter 
  opts->narrowdown=0.5; 

  // Default value of the cell shift: 0000 (no shifts)
  opts->shift=0;

  // Default value of the cell scaling: 1111 (no scaling)
  opts->scale=1111;

  // Default signal-to-noise threshold cutoff
  opts->snrcutoff=6;

  // Reading arguments 

  while (1) {
    static struct option long_options[] = {
      {"help", no_argument, &help_flag, 1},
      // Cell shifts  
      {"shift", required_argument, 0, 's'},
      // Cell scaling 
      {"scale", required_argument, 0, 'z'},
      // Reference frame number 
      {"refr", required_argument, 0, 'r'},
      // output directory
      {"output", required_argument, 0, 'o'},
      // input data directory
      {"data", required_argument, 0, 'd'},
      // fpo value
      {"fpo", required_argument, 0, 'p'},
      // data sampling time 
      {"dt", required_argument, 0, 't'},
      // triggers' name prefactor 
      {"trigname", required_argument, 0, 'e'},
      // Location of the reference grid.bin and starting_date files  
      {"refloc", required_argument, 0, 'g'},
      // Minimal number of coincidences recorded in the output  
      {"mincoin", required_argument, 0, 'm'},
      // Narrow down the frequency band (+- the center of band) 
      {"narrowdown", required_argument, 0, 'n'},
      // Signal-to-noise threshold cutoff  
      {"snrcutoff", required_argument, 0, 'c'},
      {0, 0, 0, 0}
    };

    if (help_flag) {

      printf("polgraw-allsky periodic GWs: search for concidences among candidates\n");
      printf("Usage: ./coincidences -[switch1] <value1> -[switch2] <value2> ...\n") ;
      printf("Switches are:\n\n");
      printf("-data         Data directory (default is ./candidates)\n");
      printf("-output       Output directory (default is ./coinc-results)\n");
      printf("-shift        Cell shifts in fsda directions (4 digit number, e.g. 0101, default 0000)\n");
      printf("-scale        Cell scaling in fsda directions (4 digit number, e.g. 4824, default 1111)\n");
      printf("-refr         Reference frame number\n");
      printf("-fpo          Reference band frequency fpo value\n");
      printf("-dt           Data sampling time dt (default value: 0.5)\n");
      printf("-trigname     Part of triggers' name (for identifying files)\n");
      printf("-refloc       Location of the reference grid.bin and starting_date files\n");
      printf("-mincoin      Minimal number of coincidences recorded\n");
      printf("-narrowdown   Narrow-down the frequency band (range [0, 0.5] +- around center)\n");
      printf("-snrcutoff    Signal-to-noise threshold cutoff (default value: 6)\n\n");

      printf("Also:\n\n");
      printf("--help		This help\n");

      exit (0);
    }

    int option_index = 0;
    int c = getopt_long_only (argc, argv, "p:o:d:s:z:r:t:e:g:m:n:c:", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 'p':
      sett->fpo = atof(optarg);
      break;
    case 's': // Cell shifts 
      opts->shift = (int)atof(optarg); // WARNING: Why atof when assigning to int?
      break;
    case 'z': // Cell scaling   
      opts->scale = atoi(optarg);
      break;
    case 'r':
      opts->refr = atoi(optarg);
      break;
    case 'o':
      strcpy(opts->prefix, optarg);
      break;
    case 'd':
      strcpy(opts->dtaprefix, optarg);
      break;
    case 't':
      sett->dt = atof(optarg);
      break;
    case 'e':
      strcpy(opts->trigname, optarg);
      break;
    case 'g':
      strcpy(opts->refloc, optarg);
      break;
    case 'm':
      opts->mincoin = atoi(optarg);
      break;
    case 'n':
      opts->narrowdown = atof(optarg);
      break;
    case 'c':
      opts->snrcutoff = atof(optarg);
      break;
    case '?':
      break;
    default:
      break ;
    } // switch c
  } // while 1

  // Putting the parameter in triggers' frequency range [0, pi] 
  opts->narrowdown *= M_PI; 

  printf("#mb add info at the beginning...\n"); 
  printf("The SNR threshold cutoff is %.12f, ", opts->snrcutoff); 
  printf("corresponding to F-statistic value of %.12f\n", 
    pow(opts->snrcutoff, 2)/2. + 2); 

}*/ // end of command line options handling: coincidences  



#if 0
/* Manage grid matrix (read from grid.bin, find eigenvalues 
 * and eigenvectors) and reference GPS time from starting_time
 * (expected to be in the same directory)    
 */ 

void manage_grid_matrix(
            Search_settings *sett, 
            Command_line_opts_coinc *opts) {

  sett->M = (double *)calloc(16, sizeof (double));

  FILE *data;
  char filename[512];
  sprintf (filename, "%s/grid.bin", opts->refloc);

  if ((data=fopen (filename, "r")) != NULL) {

    printf("Reading the reference grid.bin at %s\n", opts->refloc);

    fread ((void *)&sett->fftpad, sizeof (int), 1, data);

    printf("fftpad from the grid file: %d\n", sett->fftpad); 

    fread ((void *)sett->M, sizeof(double), 16, data);
    // We actually need the second (Fisher) matrix from grid.bin, 
    // hence the second fread: 
    fread ((void *)sett->M, sizeof(double), 16, data);
    fclose (data);
  } else {
    perror (filename);
    exit(EXIT_FAILURE);
  }

  /* //#mb seems not needed at the moment 
     sprintf (filename, "%s/starting_date", opts->refloc);

     if ((data=fopen (filename, "r")) != NULL) {
     fscanf(data, "%le", &opts->refgps);

     printf("Reading the reference starting_date file at %s The GPS time is %12f\n", opts->refloc, opts->refgps);
     fclose (data);
     } else {
     perror (filename);
     exit(EXIT_FAILURE);
     }
  */ 

  // Calculating the eigenvectors and eigenvalues 
  gsl_matrix_view m = gsl_matrix_view_array(sett->M, 4, 4);

  gsl_vector *eval = gsl_vector_alloc(4);
  gsl_matrix *evec = gsl_matrix_alloc(4, 4);

  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(4); 
  gsl_eigen_symmv(&m.matrix, eval, evec, w);
  gsl_eigen_symmv_free(w);

  double eigval[4], eigvec[4][4]; 
  // Saving the results to the settings struct sett->vedva[][]
  { int i, j;
    for(i=0; i<4; i++) { 
      eigval[i] = gsl_vector_get(eval, i); 
      gsl_vector_view evec_i = gsl_matrix_column(evec, i);

      for(j=0; j<4; j++)   
    eigvec[j][i] = gsl_vector_get(&evec_i.vector, j);               
    } 

    // This is an auxiliary matrix composed of the eigenvector 
    // columns multiplied by a matrix with sqrt(eigenvalues) on diagonal  
    for(i=0; i<4; i++) { 
      for(j=0; j<4; j++) { 
    sett->vedva[i][j]  = eigvec[i][j]*sqrt(eigval[j]); 
    //        printf("%.12le ", sett->vedva[i][j]); 
      } 
      //      printf("\n"); 
    }

  } 

  /* 
  //#mb matrix generated in matlab, for tests 
  double _tmp[4][4] = { 
  {-2.8622034614137332e-001, -3.7566564762376159e-002, -4.4001551065376701e-012, -3.4516253934827171e-012}, 
  {-2.9591999145463371e-001, 3.6335210834374479e-002, 8.1252443441098394e-014, -6.8170555119669981e-014}, 
  {1.5497867603229576e-005, 1.9167007413107127e-006, 1.0599051611325639e-008, -5.0379548388381567e-008}, 
  {2.4410008440913992e-005, 3.2886518554938671e-006, -5.7338464150027107e-008, -9.3126913365595100e-009},
  };

  { int i,j; 
  for(i=0; i<4; i++) 
  for(j=0; j<4; j++) 
  sett->vedva[i][j]  = _tmp[i][j]; 
  }

  printf("\n"); 

  { int i, j; 
  for(i=0; i<4; i++) { 
  for(j=0; j<4; j++) {
  printf("%.12le ", sett->vedva[i][j]);
  }
  printf("\n"); 
  } 

  } 
  */ 

  gsl_vector_free (eval);
  gsl_matrix_free (evec);

} // end of manage grid matrix  

#endif
