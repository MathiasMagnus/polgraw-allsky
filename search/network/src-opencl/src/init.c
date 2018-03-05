// C behavioral defines
//
// MSVC: macro to include constants, such as M_PI (include before math.h)
#define _USE_MATH_DEFINES

// OpenCL behavioral defines
//
// 1.2+ OpenCL headers: tells the headers not to bitch about clCreateCommandQueue being renamed to clCreateCommandQueueWithProperties
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1

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


/// <summary>Command line options handling: search</summary>
///
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

/// <summary>Initialize OpenCL devices based on user preference.</summary>
/// <remarks>Currently, only a sinle platform can be selected.</remarks>
///
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

    char* source = load_program_file(kernel_path);

    cl_handles->prog = build_program_source(cl_handles->ctx, source);

    cl_handles->kernels = create_kernels(cl_handles->prog);

    free(source);
}

/// <summary>Tries selecting the platform with the specified index.</summary>
///
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

/// <summary>Selects all devices of the specified type.</summary>
///
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
        result = (cl_device_id*)malloc(*count * sizeof(cl_device_id));
        CL_err = clGetDeviceIDs(platform, dev_type, *count, result, 0);
        checkErr(CL_err, "clGetDeviceIDs(devices)");

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

/// <summary>Create a contxet that holds all specified devices.</summary>
///
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

/// <summary>Create a set of command queues to all the devices in the context.</summary>
///
cl_command_queue* create_command_queue_set(cl_context context)
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint count = 0;
    cl_device_id* devices = NULL;
    cl_command_queue* result = NULL;

    CL_err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &count, NULL);
    checkErr(CL_err, "clGetContextInfo(CL_CONTEXT_NUM_DEVICES)");

    devices = (cl_device_id*)malloc(count * sizeof(cl_device_id));

    CL_err = clGetContextInfo(context, CL_CONTEXT_DEVICES, count * sizeof(cl_device_id), devices, NULL);
    checkErr(CL_err, "clGetContextInfo(CL_CONTEXT_DEVICES)");

    result = (cl_command_queue*)malloc(count * sizeof(cl_command_queue));

    for (cl_uint i = 0; i < count; ++i)
    {
        result[i] = clCreateCommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &CL_err);
        checkErr(CL_err, "clCreateCommandQueue()");

        CL_err = clReleaseDevice(devices[i]);
        checkErr(CL_err, "clReleaseDevice()");
    }

    free(devices);

    return result;
}

/// <summary>Load kernel file from disk.</summary>
///
char* load_program_file(const char* filename)
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

/// <summary>Build program file.</summary>
///
cl_program build_program_source(cl_context context,
                                const char* source)
{
    cl_int CL_err = CL_SUCCESS;
    cl_program result = NULL;

    cl_uint numDevices = 0;
    cl_device_id* devices = NULL;
#ifdef _WIN32
    const size_t length = strnlen_s(source, UINT_MAX);
#else
    const size_t length = strlen(source);
#endif

    result = clCreateProgramWithSource(context, 1, &source, &length, &CL_err);
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
    //strcat(build_params, " -Werror" ); // Warnings will be treated like errors, this is useful for debug
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

    return result;
}

/// <summary>Create a kernel for all entry points in the program.</summary>
///
cl_kernel* create_kernels(cl_program program)
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint kernel_count = 11;
    cl_kernel* result = (cl_kernel*)malloc(kernel_count * sizeof(cl_kernel));

    for (cl_uint i = 0; i < kernel_count; ++i)
        result[i] = obtain_kernel(program, i);

    return result;
}

/// <summary>Obtain the name of the kernel of a given index.</summary>
///
const char* obtain_kernel_name(cl_uint i)
{
    const char* result = NULL;

    switch (i)
    {
    case ComputeSinCosModF:
        result = "compute_sincosmodf";
        break;
    case Modvir:
        result = "modvir_kern";
        break;
    case TShiftPMod:
        result = "tshift_pmod_kern";
        break;
    case ResamplePostFFT:
        result = "resample_postfft";
        break;
    case ComputeB:
        result = "computeB";
        break;
    case TriDiagMul:
        result = "tridiagMul";
        break;
    case Interpolate:
        result = "interpolate";
        break;
    case PhaseMod1:
        result = "phase_mod_1";
        break;
    case PhaseMod2:
        result = "phase_mod_2";
        break;
    case ComputeFStat:
        result = "compute_Fstat";
        break;
    case FStatSimple:
        result = "fstat_norm_simple";
        break;
    default:
        perror("Unkown kernel index");
        exit(-1);
        break;
    }

    return result;
}

/// <summary>Obtain kernel with the specified index.</summary>
///
cl_kernel obtain_kernel(cl_program program, cl_uint i)
{
    cl_int CL_err = CL_SUCCESS;
    cl_kernel result = NULL;

    result = clCreateKernel(program, obtain_kernel_name(i), &CL_err);
    checkErr(CL_err, "clCreateKernel()");

    return result;
}

/// <summary>Generate grid from the M matrix.</summary>
/// <remarks>Processes the file 'grid.bin'</remarks>
///
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

/// <summary>Initialize auxiliary and F-statistics arrays.</summary>
///
void init_arrays(Detector_settings* ifo,
                 Search_settings* sett,
                 OpenCL_handles* cl_handles,
                 Command_line_opts* opts,
                 Aux_arrays *aux_arr,
                 cl_mem* F_d)
{
    cl_int CL_err = CL_SUCCESS;
    int i;
    size_t status;

    // Allocates and initializes to zero the data, detector ephemeris
    // and the F-statistic arrays

    FILE *data;

    sett->Ninterp = sett->interpftpad*sett->nfft;
    sett->nfftf = sett->fftpad*sett->nfft;

    for (i = 0; i<sett->nifo; i++)
    {
        ifo[i].sig.xDat = (real_t*)calloc(sett->N, sizeof(real_t));

        ifo[i].sig.xDat_d = clCreateBuffer(cl_handles->ctx,
                                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                           sett->N * sizeof(real_t),
                                           ifo[i].sig.xDat,
                                           &CL_err);
        checkErr(CL_err, "clCreateBuffer(ifo[i].sig.xDat_d)");

        // Input time-domain data handling
        // 
        // The file name ifo[i].xdatname is constructed 
        // in settings.c, while looking for the detector 
        // subdirectories
        if ((data = fopen(ifo[i].xdatname, "rb")) != NULL)
        {
            status = fread((void *)(ifo[i].sig.xDat),
                           sizeof(real_t),
                           sett->N,
                           data);
            fclose(data);
        }
        else
        {
            perror(ifo[i].xdatname);
            exit(EXIT_FAILURE);
        }

        int j, Nzeros = 0;
        // Checking for null values in the data
        for (j = 0; j < sett->N; j++)
            if (!ifo[i].sig.xDat[j]) Nzeros++;

        ifo[i].sig.Nzeros = Nzeros;

        // factor N/(N - Nzeros) to account for null values in the data
        ifo[i].sig.crf0 = (real_t)sett->N / (sett->N - ifo[i].sig.Nzeros);

        // Estimation of the variance for each detector 
        ifo[i].sig.sig2 = (ifo[i].sig.crf0)*var(ifo[i].sig.xDat, sett->N);

        ifo[i].sig.DetSSB = (real_t*)calloc(3 * sett->N, sizeof(real_t));

        ifo[i].sig.DetSSB_d = clCreateBuffer(cl_handles->ctx,
                                             CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                             3 * sett->N * sizeof(real_t),
                                             ifo[i].sig.DetSSB,
                                             &CL_err);
        checkErr(CL_err, "clCreateBuffer(ifo[i].sig.DetSSB_d)");

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
            status = fread((void *)(ifo[i].sig.DetSSB),
                           sizeof(real_t),
                           3 * sett->N,
                           data);

            // Deterministic phase defining the position of the Earth
            // in its diurnal motion at t=0 
            status = fread((void *)(&ifo[i].sig.phir),
                           sizeof(real_t),
                           1,
                           data);

            // Earth's axis inclination to the ecliptic at t=0
            status = fread((void *)(&ifo[i].sig.epsm),
                           sizeof(real_t),
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

        // sincos 
        ifo[i].sig.sphir = sin(ifo[i].sig.phir);
        ifo[i].sig.cphir = cos(ifo[i].sig.phir);
        ifo[i].sig.sepsm = sin(ifo[i].sig.epsm);
        ifo[i].sig.cepsm = cos(ifo[i].sig.epsm);

        sett->sepsm = ifo[i].sig.sepsm;
        sett->cepsm = ifo[i].sig.cepsm;

        ifo[i].sig.xDatma_d = clCreateBuffer(cl_handles->ctx,
                                             CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                             sett->N * sizeof(complex_devt),
                                             NULL,
                                             &CL_err);
        checkErr(CL_err, "clCreateBuffer(ifo[i].sig.xDatma_d)");

        ifo[i].sig.xDatmb_d = clCreateBuffer(cl_handles->ctx,
                                             CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                             sett->N * sizeof(complex_devt),
                                             NULL,
                                             &CL_err);
        checkErr(CL_err, "clCreateBuffer(ifo[i].sig.xDatmb_d)");

        ifo[i].sig.aa_d = clCreateBuffer(cl_handles->ctx,
                                         CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                         sett->N * sizeof(real_t),
                                         NULL,
                                         &CL_err);
        checkErr(CL_err, "clCreateBuffer(ifo[i].sig.aa_d)");

        ifo[i].sig.bb_d = clCreateBuffer(cl_handles->ctx,
                                         CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                         sett->N * sizeof(real_t),
                                         NULL,
                                         &CL_err);
        checkErr(CL_err, "clCreateBuffer(ifo[i].sig.bb_d)");

        ifo[i].sig.shft_d = clCreateBuffer(cl_handles->ctx,
                                           CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                           sett->N * sizeof(real_t),
                                           NULL,
                                           &CL_err);
        checkErr(CL_err, "clCreateBuffer(ifo[i].sig.shft_d)");

        ifo[i].sig.shftf_d = clCreateBuffer(cl_handles->ctx,
                                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                            sett->N * sizeof(real_t),
                                            NULL,
                                            &CL_err);
        checkErr(CL_err, "clCreateBuffer(ifo[i].sig.shftf_d)");

    } // end loop for detectors 

      // Check if the ephemerids have the same epsm parameter
    for (i = 1; i<sett->nifo; i++)
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

    *F_d = clCreateBuffer(cl_handles->ctx,
                          CL_MEM_READ_WRITE,
                          2 * sett->nfft * sizeof(real_t),
                          NULL,
                          &CL_err);
    checkErr(CL_err, "clCreateBuffer(F_d)");

    // Auxiliary arrays, Earth's rotation

    aux_arr->t2_d = clCreateBuffer(cl_handles->ctx,
                                   CL_MEM_READ_WRITE,
                                   sett->N * sizeof(real_t),
                                   NULL,
                                   &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->t2_d)");

    aux_arr->cosmodf_d = clCreateBuffer(cl_handles->ctx,
                                        CL_MEM_READ_WRITE,
                                        sett->N * sizeof(real_t),
                                        NULL,
                                        &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->cosmodf_d)");

    aux_arr->sinmodf_d = clCreateBuffer(cl_handles->ctx,
                                        CL_MEM_READ_WRITE,
                                        sett->N * sizeof(real_t),
                                        NULL,
                                        &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->sinmodf_d)");

    aux_arr->tshift_d = clCreateBuffer(cl_handles->ctx,
                                       CL_MEM_READ_WRITE,
                                       sett->N * sizeof(real_t),
                                       NULL,
                                       &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->tshift_d)");

    aux_arr->ifo_amod_d = clCreateBuffer(cl_handles->ctx,
                                         CL_MEM_READ_ONLY,
                                         sett->nifo * sizeof(Ampl_mod_coeff),
                                         NULL,
                                         &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->ifo_amod_d)");

    aux_arr->maa_d = clCreateBuffer(cl_handles->ctx,
                                    CL_MEM_READ_ONLY,
                                    sizeof(real_t),
                                    NULL,
                                    &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->maa_d)");

    aux_arr->mbb_d = clCreateBuffer(cl_handles->ctx,
                                    CL_MEM_READ_ONLY,
                                    sizeof(real_t),
                                    NULL,
                                    &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->mbb_d)");

    init_spline_matrices(cl_handles,
                         &aux_arr->diag_d,
                         &aux_arr->ldiag_d,
                         &aux_arr->udiag_d,
                         &aux_arr->B_d,
                         sett->Ninterp);

    CL_err = clSetKernelArg(cl_handles->kernels[ComputeSinCosModF], 0, sizeof(cl_mem), &aux_arr->sinmodf_d);
    checkErr(CL_err, "clSetKernelArg(0)");
    CL_err = clSetKernelArg(cl_handles->kernels[ComputeSinCosModF], 1, sizeof(cl_mem), &aux_arr->cosmodf_d);
    checkErr(CL_err, "clSetKernelArg(1)");
    CL_err = clSetKernelArg(cl_handles->kernels[ComputeSinCosModF], 2, sizeof(real_t), &sett->omr);
    checkErr(CL_err, "clSetKernelArg(2)");
    CL_err = clSetKernelArg(cl_handles->kernels[ComputeSinCosModF], 3, sizeof(cl_int), &sett->N);
    checkErr(CL_err, "clSetKernelArg(3)");

    cl_event exec;
    size_t size_N = (size_t)sett->N; // Variable so pointer can be given to API

    CL_err = clEnqueueNDRangeKernel(cl_handles->exec_queues[0],
                                    cl_handles->kernels[ComputeSinCosModF],
                                    1,
                                    NULL,
                                    &size_N,
                                    NULL,
                                    0,
                                    NULL,
                                    &exec);
    checkErr(CL_err, "clEnqueueNDRangeKernel(ComputeSinCosModF)");

    CL_err = clWaitForEvents(1, &exec);
    checkErr(CL_err, "clWaitForEvents(exec)");

    // OpenCL cleanup
    clReleaseEvent(exec);

} // end of init arrays 

/// <summary>Set search ranges based on user preference.</summary>
///
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

/// <summary>Sets up BLAS plans.</summary>
///
void init_blas(Search_settings* sett,
               OpenCL_handles* cl_handles,
               BLAS_handles* blas_handles)
{
    blas_handles->BLAS_err = clblasSetup();
    checkErr(blas_handles->BLAS_err, "clblasSetup()");
}

/// <summary>Sets up FFT plans.</summary>
///
void plan_fft(Search_settings* sett,
              OpenCL_handles* cl_handles,
              FFT_plans* plans,
              FFT_arrays* fft_arr)
{
    cl_int CL_err = CL_SUCCESS;
    clfftStatus CLFFT_status = CLFFT_SUCCESS;

    fft_arr->arr_len = (sett->fftpad*sett->nfft > sett->Ninterp ?
                        sett->fftpad*sett->nfft :
                        sett->Ninterp);

    fft_arr->xa_d = clCreateBuffer(cl_handles->ctx,
                                   CL_MEM_READ_WRITE,
                                   fft_arr->arr_len * sizeof(complex_t),
                                   NULL,
                                   &CL_err);
    checkErr(CL_err, "clCreateBuffer(fft_arr->xa_d)");

    fft_arr->xb_d = clCreateBuffer(cl_handles->ctx,
                                   CL_MEM_READ_WRITE,
                                   fft_arr->arr_len * sizeof(complex_t),
                                   NULL,
                                   &CL_err);
    checkErr(CL_err, "clCreateBuffer(fft_arr->xb_d)");

    clfftSetupData fftSetup;
    CLFFT_status = clfftSetup(&fftSetup);
    checkErrFFT(CLFFT_status, "clffftSetup");

    clfftDim dim = CLFFT_1D;
    size_t size_arr_len = (size_t)fft_arr->arr_len;
    CLFFT_status = clfftCreateDefaultPlan(&plans->plan, cl_handles->ctx, dim, &size_arr_len);
    checkErrFFT(CLFFT_status, "clCreateDefaultPlan");

    CLFFT_status = clfftSetPlanPrecision(plans->plan, CLFFT_TRANSFORM_PRECISION);
    checkErrFFT(CLFFT_status, "clfftSetPlanPrecision(CLFFT_SINGLE)");
    CLFFT_status = clfftSetLayout(plans->plan, CLFFT_TRANSFORM_LAYOUT, CLFFT_TRANSFORM_LAYOUT);
    checkErrFFT(CLFFT_status, "clfftSetLayout(CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED)");
    CLFFT_status = clfftSetResultLocation(plans->plan, CLFFT_INPLACE);
    checkErrFFT(CLFFT_status, "clfftSetResultLocation(CLFFT_INPLACE)");

    CLFFT_status = clfftBakePlan(plans->plan,
                           1u,//cl_handles->dev_count,
                           cl_handles->exec_queues,
                           NULL,
                           NULL);
    checkErrFFT(CLFFT_status, "clfftBakePlan()");

} // plan_fft


// Checkpointing //

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
      if((fscanf(state, "%d %d %d %d %d", &s_range->pst, &s_range->mst,
         &s_range->nst, &s_range->sst, FNum)) == EOF) {

    // This means that state file is empty (=end of the calculations)
    fprintf (stderr, "State file empty: nothing to do...\n");
    fclose (state);
    return;

      }

      fclose (state);

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

/// <summary>Frees all resources for termination.</summary>
///
void cleanup(Detector_settings* ifo,
             Search_settings *sett,
             Command_line_opts *opts,
             Search_range *s_range,
             OpenCL_handles* cl_handles,
             BLAS_handles* blas_handles,
             FFT_plans *plans,
             FFT_arrays *fft_arr,
             Aux_arrays *aux,
             cl_mem F_d)
{
    for (int i = 0; i<sett->nifo; i++)
    {
        free(ifo[i].sig.xDat);
        free(ifo[i].sig.DetSSB);

        clReleaseMemObject(ifo[i].sig.xDatma_d);
        clReleaseMemObject(ifo[i].sig.xDatmb_d);

        clReleaseMemObject(ifo[i].sig.aa_d);
        clReleaseMemObject(ifo[i].sig.bb_d);

        clReleaseMemObject(ifo[i].sig.shft_d);
        clReleaseMemObject(ifo[i].sig.shftf_d);
    }

    clReleaseMemObject(aux->cosmodf_d);
    clReleaseMemObject(aux->sinmodf_d);
    clReleaseMemObject(aux->t2_d);

    clReleaseMemObject(F_d);

    clReleaseMemObject(fft_arr->xa_d);

    free(sett->M);

    clfftDestroyPlan(&plans->plan);
    //clfftDestroyPlan(&plans->pl_int);
    //clfftDestroyPlan(&plans->pl_inv);

    clfftTeardown();

} // end of cleanup & memory free 



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

/*---------------------------------------------------------------------------*/

/*
  Initialize CUDA: cuinit
  - sets cuda device to (in priority order): cdev, 0 
  - returns: device id or -1 on error
*/
int cuinit(int cdev)
{
  //int dev, deviceCount = 0;
  //cudaDeviceProp deviceProp;
  
//  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
//    printf("ERROR: cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
//    return(-1);
//  }
//  if (deviceCount == 0) {
//    printf("ERROR: There is no device supporting CUDA\n");
//    return(-1);
//  }
//  if (cdev < 0 && cdev >= deviceCount) {
//    printf("\nWARNING: Device %d is not available! Trying device 0\n", cdev);
//    cdev = 0;
//  }
//
//  printf("__________________________________CUDA devices___________________________________\n");
//  printf("Set | ID |        Name        |   Gmem(B)   | Smem(B) | Cmem(B) | C.Cap. | Thr/bl |\n");
//  
//  for (dev = 0; dev < deviceCount; ++dev) {
//    cudaGetDeviceProperties(&deviceProp, dev);
//    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
//      printf("- | %1d | %16s | Error | Error | Error | Error | Error |\n", dev, deviceProp.name );
//      if ( dev==cdev ) {
//	printf("ERROR: Can't set device %d\n", cdev);
//	return(-1);
//      }
//    }
//    if (dev==cdev) {
//      printf(" *  |");
//      cudaSetDevice(cdev);
//    } else {
//      printf("    |");
//    }
//    printf(" %1d  | %18.18s | %11Zu | %7Zu | %7Zu |   %d.%d  | %6d |\n", 
//	   dev, deviceProp.name, deviceProp.totalGlobalMem, deviceProp.sharedMemPerBlock, 
//	   deviceProp.totalConstMem, deviceProp.major, deviceProp.minor, deviceProp.maxThreadsPerBlock );
//  }
//  printf("---------------------------------------------------------------------------------\n");
//  
//  /* enable mapped memory */
//  cudaSetDeviceFlags(cudaDeviceMapHost);
//
//  /* force initialization */
//  cudaThreadSynchronize();
  return(cdev);
}
