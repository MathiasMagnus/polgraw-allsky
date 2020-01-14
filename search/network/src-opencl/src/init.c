// C behavioral defines
//
// MSVC: macro to include constants, such as M_PI (include before math.h)
#define _USE_MATH_DEFINES
// MSVC: macro to disable min/max macros on Windows
#define NOMINMAX
// ISO: request safe versions of functions
#define __STDC_WANT_LIB_EXT1__ 1

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
#include <stdbool.h>


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

  strcpy(opts->plat_ids, "0");
  strcpy(opts->dev_types, "def");
  strcpy(opts->dev_ids, "0");

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
      // which platform id to use
      {"platform", required_argument, 0, 'P'},
      // which device type to use
      {"type", required_argument, 0, 'T'}, 
      // which device id to use
      {"device", required_argument, 0, 'D'},
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
      printf("-P, -platform     OpenCL platform ids to use [int],... (default is 0)\n\n");
      printf("-T, -type         OpenCL device types to use [cpu|gpu|acc|all],... (default maps to CL_DEVICE_TYPE_DEFAULT)\n\n");
      printf("-D, -device       OpenCL device ids to use [int],... (default is 0)\n\n");

      printf("Also:\n\n");
      printf("--whitenoise      White Gaussian noise assumed\n");
      printf("--nospindown      Spindowns neglected\n");
      printf("--nocheckpoint    State file won't be created (no checkpointing)\n");
      printf("--help            This help\n");

      exit(EXIT_SUCCESS);
    }

    int option_index = 0;
    int c = getopt_long_only(argc, argv, "i:b:o:d:l:r:g:c:t:h:p:x:s:u:P:T:D:", 
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
    case 'P':
      strcpy(opts->plat_ids, optarg);
      break;
    case 'T':
      strcpy(opts->dev_types, optarg);
      break;
    case 'D':
      strcpy(opts->dev_ids, optarg);
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

  if (strlen(opts->plat_ids))
  {
    cl_sett->count = 0;

    char* token = strtok(opts->plat_ids, ",");
    while (token != NULL)
    {
      char tmp[2];
      strncpy(tmp, token, 1); tmp[1] = '\0';

      cl_sett->plat_ids[cl_sett->count++] = atoi(tmp);

      token = strtok(NULL, ",");
    }
  }

  if (strlen(opts->dev_types))
  {
      int count = 0;

      char* token = strtok(opts->dev_types, ",");
      while (token != NULL)
      {
          if (!strncmp(token, "def", 3))
              cl_sett->dev_types[count++] = CL_DEVICE_TYPE_DEFAULT;
          if (!strncmp(token, "cpu", 3))
              cl_sett->dev_types[count++] = CL_DEVICE_TYPE_CPU;
          if (!strncmp(token, "gpu", 3))
              cl_sett->dev_types[count++] = CL_DEVICE_TYPE_GPU;
          if (!strncmp(token, "acc", 3))
              cl_sett->dev_types[count++] = CL_DEVICE_TYPE_ACCELERATOR;
          if (!strncmp(token, "all", 3))
              cl_sett->dev_types[count++] = CL_DEVICE_TYPE_ALL;

          token = strtok(NULL, ",");
      }

      if (count != (int)cl_sett->count)
          printf("ERROR: Different number of platform ids and device types provided: %d vs. %d\n", cl_sett->count, count);
  }

  if (strlen(opts->dev_ids))
  {
      int count = 0;

      char* token = strtok(opts->dev_ids, ",");
      while (token != NULL)
      {
          char tmp[2];
          strncpy(tmp, token, 1); tmp[1] = '\0';

          cl_sett->dev_ids[count++] = atoi(tmp);

          token = strtok(NULL, ",");
      }

      if (count != (int)cl_sett->count)
          printf("ERROR: Different number of platform ids and device ids provided: %d vs. %d\n", cl_sett->count, count);
  }

} // end of command line options handling 

void init_opencl(OpenCL_handles* cl_handles,
                 OpenCL_settings* cl_sett)
{
  cl_handles->count = cl_sett->count;

  cl_handles->plats = select_platforms(cl_sett->count,
                                       cl_sett->plat_ids);

  cl_handles->devs = select_devices(cl_sett->count,
                                    cl_sett->dev_types,
                                    cl_handles->plats,
                                    cl_sett->dev_ids);

  cl_handles->ctxs = create_contexts(cl_handles->count,
                                     cl_handles->devs);

  cl_handles->write_queues = create_command_queue_set(cl_handles->count, cl_handles->ctxs);
  cl_handles->exec_queues  = create_command_queue_set(cl_handles->count, cl_handles->ctxs);
  cl_handles->read_queues  = create_command_queue_set(cl_handles->count, cl_handles->ctxs);

  {
    char** sources = load_kernel_sources();

    cl_handles->progs = build_programs_with_sources(cl_handles->count,
                                                    cl_handles->ctxs,
                                                    (const char**)sources);

    free_kernel_sources(sources);
  }

  cl_handles->kernels = create_kernels(cl_handles->count, cl_handles->progs);
}

cl_platform_id* select_platforms(cl_uint count, cl_uint ids[MAX_DEVICES])
{
  // Workaround for Nvidia kernel cache bug: https://stackoverflow.com/a/31339617/1476661
#ifdef _WIN32
  _putenv_s("CUDA_CACHE_DISABLE", "1");
#else
  setenv("CUDA_CACHE_DISABLE", "1", 1);
#endif
    cl_int CL_err = CL_SUCCESS;
    cl_uint numPlatforms = 0;

    CL_err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(CL_err, "clGetPlatformIDs(numPlatforms)");

    if (numPlatforms == 0)
    {
        perror("No OpenCL platform detected.");
        exit(-1);
    }

    cl_platform_id *platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id)),
                   *result = (cl_platform_id*)malloc(count * sizeof(cl_platform_id));
    CL_err = clGetPlatformIDs(numPlatforms, platforms, NULL); checkErr(CL_err, "clGetPlatformIDs(platforms)");

    for (cl_uint i = 0; i < count; ++i)
    {
        if (ids[i] < numPlatforms)
        {
            result[i] = platforms[ids[i]];
        }
        else
        {
            printf("ERROR: User requested platform id does not exist. Id %d vs. numPlatforms %d\n", ids[i], numPlatforms);
            exit(-1);
        }
    }

    return result;
}

cl_device_id* select_devices(cl_uint count,
                             cl_device_type* dev_types,
                             cl_platform_id* plat_ids,
                             cl_uint* dev_ids)
{
    cl_int CL_err = CL_SUCCESS;
    cl_device_id* result = (cl_device_id*)malloc(count * sizeof(cl_device_id));

    for (cl_uint i = 0; i < count; ++i)
    {
        cl_uint plat_devs;
        CL_err = clGetDeviceIDs(plat_ids[i], dev_types[i], 0, 0, &plat_devs); checkErr(CL_err, "clGetDeviceIDs(numDevices)");

        cl_device_id* devs = (cl_device_id*)malloc(plat_devs * sizeof(cl_device_id));
        CL_err = clGetDeviceIDs(plat_ids[i], dev_types[i], plat_devs, devs, 0); checkErr(CL_err, "clGetDeviceIDs(devices)");

        if (dev_ids[i] < plat_devs)
        {
            result[i] = devs[dev_ids[i]];
            CL_err = clRetainDevice(result[i]); checkErr(CL_err, "clRetainDevice(result[i])");
        }
        else
        {
            printf("ERROR: User requested device id %d on platform %d not exist.\n", dev_ids[i], i);
            exit(-1);
        }

        for (cl_uint j = 0; j < plat_devs; ++j)
        {
            CL_err = clReleaseDevice(devs[j]); checkErr(CL_err, "clReleaseDevice(devs[j])");
        }
        free(devs);
    }

#ifdef WIN32
    printf_s("Selected OpenCL device(s):\n"); // TODO: don't throw away error code.
#else
    printf("Selected OpenCL device(s):\n");
#endif

    for (cl_uint i = 0; i < count; ++i)
    {
        char pbuf[100];
        CL_err = clGetDeviceInfo(result[i], CL_DEVICE_NAME, sizeof(pbuf), pbuf, NULL); checkErr(CL_err, "clGetDeviceInfo(CL_DEVICE_NAME)");
#ifdef WIN32
        printf_s("\t%s\n", pbuf); // TODO: don't throw away error code.
#else
        printf("\t%s\n", pbuf);
#endif
    }

    // If there are both CPUs and non-CPUs used, break up the CPU device to leave headroom to drive the non-CPU devices.
    bool non_cpu_found = false;
    for (cl_uint i = 0; i < count; ++i)
    {
      cl_device_type type;
      CL_err = clGetDeviceInfo(result[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL); checkErr(CL_err, "clGetDeviceInfo(CL_DEVICE_TYPE)");

      if (type != CL_DEVICE_TYPE_CPU) non_cpu_found = true;
    }
    for (cl_uint i = 0; i < count; ++i)
    {
      cl_device_type type;
      CL_err = clGetDeviceInfo(result[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL); checkErr(CL_err, "clGetDeviceInfo(CL_DEVICE_TYPE)");

      if (type == CL_DEVICE_TYPE_CPU && non_cpu_found)
      {
        cl_uint cu_count;
        CL_err = clGetDeviceInfo(result[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &cu_count, NULL); checkErr(CL_err, "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");

        cl_uint target_cu_count = 1 >= (cl_int)cu_count - ((cl_int)count - 1) ? 1 : cu_count - (count - 1);
        const cl_device_partition_property props[] =
        { CL_DEVICE_PARTITION_BY_COUNTS,
          target_cu_count,
          CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0 };
        cl_device_id sub_device;
        CL_err = clCreateSubDevices(result[i], props, 1, &sub_device, NULL); checkErr(CL_err, "clCreateSubDevices(CL_DEVICE_PARTITION_BY_COUNTS)");

        // Override complete CPU device with sub-device (we leak the parent device (on purpose))
        result[i] = sub_device;
#ifdef WIN32
        printf_s("Selected CPU device is using %d of %d compute-units.\n", target_cu_count, cu_count); // TODO: don't throw away error code.
#else
        printf("Selected CPU device is using %d of %d compute-units.\n", target_cu_count, cu_count);
#endif
      }
    }

    return result;
}

cl_context* create_contexts(cl_uint count,
                            cl_device_id* devices)
{
    cl_int CL_err = CL_SUCCESS;
    cl_context* result = (cl_context*)malloc(count * sizeof(cl_context));

    for (cl_uint i = 0; i < count; ++i)
    {
        cl_platform_id platform = NULL;

        CL_err = clGetDeviceInfo(devices[i], CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);
        checkErr(CL_err, "clGetDeviceInfo(CL_DEVICE_PLATFORM)");

        cl_context_properties cps[3];
        cps[0] = CL_CONTEXT_PLATFORM;
        cps[1] = (cl_context_properties)platform;
        cps[2] = 0;

        result[i] = clCreateContext(cps, 1, &devices[i], NULL, NULL, &CL_err); checkErr(CL_err, "clCreateContext()");
    }

    return result;
}

cl_command_queue** create_command_queue_set(cl_uint count, cl_context* contexts)
{
    cl_int CL_err = CL_SUCCESS;

    cl_command_queue** result = (cl_command_queue**)malloc(count * sizeof(cl_command_queue*));

    for (cl_uint i = 0; i < count; ++i)
    {
        result[i] = (cl_command_queue*)malloc(MAX_DETECTORS * sizeof(cl_command_queue));

        cl_uint num_devices;
        CL_err = clGetContextInfo(contexts[i], CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL); checkErr(CL_err, "clGetContextInfo(CL_CONTEXT_NUM_DEVICES)");

        if (num_devices != 1) checkErr(EXIT_FAILURE, "Unexpectedly found more than one device in context");

        cl_device_id device;
        CL_err = clGetContextInfo(contexts[i], CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL); checkErr(CL_err, "clGetContextInfo(CL_CONTEXT_DEVICES)");

        for (cl_uint j = 0; j < MAX_DETECTORS; ++j)
        {
            result[i][j] = clCreateCommandQueue(contexts[i], device, CL_QUEUE_PROFILING_ENABLE, &CL_err); checkErr(CL_err, "clCreateCommandQueue()");
        }
    }

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

    for (size_t i = 0; i < kernel_path_count; ++i)
        result[i] = load_file(kernel_paths[i]);

    return result;
}

void free_kernel_sources(char** sources)
{
    for (size_t i = 0; i < kernel_path_count; ++i)
        free(sources[i]);

    free(sources);
}

cl_program* build_programs_with_sources(cl_uint count,
                                        cl_context* contexts,
                                        const char** sources)
{
    cl_int CL_err = CL_SUCCESS;

    size_t* lengths = (size_t*)malloc(kernel_path_count * sizeof(size_t));
    for(size_t i = 0 ; i < kernel_path_count; ++i)
#ifdef _WIN32
      lengths[i] = strnlen_s(sources[i], UINT_MAX);
#else
      lengths[i] = strnlen(sources[i], UINT_MAX);
#endif

    cl_program* result = (cl_program*)malloc(count * sizeof(cl_program));
    for (cl_uint i = 0; i < count; ++i)
    {
        cl_device_id device;
        CL_err = clGetContextInfo(contexts[i], CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL); checkErr(CL_err, "clGetContextInfo(CL_CONTEXT_DEVICES)");

        cl_uint uint_kernel_path_count = (cl_uint)kernel_path_count;
        result[i] = clCreateProgramWithSource(contexts[i], uint_kernel_path_count, sources, lengths, &CL_err);
        checkErr(CL_err, "clCreateProgramWithSource()");

        char build_params[2048];
        strcpy(build_params, " -I");
        strcat(build_params, kernel_inc_path);
        strcat(build_params, " -cl-std=CL1.2");
        strcat(build_params, " -Werror"); // Warnings will be treated like errors, this is useful for debug

        CL_err = clBuildProgram(result[i], 1, &device, build_params, NULL, NULL);

        if (CL_err != CL_SUCCESS)
        {
            cl_int build_err = CL_err;
            size_t len = 0;
            char* buffer;

            CL_err = clGetProgramBuildInfo(result[i], device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            checkErr(CL_err, "clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG)");

            buffer = (char*)calloc(len, sizeof(char));

            clGetProgramBuildInfo(result[i], device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);

            fprintf(stderr, "%s\n", buffer);

            free(buffer);

            checkErr(build_err, "clBuildProgram");
        }
    }

    free(lengths);

    return result;
}

cl_kernel** create_kernels(cl_uint count,
                           cl_program* programs)
{
    cl_kernel** result = (cl_kernel**)malloc(count * sizeof(cl_kernel*));

    for (cl_uint i = 0; i < count; ++i)
    {
        result[i] = (cl_kernel*)malloc(kernel_count * sizeof(cl_kernel));
        
        for (cl_uint j = 0; j < kernel_count; ++j)
            result[i][j] = obtain_kernel(programs[i], j);
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
    char filename[1024];

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
  size_t status = 0;

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

      ifo[i].sig.xDat_d = (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));
      for (cl_uint j = 0; j < cl_handles->count; ++j)
      {
        ifo[i].sig.xDat_d[j] = clCreateBuffer(cl_handles->ctxs[j],
                                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                              sett->N * sizeof(xDat_real),
                                              tmp,
                                              &CL_err);
        checkErr(CL_err, "clCreateBuffer(ifo[i].sig.xDat_d)");
      }
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
    char filename[1024];

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

    ifo[i].sig.DetSSB_d = (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));
    for (cl_uint j = 0; j < cl_handles->count; ++j)
    {
        ifo[i].sig.DetSSB_d[j] = clCreateBuffer(cl_handles->ctxs[j],
                                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                sett->N * sizeof(DetSSB_real3),
                                                ifo[i].sig.DetSSB,
                                                &CL_err);
        checkErr(CL_err, "clCreateBuffer(ifo[i].sig.DetSSB_d)");
    }

    // sincos 
    ifo[i].sig.sphir = sin(ifo[i].sig.phir);
    ifo[i].sig.cphir = cos(ifo[i].sig.phir);
    ifo[i].sig.sepsm = sin(ifo[i].sig.epsm);
    ifo[i].sig.cepsm = cos(ifo[i].sig.epsm);

    sett->sepsm = ifo[i].sig.sepsm;
    sett->cepsm = ifo[i].sig.cepsm;

	ifo[i].sig.xDatma_d = (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));
	ifo[i].sig.xDatmb_d = (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));
	ifo[i].sig.aa_d =     (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));
	ifo[i].sig.bb_d =     (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));
	ifo[i].sig.shft_d =   (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));
	ifo[i].sig.shftf_d =  (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));

	for (cl_uint j = 0; j < cl_handles->count; ++j)
	{
      ifo[i].sig.xDatma_d[j] = clCreateBuffer(cl_handles->ctxs[j],
                                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                              sett->N * sizeof(xDatm_complex),
                                              NULL,
                                              &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.xDatma_d)");
	  
      ifo[i].sig.xDatmb_d[j] = clCreateBuffer(cl_handles->ctxs[j],
                                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                              sett->N * sizeof(xDatm_complex),
                                              NULL,
                                              &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.xDatmb_d)");
	  
      ifo[i].sig.aa_d[j] = clCreateBuffer(cl_handles->ctxs[j],
                                          CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                          sett->N * sizeof(ampl_mod_real),
                                          NULL,
                                          &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.aa_d)");
	  
      ifo[i].sig.bb_d[j] = clCreateBuffer(cl_handles->ctxs[j],
                                          CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                          sett->N * sizeof(ampl_mod_real),
                                          NULL,
                                          &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.bb_d)");
	  
      ifo[i].sig.shft_d[j] = clCreateBuffer(cl_handles->ctxs[j],
                                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                            sett->N * sizeof(shift_real),
                                            NULL,
                                            &CL_err);
      checkErr(CL_err, "clCreateBuffer(ifo[i].sig.shft_d)");
	  
      ifo[i].sig.shftf_d[j] = clCreateBuffer(cl_handles->ctxs[j],
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

  fft_arr->xa_d = (cl_mem**)malloc(cl_handles->count * sizeof(cl_mem*));
  fft_arr->xb_d = (cl_mem**)malloc(cl_handles->count * sizeof(cl_mem*));

  for (cl_uint id = 0; id < cl_handles->count; ++id)
  {
    fft_arr->xa_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem));
    fft_arr->xb_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem));

	for (int n = 0; n < sett->nifo; ++n)
    {
      cl_int CL_err = CL_SUCCESS;
      fft_arr->xa_d[id][n] =
        clCreateBuffer(cl_handles->ctxs[id],
                       CL_MEM_READ_WRITE,
                       fft_arr->arr_len * sizeof(fft_complex),
                       NULL,
                       &CL_err);
      checkErr(CL_err, "clCreateBuffer(fft_arr->xa_d)");
    
      fft_arr->xb_d[id][n] =
        clCreateBuffer(cl_handles->ctxs[id],
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
  
  aux_arr->ifo_amod_d = (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));
  aux_arr->tshift_d = (cl_mem**)malloc(cl_handles->count * sizeof(cl_mem*));
  aux_arr->aadots_d = (cl_mem**)malloc(cl_handles->count * sizeof(cl_mem*));
  aux_arr->bbdots_d = (cl_mem**)malloc(cl_handles->count * sizeof(cl_mem*));
  aux_arr->maa_d = (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));
  aux_arr->mbb_d = (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));

  aux_arr->F_d = (cl_mem*)malloc(cl_handles->count * sizeof(cl_mem));

  for (cl_uint id = 0; id < cl_handles->count; ++id)
  {
    aux_arr->tshift_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem));
    aux_arr->aadots_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem));
    aux_arr->bbdots_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem));

    aux_arr->ifo_amod_d[id] =
      clCreateBuffer(cl_handles->ctxs[id],
                     CL_MEM_READ_ONLY,
                     sett->nifo * sizeof(Ampl_mod_coeff),
                     NULL,
                     &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->ifo_amod_d)");

    for (int n = 0; n < sett->nifo; ++n)
    {
      aux_arr->tshift_d[id][n] =
        clCreateBuffer(cl_handles->ctxs[id],
                       CL_MEM_READ_WRITE,
                       sett->N * sizeof(shift_real),
                       NULL,
                       &CL_err);
      checkErr(CL_err, "clCreateBuffer(aux_arr->tshift_d)");

      aux_arr->aadots_d[id][n] =
        clCreateBuffer(cl_handles->ctxs[id],
                       CL_MEM_READ_ONLY,
                       sizeof(ampl_mod_real),
                       NULL,
                       &CL_err);
      checkErr(CL_err, "clCreateBuffer(aux_arr->aadots_d)");

      aux_arr->bbdots_d[id][n] =
        clCreateBuffer(cl_handles->ctxs[id],
                       CL_MEM_READ_ONLY,
                       sizeof(ampl_mod_real),
                       NULL,
                       &CL_err);
      checkErr(CL_err, "clCreateBuffer(aux_arr->bbdots_d)");
    }

    aux_arr->maa_d[id] =
      clCreateBuffer(cl_handles->ctxs[id],
                     CL_MEM_READ_ONLY,
                     sizeof(ampl_mod_real),
                     NULL,
                     &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->maa_d)");

    aux_arr->mbb_d[id] =
      clCreateBuffer(cl_handles->ctxs[id],
                     CL_MEM_READ_ONLY,
                     sizeof(ampl_mod_real),
                     NULL,
                     &CL_err);
    checkErr(CL_err, "clCreateBuffer(aux_arr->mbb_d)");

    aux_arr->F_d[id] =
      clCreateBuffer(cl_handles->ctxs[id],
                     CL_MEM_READ_WRITE,
                     2 * sett->nfft * sizeof(fstat_real),
                     NULL,
                     &CL_err);
    checkErr(CL_err, "clCreateBuffer(F_d)");


    Ampl_mod_coeff* tmp =
      (Ampl_mod_coeff*)clEnqueueMapBuffer(cl_handles->write_queues[id][0],
                                          aux_arr->ifo_amod_d[id],
                                          CL_TRUE,
                                          CL_MAP_WRITE_INVALIDATE_REGION,
                                          0,
                                          sett->nifo * sizeof(Ampl_mod_coeff),
                                          0,
                                          NULL,
                                          NULL,
                                          &CL_err);

    for (int i = 0; i < sett->nifo; ++i) tmp[i] = ifo[i].amod;

    cl_event unmap_event;
    clEnqueueUnmapMemObject(cl_handles->write_queues[id][0], aux_arr->ifo_amod_d[id], tmp, 0, NULL, &unmap_event);

    clWaitForEvents(1, &unmap_event);

    // OpenCL cleanup
    clReleaseEvent(unmap_event);
  }
}

void add_signal(Search_settings *sett,
                Command_line_opts *opts,
                Aux_arrays *aux_arr,
                Search_range *s_range)
{
  int i, j, n, gsize, reffr; 
  double snr=0, sum = 0., h0=0, cof, d1; 
  double sigma_noise = 1.0;
  double be[2];
  double sinaadd, cosaadd, sindadd, cosdadd, phaseadd, shiftadd; 
  double nSource[3], sgnlo[8], sgnlol[4];
 
  char amporsnr[4];
 
  FILE *data;
  
  // Signal parameters are read
  if ((data=fopen (opts->addsig, "rb")) != NULL) {
	
    // Fscanning for the GW amplitude h0 or signal-to-noise,  
    // the grid size and the reference frame 
    // (for which the signal freq. is not spun-down/up)

    fscanf (data, "%s", amporsnr);    

    if(!strcmp(amporsnr, "amp")) { 
      fscanf (data, "%le %d %d", &h0, &gsize, &reffr); 
      printf("add_signal(): GW amplitude h0 is %le\n", h0); 
    } else if(!strcmp(amporsnr, "snr")) { 
      fscanf (data, "%le %d %d", &snr, &gsize, &reffr); 
      printf("add_signal(): GW (network) signal-to-noise ratio is %le\n", snr); 
    } else { 
      printf("Problem with the signal file. Exiting...\n"); 
      exit(0); 
    } 

    // Fscanning signal parameters: f, fdot, delta, alpha (sgnlo[0], ..., sgnlo[3])
    // four amplitudes sgnlo[4], ..., sgnlo[7] 
    // (see sigen.c and Phys. Rev. D 82, 022005 2010, Eqs. 2.13a-d) 

    for(i=0; i<8; i++)
      fscanf(data, "%le",i+sgnlo); 
    
    fclose (data);
                 
  } else {
    perror (opts->addsig);
  }
  
  // Search-specific parametrization of freq. 
  // for the software injections
  // sgnlo[0]: frequency, sgnlo[1]: frequency. derivative  
 
  sgnlo[0] += -2.*sgnlo[1]*(sett->N)*(reffr - opts->ident); 
 
  // Check if the signal is in band 
  if(sgnlo[0]<0) exit(171);          // &laquo;  
  else if (sgnlo[0]>M_PI) exit(187); // &raquo;

  cof = sett->oms + sgnlo[0]; 
  
  for(i=0; i<2; i++) sgnlol[i] = sgnlo[i]; 
  
  // Calculate the hemisphere and be vector 
  s_range->pmr[0] = ast2lin(sgnlo[3], sgnlo[2], C_EPSMA, be);

  sgnlol[2] = be[0]*cof;  
  sgnlol[3] = be[1]*cof; 

 		 	
  // solving a linear system in order to translate 
  // sky position, frequency and spindown (sgnlo parameters) 
  // into the position in the grid

  double *MM ; 
  MM = (double *) calloc (16, sizeof (double));
  for(i=0; i<16; i++) MM[i] = sett->M[i] ;
  
  gsl_vector *x = gsl_vector_alloc (4);     
  int s;
  
  gsl_matrix_view m = gsl_matrix_view_array (MM, 4, 4);
  gsl_matrix_transpose (&m.matrix) ; 
  gsl_vector_view b = gsl_vector_view_array (sgnlol, 4);
  gsl_permutation *p = gsl_permutation_alloc (4);
  
  gsl_linalg_LU_decomp (&m.matrix, p, &s);
  gsl_linalg_LU_solve (&m.matrix, p, &b.vector, x);
  
  s_range->spndr[0] = round(gsl_vector_get(x,1)); 
  s_range->nr[0] 	= round(gsl_vector_get(x,2));
  s_range->mr[0] 	= round(gsl_vector_get(x,3));
  
  gsl_permutation_free (p);
  gsl_vector_free (x);
  free (MM);
  
  // Define the grid range in which the signal will be looked for
  s_range->spndr[1] = s_range->spndr[0] + gsize; 
  s_range->spndr[0] -= gsize;
  s_range->nr[1] = s_range->nr[0] + gsize; 
  s_range->nr[0] -= gsize;
  s_range->mr[1] = s_range->mr[0] + gsize; 
  s_range->mr[0] -= gsize;
  s_range->pmr[1] = s_range->pmr[0]; 
  
  printf("add_signal(): following grid range is used (spndr, nr, mr, pmr pairs)\n");
  printf("%d %d %d %d %d %d %d %d\n", \
   s_range->spndr[0], s_range->spndr[1], s_range->nr[0], s_range->nr[1],
   s_range->mr[0], s_range->mr[1], s_range->pmr[0], s_range->pmr[1]);

  // sgnlo[2]: declination, sgnlo[3]: right ascension 
  sindadd = sin(sgnlo[2]); 
  cosdadd = cos(sgnlo[2]); 
  sinaadd = sin(sgnlo[3]);  
  cosaadd = cos(sgnlo[3]); 
	
  // To keep coherent phase between time segments  
  double phaseshift = sgnlo[0]*sett->N*(reffr - opts->ident)   
    + sgnlo[1]*pow(sett->N*(reffr - opts->ident), 2); 


  // Allocate arrays for added signal, for each detector 
  double **signadd = (double**)malloc((sett->nifo)*sizeof(double *));
  for(n=0; n<sett->nifo; n++)
    signadd[n] = (double*)malloc((sett->N)*sizeof(double));

  // Loop for each detector - sum calculations
  for(n=0; n<sett->nifo; n++) {
    
    modvir(sinaadd, cosaadd, sindadd, cosdadd,
	   sett->N, &ifo[n], aux_arr);

    nSource[0] = cosaadd*cosdadd;
    nSource[1] = sinaadd*cosdadd;
    nSource[2] = sindadd;
					
    for (i=0; i<sett->N; i++) {
      
      shiftadd = 0.; 					 
      for (j=0; j<3; j++)
      	shiftadd += nSource[j]*ifo[n].sig.DetSSB[i*3+j];		 
      
      // Phase 
      phaseadd = sgnlo[0]*i + sgnlo[1]*aux_arr->t2[i] 
        + (cof + 2.*sgnlo[1]*i)*shiftadd
        - phaseshift; 

      // The whole signal with 4 amplitudes and modulations 
      signadd[n][i] = sgnlo[4]*(ifo[n].sig.aa[i])*cos(phaseadd) 
                    + sgnlo[6]*(ifo[n].sig.aa[i])*sin(phaseadd) 
                    + sgnlo[5]*(ifo[n].sig.bb[i])*cos(phaseadd) 
                    + sgnlo[7]*(ifo[n].sig.bb[i])*sin(phaseadd);

      // Sum over signals
      sum += pow(signadd[n][i], 2.);
    
    } // data loop
   
  } // detector loop


  // Signal amplitude h0 from the snr 
  // (currently only makes sense for Gaussian noise with fixed sigma)
  if(snr)
    h0 = (snr*sigma_noise)/(sqrt(sum));

  // Loop for each detector - adding signal to data (point by point)  								
  for(n=0; n<sett->nifo; n++) {
    for (i=0; i<sett->N; i++) {

      // Adding the signal to the data vector 
      if(ifo[n].sig.xDat[i]) { 
        ifo[n].sig.xDat[i] += h0*signadd[n][i];

      } 

    } // data loop

  } // detector loop

  // printf("snr=%le h0=%le\n", snr, h0);

  // Free auxiliary 2d array 
  for(n=0; n<sett->nifo; n++) 
    free(signadd[n]);
  free(signadd);

  set_search_range(sett, opts, s_range); 
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

  blas_handles->aaScratch_d = (cl_mem**)malloc(cl_handles->count * sizeof(cl_mem));
  blas_handles->bbScratch_d = (cl_mem**)malloc(cl_handles->count * sizeof(cl_mem));

  for (cl_uint id = 0; id < cl_handles->count; ++id)
  {
    blas_handles->aaScratch_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem*));
    blas_handles->bbScratch_d[id] = (cl_mem*)malloc(sett->nifo * sizeof(cl_mem*));
    
    for (int idet = 0; idet < sett->nifo; ++idet)
    {
      cl_int CL_err = CL_SUCCESS;
      blas_handles->aaScratch_d[id][idet] =
        clCreateBuffer(cl_handles->ctxs[id],
                       CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       sett->N * sizeof(ampl_mod_real),
                       NULL,
                       &CL_err);
      checkErr(CL_err, "clCreateBuffer(blas_handles->aaScratch_d)");

      blas_handles->bbScratch_d[id][idet] =
        clCreateBuffer(cl_handles->ctxs[id],
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

  plans->plan =   (clfftPlanHandle*)malloc(cl_handles->count * sizeof(clfftPlanHandle));
  plans->pl_int = (clfftPlanHandle*)malloc(cl_handles->count * sizeof(clfftPlanHandle));
  plans->pl_inv = (clfftPlanHandle*)malloc(cl_handles->count * sizeof(clfftPlanHandle));

   clfftSetupData fftSetup;
   clfftStatus CLFFT_status = clfftSetup(&fftSetup);
   checkErrFFT(CLFFT_status, "clffftSetup");

  for (cl_uint id = 0; id < cl_handles->count; ++id)
  {
    // Phasemod FFT
    CLFFT_status = clfftCreateDefaultPlan(&plans->plan[id], cl_handles->ctxs[id], CLFFT_1D, &nfftf_size);
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
    CLFFT_status = clfftCreateDefaultPlan(&plans->pl_int[id], cl_handles->ctxs[id], CLFFT_1D, &nfft_size);
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
    CLFFT_status = clfftCreateDefaultPlan(&plans->pl_inv[id], cl_handles->ctxs[id], CLFFT_1D, &Ninterp_size);
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
             OpenCL_handles* cl_handles,
             BLAS_handles* blas_handles,
             FFT_plans *plans,
             FFT_arrays *fft_arr,
             Aux_arrays *aux)
{
  cleanup_fft(cl_handles, plans);

  cleanup_blas(sett, cl_handles, blas_handles);

  cleanup_arrays(ifo, sett, cl_handles, aux, fft_arr);

  free(opts->wd);

  cleanup_opencl(cl_handles);

} // end of cleanup & memory free

void cleanup_arrays(Detector_settings* ifo,
                    Search_settings* sett,
                    OpenCL_handles* cl_handles,
                    Aux_arrays* aux_arr,
                    FFT_arrays* fft_arr)
{
  cleanup_fft_arrays(fft_arr, sett->nifo, cl_handles->count);

  cleanup_aux_arrays(sett, cl_handles, aux_arr);

  cleanup_ifo_arrays(sett, cl_handles, ifo);
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

void cleanup_aux_arrays(Search_settings* sett,
                        OpenCL_handles* cl_handles,
                        Aux_arrays* aux_arr)
{
  cl_int CL_err = CL_SUCCESS;

  for (cl_uint id = 0; id < cl_handles->count; ++id)
  {
    for (int n = 0; n < sett->nifo; ++n)
    {
      CL_err = clReleaseMemObject(aux_arr->tshift_d[id][n]); checkErr(CL_err, "clReleaseMemObject(aux_arr->tshift_d[id][n])");
      CL_err = clReleaseMemObject(aux_arr->aadots_d[id][n]); checkErr(CL_err, "clReleaseMemObject(aux_arr->aadots_d[id][n])");
      CL_err = clReleaseMemObject(aux_arr->bbdots_d[id][n]); checkErr(CL_err, "clReleaseMemObject(aux_arr->bbdots_d[id][n])");
    }

    free(aux_arr->tshift_d[id]);
    free(aux_arr->aadots_d[id]);
    free(aux_arr->bbdots_d[id]);

    CL_err = clReleaseMemObject(aux_arr->maa_d[id]); checkErr(CL_err, "clReleaseMemObject(aux_arr->maa_d[id]");
    CL_err = clReleaseMemObject(aux_arr->mbb_d[id]); checkErr(CL_err, "clReleaseMemObject(aux_arr->mbb_d[id]");
    CL_err = clReleaseMemObject(aux_arr->F_d[id]);   checkErr(CL_err, "clReleaseMemObject(aux_arr->F_d[id]");
    CL_err = clReleaseMemObject(aux_arr->ifo_amod_d[id]); checkErr(CL_err, "clReleaseMemObject(aux_arr->ifo_amod_d)");
  }

  free(aux_arr->tshift_d);
  free(aux_arr->aadots_d);
  free(aux_arr->bbdots_d);
  free(aux_arr->maa_d);
  free(aux_arr->mbb_d);

  free(aux_arr->F_d);
  free(aux_arr->ifo_amod_d);
}

void cleanup_ifo_arrays(Search_settings* sett,
                        OpenCL_handles* cl_handles,
                        Detector_settings* ifo)
{
  for (int i = 0; i < sett->nifo; i++)
  {
    free(ifo[i].sig.xDat);

    cl_int CL_err = CL_SUCCESS;

    free(ifo[i].sig.DetSSB);

    for (cl_uint j = 0; j < cl_handles->count; ++j)
    {
      CL_err = clReleaseMemObject(ifo[i].sig.xDatma_d[j]); checkErr(CL_err, "clReleaseMemObject(ifo[i].sig.xDatma_d[j])");
      CL_err = clReleaseMemObject(ifo[i].sig.xDatmb_d[j]); checkErr(CL_err, "clReleaseMemObject(ifo[i].sig.xDatmb_d[j])");
      CL_err = clReleaseMemObject(ifo[i].sig.aa_d[j]);     checkErr(CL_err, "clReleaseMemObject(ifo[i].sig.aa_d[j])");
      CL_err = clReleaseMemObject(ifo[i].sig.bb_d[j]);     checkErr(CL_err, "clReleaseMemObject(ifo[i].sig.bb_d[j])");
      CL_err = clReleaseMemObject(ifo[i].sig.shft_d[j]);   checkErr(CL_err, "clReleaseMemObject(ifo[i].sig.shft_d[j])");
      CL_err = clReleaseMemObject(ifo[i].sig.shftf_d[j]);  checkErr(CL_err, "clReleaseMemObject(ifo[i].sig.shftf_d[j])");
      CL_err = clReleaseMemObject(ifo[i].sig.xDat_d[j]);   checkErr(CL_err, "clReleaseMemObject(ifo[i].sig.xDat_d[j])");
      CL_err = clReleaseMemObject(ifo[i].sig.DetSSB_d[j]); checkErr(CL_err, "clReleaseMemObject(ifo[i].sig.DetSSB_d[j])");
    }

    free(ifo[i].sig.xDatma_d);
    free(ifo[i].sig.xDatmb_d);
    free(ifo[i].sig.aa_d);
    free(ifo[i].sig.bb_d);
    free(ifo[i].sig.shft_d);
    free(ifo[i].sig.shftf_d);
    free(ifo[i].sig.xDat_d);
    free(ifo[i].sig.DetSSB_d);
  }
}

void cleanup_opencl(OpenCL_handles* cl_handles)
{
  cleanup_kernels(cl_handles->kernels, cl_handles->count);

  cleanup_program(cl_handles->progs, cl_handles->count);

  cleanup_command_queue_set(cl_handles->read_queues, cl_handles->count);
  cleanup_command_queue_set(cl_handles->exec_queues, cl_handles->count);
  cleanup_command_queue_set(cl_handles->write_queues, cl_handles->count);

  cleanup_contexts(cl_handles->ctxs, cl_handles->count);

  cleanup_devices(cl_handles->devs, cl_handles->count);
}

void cleanup_devices(cl_device_id* devices,
                     cl_uint count)
{
  for (cl_uint i = 0; i < count; ++i)
  {
    cl_int CL_err = clReleaseDevice(devices[i]);checkErr(CL_err, "clReleaseDevice(devices[i])");
  }

  free(devices);
}

void cleanup_contexts(cl_context* ctxs,
                      cl_uint count)
{
  for (cl_uint i = 0; i < count; ++i)
  {
    cl_int CL_err = clReleaseContext(ctxs[i]); checkErr(CL_err, "clReleaseContext(ctxs[i]);");
  }
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

void cleanup_program(cl_program* progs,
                     cl_uint count)
{
  for (cl_uint i = 0; i < count; ++i)
  {
    cl_int CL_err = clReleaseProgram(progs[i]); checkErr(CL_err, "clReleaseProgram(progs[i]);");
  }
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

  for (cl_uint id = 0; id < cl_handles->count; ++id)
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

void cleanup_fft(OpenCL_handles* cl_handles,
                 FFT_plans* plans)
{
  clfftStatus status = CLFFT_SUCCESS;
  for (cl_uint id = 0; id < cl_handles->count; ++id)
  {
    status = clfftDestroyPlan(&plans->plan[id]);   checkErrFFT(status, "clfftDestroyPlan(plans->plan)");
    status = clfftDestroyPlan(&plans->pl_int[id]); checkErrFFT(status, "clfftDestroyPlan(plans->pl_int)");
    status = clfftDestroyPlan(&plans->pl_inv[id]); checkErrFFT(status, "clfftDestroyPlan(plans->pl_inv)");
  }
  clfftTeardown();
}
