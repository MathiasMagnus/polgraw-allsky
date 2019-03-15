#ifndef __INIT_H__
#define __INIT_H__

// Polgraw includes
#include <struct.h>


/// <summary>Command line options handling: search</summary>
///
void handle_opts(Search_settings* sett,
                 OpenCL_settings* cl_sett,
		         Command_line_opts* opts,
		         int argc,  
		         char* argv[]);

/// <summary>Initialize OpenCL devices based on user preference.</summary>
/// <remarks>Currently, only a single platform can be selected. At some point,
///          multi-platform may be implemented. This is mighty useful for
///          Intel CPU+IGP & Nvidia/AMD dGPU setups, primarily notebooks.
///          In non-mobile form factors dGPU significantly outweights CPU+IGP.</remarks>
///
void init_opencl(OpenCL_handles* cl_handles,
                 OpenCL_settings* cl_sett);

/// <summary>Tries selecting the platform with the specified index.</summary>
///
cl_platform_id select_platform(cl_uint plat_id);

/// <summary>Selects all devices of the specified type.</summary>
///
cl_device_id* select_devices(cl_platform_id platform,
                             cl_device_type dev_type,
                             cl_uint* count);

/// <summary>Create a context that holds all specified devices.</summary>
///
cl_context create_standard_context(cl_device_id* devices,
                                   cl_uint count);

/// <summary>Create a set of command queues to all the devices in the context.</summary>
///
cl_command_queue** create_command_queue_set(cl_context context);

/// <summary>Load kernel file from disk.</summary>
///
char* load_file(const char* filename);

/// <summary>Load all kernel files from disk.</summary>
///
char** load_kernel_sources();

/// <summary>Free memory of source files.</summary>
///
void free_kernel_sources(char** sources);

/// <summary>Build program file</summary>
///
cl_program build_program_with_sources(cl_context context,
                                      const char** sources);

/// <summary>Create a kernel for all entry points in the program for each device.</summary>
///
cl_kernel** create_kernels(cl_program program,
	                       cl_uint count);

/// <summary>Obtain the name of the kernel of a given index.</summary>
///
const char* obtain_kernel_name(cl_uint i);

/// <summary>Obtain kernel with the specified index.</summary>
///
cl_kernel obtain_kernel(cl_program program, cl_uint i);

/// <summary>Generate grid from the M matrix.</summary>
/// <remarks>Processes the file 'grid.bin'</remarks>
///
void read_grid(Search_settings *sett,
               Command_line_opts *opts);

/// <summary>Initialize auxiliary and F-statistics arrays.</summary>
///
void init_arrays(Detector_settings* ifo,
                 Search_settings* sett,
                 OpenCL_handles* cl_handles,
		         Command_line_opts* opts, 
		         Aux_arrays* aux_arr,
                 FFT_arrays* fft_arr);

/// <summary>Initialize interferometer arrays.</summary>
///
void init_ifo_arrays(Search_settings* sett,
	                 OpenCL_handles* cl_handles,
                     Command_line_opts* opts,
	                 Detector_settings* ifo);

/// <summary>Initialize auxiliary arrays.</summary>
///
void init_aux_arrays(Search_settings* sett,
                     Detector_settings* ifo,
                     OpenCL_handles* cl_handles,
                     Aux_arrays* aux_arr);

/// <summary>Initialize FFT arrays.</summary>
///
void init_fft_arrays(Search_settings* sett,
                     OpenCL_handles* cl_handles,
                     FFT_arrays* fft_arr);

void add_signal(
		Search_settings *sett,
		Command_line_opts *opts,
		Aux_arrays *aux_arr,
		Search_range *s_range);

/// <summary>Set search ranges based on user preference.</summary>
///
void set_search_range(Search_settings *sett, 
                      Command_line_opts *opts, 
                      Search_range *s_range);

/// <summary>Sets up BLAS internal states.</summary>
///
void init_blas(Search_settings* sett,
               OpenCL_handles* cl_handles,
               BLAS_handles* blas_handles);

/// <summary>Sets up FFT plans.</summary>
///
void init_fft(Search_settings* sett,
              OpenCL_handles* cl_handles,
	          FFT_plans* plans);

/// <summary>Initialize the OpenMP runtime</summary>
///
void init_openmp(cl_uint count);

void read_checkpoints(Command_line_opts *opts, 
		              Search_range *s_range,
		              int *Fnum);

/// <summary>Release and free all resources in reverse order for termination.</summary>
///
void cleanup(Detector_settings* ifo,
             Search_settings *sett,
	         Command_line_opts *opts,
             OpenCL_handles* cl_handles,
             BLAS_handles* blas_handles,
	         FFT_plans *plans,
	         FFT_arrays *fft_arr,
	         Aux_arrays *aux);

/// <summary>Cleanup auxiliary and F-statistics arrays.</summary>
///
void cleanup_arrays(Detector_settings* ifo,
                    Search_settings* sett,
                    OpenCL_handles* cl_handles,
                    Aux_arrays* aux_arr,
                    FFT_arrays* fft_arr);

/// <summary>Release and free FFT arrays.</summary>
///
void cleanup_fft_arrays(FFT_arrays* fft_arr,
                        int nifo,
	                    cl_uint count);

/// <summary>Release interferometer arrays.</summary>
///
void cleanup_ifo_arrays(Search_settings* sett,
                        OpenCL_handles* cl_handles,
                        Detector_settings* ifo);

/// <summary>Release auxiliary arrays.</summary>
///
void cleanup_aux_arrays(Search_settings* sett,
                        OpenCL_handles* cl_handles,
                        Aux_arrays* aux_arr);

/// <summary>Frees OpenCL resources.</summary>
///
void cleanup_opencl(OpenCL_handles* cl_handles);

/// <summary>Release and free OpenCL devices.</summary>
///
void cleanup_devices(cl_device_id* devices,
	                 cl_uint count);

/// <summary>Release OpenCL context.</summary>
///
void cleanup_context(cl_context ctx);

/// <summary>Releases and frees a set of command queues.</summary>
///
void cleanup_command_queue_set(cl_command_queue** queues,
	                           size_t count);

/// <summary>Release program.</summary>
///
void cleanup_program(cl_program prog);

/// <summary>Releases and frees all kernels.</summary>
///
void cleanup_kernels(cl_kernel** kernels,
	                cl_uint count);

/// <summary>Cleanup BLAS internal states.</summary>
///
void cleanup_blas(Search_settings* sett,
	              OpenCL_handles* cl_handles,
	              BLAS_handles* blas_handles);

/// <summary>Cleanup FFT plans.</summary>
///
void cleanup_fft(OpenCL_handles* cl_handles,
	             FFT_plans* plans);

// Coincidences specific functions 
void handle_opts_coinc(
		       Search_settings *sett,
		       Command_line_opts_coinc *opts,
		       int argc,  
		       char* argv[]);  

void manage_grid_matrix(
			Search_settings *sett,
			Command_line_opts_coinc *opts);

void convert_to_linear(
		       Search_settings *sett,
		       Command_line_opts_coinc *opts, 
		       Candidate_triggers *trig);

#endif // __INIT_H__
