#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.9)

# The name of our project
set (Target_NAME gwsearch-ocl)

# Adding source code files according to configuration
set (HDRS inc/auxi.h
          inc/precision.h
          inc/floats.h
		  inc/log.h
          inc/init.h
          inc/jobcore.h
		  inc/sky_positions.h
          inc/settings.h
          inc/spline_z.h
          inc/struct.h
          inc/timer.h
          inc/CL/util.h
		  inc/modvir.h
		  inc/tshift_pmod.h
		  inc/fft_interpolate.h
		  inc/spline_interpolate.h
		  inc/blas_dot.h
		  inc/calc_mxx.h
		  inc/phase_mod.h
		  inc/time_to_frequency.h
		  inc/fstat.h
		  inc/find_peaks.h)

set (SRCS src/main.c
          src/auxi.c
		  src/log.c
		  src/log_sp.c
		  src/log_dp.c
          src/settings.c
          src/init.c
          src/jobcore.c
		  src/sky_positions.c
          #src/spline_z.c
          src/CL/util.c
		  src/modvir.c
		  src/tshift_pmod.c
		  src/fft_interpolate.c
		  src/spline_interpolate.c
		  src/blas_dot.c
		  src/calc_mxx.c
		  src/phase_mod.c
		  src/time_to_frequency.c
		  src/fstat.c
		  src/find_peaks.c)

set (KRN_HDRS krn/floats.h.cl
              #krn/kernels.h.cl
		      krn/modvir.h.cl
			  krn/tshift_pmod.h.cl
			  krn/fft_interpolate.h.cl
			  krn/phase_mod.h.cl
			  krn/fstat.h.cl)

set (KRN_SRCS #krn/kernels.cl
		      krn/modvir.cl
			  krn/tshift_pmod.cl
			  krn/fft_interpolate.cl
			  krn/phase_mod.cl
			  krn/fstat.cl)

# Generate the configuration file for application to locate kernel files
set (KRN_LIST)
foreach (Kernel ${KRN_SRCS})
  string(APPEND KRN_LIST "\"${CMAKE_CURRENT_SOURCE_DIR}/${Kernel}\",")
endforeach ()
string (REGEX REPLACE ".$" "" KRN_LIST ${KRN_LIST}) # Remove trailing ,
list (LENGTH KRN_SRCS KRN_LIST_LENGTH)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/inc/locations.in.h
                ${CMAKE_CURRENT_BINARY_DIR}/inc/locations.h)

list (APPEND HDRS ${CMAKE_CURRENT_BINARY_DIR}/inc/locations.h)

# Variable to hold ALL files to build and be visible in IDE
set (BUILD ${HDRS} ${SRCS} ${KRN_HDRS} ${KRN_SRCS})

# Create filters for IDEs
source_group ("Host//Headers" FILES ${HDRS})
source_group ("Host//Sources" FILES ${SRCS})
source_group ("Device//Headers" FILES ${KRN_HDRS})
source_group ("Device//Sources" FILES ${KRN_SRC})

# Specify executable sources
add_executable (${Target_NAME} ${BUILD})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${Target_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc
                                                  ${CMAKE_CURRENT_BINARY_DIR}/inc
                                                  ${CLFFT_INCLUDE_DIRS}
                                                  ${CLBLAS_INCLUDE_DIRS})

# Link dependant libraries
target_link_libraries (${Target_NAME} PRIVATE OpenCL::OpenCL
                                              OpenMP::OpenMP_C
                                              clFFT
                                              clBLAS)

# Create project groups for IDEs
set_target_properties (${Target_NAME} PROPERTIES FOLDER "GPU")

# Add verbose output if opted in for
target_compile_definitions (${Target_NAME} PRIVATE $<$<BOOL:${PG_VERBOSE}>:VERBOSE>
                                                   $<$<BOOL:${BUILD_TESTING}>:TESTING>)

#
# Set compiler-specific variables
#
  # If compiling with Microsoft Visual C
  if (CMAKE_C_COMPILER_ID MATCHES "MSVC")

    # Link it to our executable
    target_link_libraries (${Target_NAME} PRIVATE getopt)

    target_include_directories (${Target_NAME} PRIVATE ${POSIX_INCLUDE_DIR})

    target_compile_definitions (${Target_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)

  endif (CMAKE_C_COMPILER_ID MATCHES "MSVC")

#########################
#                       #
#         CTest         #
#                       #
#########################

# Add unit test
if (BUILD_TESTING)

  set (Data_Sets aux_cosmodf
                 aux_sinmodf
                 ifo_sig_aa
                 ifo_sig_bb
                 ifo_sig_shft
                 ifo_sig_shftf
                 xa_time
                 xb_time
                 xa_fourier
                 xb_fourier
                 xa_fourier_resampled
                 xb_fourier_resampled
                 xa_time_resampled
                 xb_time_resampled
                 ifo_sig_xDatma
                 ifo_sig_xDatmb
                 rescaled_ifo_sig_xDatmb
                 rescaled_ifo_sig_xDatma
				 pre_fft_phasemod_ifo_sig_shft
				 pre_fft_phasemod_ifo_sig_xDatma
				 pre_fft_phasemod_ifo_sig_xDatmb
				 pre_fft_phasemod_xa
				 pre_fft_phasemod_xb
				 pre_fft_post_zero_xa
				 pre_fft_post_zero_xb
				 post_fft_phasemod_xa
				 post_fft_phasemod_xb
				 Fstat
				 Fstat_norm)

  set (CL_Data_Files)
  set (C_Data_Files)

  foreach (Data_Set IN LISTS Data_Sets) # Loops over stages
    foreach (ifo RANGE 0 1) # Loops over detectors
	  list(APPEND CL_Data_Files "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cl_${Data_Set}.${ifo}.dat")
	  list(APPEND C_Data_Files  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/c_${Data_Set}.${ifo}.dat")
    endforeach (ifo)
  endforeach (Data_Set)

  # Custom command-target pair to generate reference data sets
  add_custom_command (COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gwsearch-cpu
                      ARGS -data ${PROJECT_SOURCE_DIR}/../../testdata/test-gpu-cpu
					       -output .
                           -ident 001
                           -band 0666
						   -dt 2
						   --nocheckpoint
						   -range ${PROJECT_SOURCE_DIR}/../../testdata/test-gpu-cpu/range.dat
                           -usedet H1L1
                      WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                      OUTPUT ${C_Data_Files}
					  DEPENDS gwsearch-cpu)

  add_custom_target (validation-cpu
                     DEPENDS ${C_Data_Files}
					 COMMENT "Generating reference datasets to validate against.")

  # Custom command-target pair to generate OpenCL data sets
  add_custom_command (COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${Target_NAME}
                      ARGS -data ${PROJECT_SOURCE_DIR}/../../testdata/test-gpu-cpu
					       -output .
                           -ident 001
                           -band 0666
						   -dt 2
						   --nocheckpoint
						   -range ${PROJECT_SOURCE_DIR}/../../testdata/test-gpu-cpu/range.dat
                           -usedet H1L1
						   -device gpu
                      WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                      OUTPUT ${CL_Data_Files}
					  DEPENDS ${Target_NAME})

  add_custom_target (validation-cl
                     DEPENDS ${CL_Data_Files}
					 COMMENT "Generating OpenCL datasets to validate.")

  add_custom_target (validation-data
                     DEPENDS validation-cpu
					         validation-cl)

  add_executable (validation src/validation.cpp)

  set_target_properties (validation PROPERTIES CXX_STANDARD 14
                                               CXX_STANDARD_REQUIRED ON
                                               CXX_EXTENSIONS OFF)

  foreach (Test_Target IN ITEMS validation
                                validation-cpu
                                validation-cl
								validation-data)

    set_target_properties(${Test_Target} PROPERTIES FOLDER "Test")

  endforeach (Test_Target)

  # Calculate length of iteration for pair-wise traversal
  list (LENGTH CL_Data_Files Data_Count)
  math (EXPR Count "${Data_Count} - 1")

  foreach (Index RANGE 0 ${Count})

    list (GET CL_Data_Files ${Index} CL_Data_File)
	list (GET C_Data_Files  ${Index} C_Data_File)

    get_filename_component (CL_File_Name ${CL_Data_File} NAME)
	string (REPLACE "cl_" "" Test_Name ${CL_File_Name})

	add_test (NAME ${Test_Name}
	          COMMAND validation ${CL_Data_File} ${C_Data_File})

  endforeach (Index)
		
endif (BUILD_TESTING)