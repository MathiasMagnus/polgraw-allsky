#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.1.0)

# The name of our project
set (Target_NAME gwsearch-ocl)

# Adding source code files according to configuration
set (HDRS inc/auxi.h
          inc/floats.h
          inc/init.h
          inc/jobcore.h
          inc/settings.h
          inc/spline_z.h
          inc/struct.h
          inc/timer.h
          inc/CL/util.h)

set (SRCS src/main.c
          src/auxi.c
		  src/floats.c
          src/settings.c
          #src/timer.c
          src/init.c
          src/jobcore.c
          src/spline_z.c
          src/CL/util.c)

set (KRNS krn/floats.h.cl
          krn/kernels.h.cl
          krn/kernels.cl)

# Generate the configuration file for application to locate kernel files
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/inc/locations.in.h
                ${CMAKE_CURRENT_BINARY_DIR}/inc/locations.h)

list (APPEND HDRS ${CMAKE_CURRENT_BINARY_DIR}/inc/locations.h)

# Variable to hold ALL files to build and be visible in IDE
set (BUILD ${HDRS} ${SRCS} ${KRNS})

# Create filters for IDEs
source_group ("Headers" FILES ${HDRS})
source_group ("Sources" FILES ${SRCS})
source_group ("Kernels" FILES ${KRNS})

# Specify executable sources
add_executable (${Target_NAME} ${BUILD})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${Target_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc
                                                  ${CMAKE_CURRENT_BINARY_DIR}/inc
                                                  ${OpenCL_INCLUDE_DIRS}
                                                  ${CLFFT_INCLUDE_DIRS}
                                                  ${CLBLAS_INCLUDE_DIRS})

# Link dependant libraries
target_link_libraries (${Target_NAME} PRIVATE ${OpenCL_LIBRARIES}
                                              clFFT
                                              clBLAS)

# Create project groups for IDEs
set_target_properties (${Target_NAME} PROPERTIES FOLDER "GPU")

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
                 rescaled_ifo_sig_xDatma)

  set (CL_Data_Files)
  set (C_Data_Files)

  foreach (Data_Set IN LISTS Data_Sets) # Loops over stages
    foreach (ifo RANGE 0 1) # Loops over detectors
	  list(APPEND CL_Data_Files "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cl_${Data_Set}.${ifo}.dat")
	  list(APPEND C_Data_Files  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/c_${Data_Set}.${ifo}.dat")
    endforeach (ifo)
  endforeach (Data_Set)

  #message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

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

  list (LENGTH CL_Data_Files Data_Count)
  foreach (Index RANGE 0 ${Count})

    list (GET CL_Data_Files ${Index} CL_Data_File)
	list (GET C_Data_Files  ${Index} C_Data_File)

    get_filename_component (CL_File_Name ${CL_Data_File} NAME)
	string (REPLACE "cl_" "" Test_Name ${CL_File_Name})

	add_test (NAME ${Test_Name}
	          COMMAND validation ${Data_File} ${Ref_Data_File})

  endforeach (Index)
		
endif (BUILD_TESTING)