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
if (BUILD_TESTS)

  add_test(NAME ${Target_NAME}
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${Target_NAME}
                   -data ${PROJECT_SOURCE_DIR}/../../testdata/test-data-network-injection
                   -ident 205
                   -label J0322+0441
                   -band 000
                   -fpo 1391.3
                   -r ${PROJECT_SOURCE_DIR}/../../testdata/range-J0322+0441.txt)
		
endif (BUILD_TESTS)
