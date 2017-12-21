#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.1.0)

# The name of our project
set (Target_NAME gwsearch-gpu)

# Adding source code files according to configuration
set (HDRS ${CMAKE_CURRENT_SOURCE_DIR}/auxi.h
          ${CMAKE_CURRENT_SOURCE_DIR}/floats.h
          ${CMAKE_CURRENT_SOURCE_DIR}/init.h
          ${CMAKE_CURRENT_SOURCE_DIR}/jobcore.h
          ${CMAKE_CURRENT_SOURCE_DIR}/settings.h
          ${CMAKE_CURRENT_SOURCE_DIR}/spline_z.h
          ${CMAKE_CURRENT_SOURCE_DIR}/struct.h
          ${CMAKE_CURRENT_SOURCE_DIR}/timer.h
          ${CMAKE_CURRENT_SOURCE_DIR}/CL/util.h)

set (SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.c
          ${CMAKE_CURRENT_SOURCE_DIR}/auxi.c
          ${CMAKE_CURRENT_SOURCE_DIR}/settings.c
          #${CMAKE_CURRENT_SOURCE_DIR}/timer.c
          ${CMAKE_CURRENT_SOURCE_DIR}/init.c
          ${CMAKE_CURRENT_SOURCE_DIR}/jobcore.c
          ${CMAKE_CURRENT_SOURCE_DIR}/spline_z.c
          ${CMAKE_CURRENT_SOURCE_DIR}/CL/util.c)

set (KRNS ${CMAKE_CURRENT_SOURCE_DIR}/floats.hcl
          ${CMAKE_CURRENT_SOURCE_DIR}/kernels.hcl
          ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl)

# Generate the configuration file for application to locate kernel files
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/locations.in.h
                ${CMAKE_CURRENT_BINARY_DIR}/locations.h)

list (APPEND HDRS ${CMAKE_CURRENT_BINARY_DIR}/locations.h)

# Variable to hold ALL files to build and be visible in IDE
set (BUILD ${HDRS} ${SRCS} ${KRNS})

# Create filters for IDEs
source_group ("Headers" FILES ${HDRS})
source_group ("Sources" FILES ${SRCS})
source_group ("Kernels" FILES ${KRNS})

# Specify executable sources
add_executable (${Target_NAME} ${BUILD})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${Target_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ${CMAKE_CURRENT_BINARY_DIR}
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
                   -data ${POLGRAW_SEARCH_NETWORK_ROOT}/../../testdata/test-data-network-injection
                   -ident 205
                   -label J0322+0441
                   -band 000
                   -fpo 1391.3
                   -r ${POLGRAW_SEARCH_NETWORK_ROOT}/../../testdata/range-J0322+0441.txt)
		
endif (BUILD_TESTS)
