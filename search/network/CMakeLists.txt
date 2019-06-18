#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of 'cmake' necessary to build this project
cmake_minimum_required (VERSION 3.9.0)
	
# Behavioural options for the project
option (BUILD_C "Build native C search suite" ON)
option (BUILD_OPENMP "Build OpenMP accelerated search suite" OFF)
option (BUILD_OPENCL "Build OpenCL accelerated search suite" ON)
option (BUILD_SYCL "Build SYCL accelerated search suite" OFF)
option (BUILD_CUDA "Build CUDA accelerated search suite" OFF)
option (BUILD_TESTING "Build unit tests" OFF)

#
# Set platform-independent variables
#
  # Append modules directory of shipped FindModule scripts
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)
	
#
# Set platform-specific variables
#
  # If compiling on Windows
  if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_subdirectory(posix)
  endif (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	
#
# Process targets based on user-input
#
  if (${BUILD_C})
    add_subdirectory (src-cpu)
  endif (${BUILD_C})

  if (${BUILD_OPENMP})
    message (FATAL_ERROR "OpenMP search suite CMake build automation not yet implemented.")
  endif (${BUILD_OPENMP})

  if (${BUILD_OPENCL})
    add_subdirectory (src-opencl)
  endif (${BUILD_OPENCL})

  if (${BUILD_SYCL})
    add_subdirectory (src-sycl)
  endif (${BUILD_SYCL})
  
  if (${USE_CUDA})
  message (FATAL_ERROR "CUDA search suite CMake build automation not yet implemented.")
  endif (${USE_CUDA})
