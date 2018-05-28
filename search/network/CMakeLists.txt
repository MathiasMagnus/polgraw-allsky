###############################################################################
#
# Top level CMakeLists.txt file for the polgraw-allsky repo's search/network
# Hacker: Mate Ferenc Nagy-Egri
# Last modified: 2016.10.04.
#
###############################################################################

#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of 'cmake' necessary to build this project
cmake_minimum_required (VERSION 3.1.0)
	
# The project name and version, release numbers
project (polgraw-search LANGUAGES C)
	
# Behavioural options for the project
option (BUILD_C "Build native C search suite" ON)
option (BUILD_OPENMP "Build OpenMP accelerated search suite" OFF)
option (BUILD_OPENCL "Build OpenCL accelerated search suite" ON)
option (BUILD_CUDA "Build CUDA accelerated search suite" OFF)
option (BUILD_TESTING "Build unit tests" OFF)
option (PG_VERBOSE "Verbose output of search suite" OFF)

#
# Set platform-independent variables
#

  # Append modules directory of shipped FindModule scripts
  list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

  # Find delpendencies
  if (${BUILD_C})
    find_package(GSL REQUIRED)
    find_package(FFTW REQUIRED)
  endif (${BUILD_C})

  if (${BUILD_OPENMP})
    find_package(OpenMP REQUIRED)
  endif (${BUILD_OPENMP})

  if (${BUILD_OPENCL})
    find_package(OpenCL REQUIRED)
    find_package(clFFT REQUIRED CONFIG)
    find_package(clBLAS REQUIRED CONFIG)
  endif (${BUILD_OPENCL})
  
  if (${USE_CUDA})
    find_package(CUDA REQUIRED)
  endif (${USE_CUDA})
  
  # Use solution folders in IDEs (logical grouping of targets)
  set_property (GLOBAL PROPERTY USE_FOLDERS ON)
  
  # Override predefined default output directories
  set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
	
#
# Set platform-specific variables
#
  # If compiling on Windows
  if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_subdirectory(posix)
  endif (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
		
  # If compiling on Linux-like
  if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  endif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")

		
#########################
#                       #
#         CTest         #
#                       #
#########################

# Enable testing
if (BUILD_TESTING)	
  include (CTest)
  enable_language (CXX)
endif (BUILD_TESTING)
	
#########################
#                       #
#         CMake         #
#                       #
#########################
	
#
# Process targets based on user-input
#
  if (${BUILD_C})
    add_subdirectory (src-cpu)
  endif (${BUILD_C})

  if (${BUILD_OPENMP})
    message (FATAL "OpenMP search suite CMake build automation not yet implemented.")
  endif (${BUILD_OPENMP})

  if (${BUILD_OPENCL})
    add_subdirectory (src-opencl)
  endif (${BUILD_OPENCL})
  
  if (${USE_CUDA})
  message (FATAL "CUDA search suite CMake build automation not yet implemented.")
  endif (${USE_CUDA})
