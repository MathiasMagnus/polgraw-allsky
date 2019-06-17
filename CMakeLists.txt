###############################################################################
#
# Top level CMakeLists.txt file for the polgraw-allsky repo
# Hacker: Máté Ferenc Nagy-Egri
#
###############################################################################

#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of 'cmake' necessary to build this project
cmake_minimum_required (VERSION 3.7.0)
	
# The project name
project (polgraw-allsky)
	
# Behavioural options for the project
option (PG_GRIDGEN "Build the grid generating library and utility" ON)
option (PG_SEARCH "Build the search libraries and applications" ON)
option (PG_VERBOSE "Verbose output of search suite" OFF)
option (BUILD_TESTING "Build unit tests" OFF)

#
# Set platform-independent variables
#
  
  if (${BUILD_CUDA})
    message(FATAL_ERROR "Building of the CUDA search pipeline with CMake is not implemented yet")
  endif (${BUILD_CUDA})
  
  # Use solution folders in IDEs (logical grouping of targets)
  set_property (GLOBAL PROPERTY USE_FOLDERS ON)
  
  # Override predefined default output directories
  set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})

#
# Set compiler-specific variables
#
  # If compiling with Visual C/C++
  if (CMAKE_C_COMPILER_ID MATCHES "MSVC")
    # Remove the CMake & Visual Studio default /W3
    string(REGEX REPLACE "/W[0-9]" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
  endif (CMAKE_C_COMPILER_ID MATCHES "MSVC")

  if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # Remove the CMake & Visual Studio default /W3
    string(REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  endif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		
#########################
#                       #
#         CTest         #
#                       #
#########################

# Enable testing
if (BUILD_TESTING)	
  include (CTest)
  enable_testing()
endif (BUILD_TESTING)
	
#########################
#                       #
#         CMake         #
#                       #
#########################
	
#
# Process targets based on user-input
#
  if (${PG_GRIDGEN})
    add_subdirectory (gridgen)
  endif (${PG_GRIDGEN})

  if (${PG_SEARCH})
    add_subdirectory (search)
  endif (${PG_SEARCH})