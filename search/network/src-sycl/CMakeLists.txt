#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.9)

# The name of our project
set (Target_NAME gwsearch-sycl)

# Include modules
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

# Find dependencies
find_package(TCLAP REQUIRED)
find_package(OpenCL REQUIRED)
find_package(clFFT REQUIRED CONFIG)
find_package(clBLAS REQUIRED CONFIG)
find_package(ComputeCpp REQUIRED)

# Adding source code files according to configuration
set (Headers inc/Options.hpp
             inc/Sky.hpp
             inc/AmplitudeModulation.hpp)

set (Sources src/Main.cpp
             src/Options.cpp)

set (SYCL_Sources src/AmplitudeModulation.cpp)

# Variable to hold ALL files to build and be visible in IDE
set (Build_FILES ${Headers} ${Sources} ${SYCL_Sources})

# Create filters for IDEs
source_group ("Headers" FILES ${Headers})
source_group ("Sources" FILES ${Sources})

# Specify executable sources
add_executable (${Target_NAME} ${Build_FILES})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${Target_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc
                                                   ${TCLAP_INCLUDE_PATH}
                                                   ${CLFFT_INCLUDE_DIRS}
                                                   ${CLBLAS_INCLUDE_DIRS})
# Link dependant libraries
target_link_libraries (${Target_NAME} PRIVATE OpenCL::OpenCL
                                              clFFT
                                              clBLAS)
# Create project groups for IDEs
set_target_properties (${Target_NAME} PROPERTIES CXX_STANDARD 14
                                                 CXX_STANDARD_REQUIRED ON)
# Add verbose output if opted in for
target_compile_definitions (${Target_NAME} PRIVATE $<$<BOOL:${PG_VERBOSE}>:VERBOSE>
                                                   $<$<BOOL:${BUILD_TESTING}>:TESTING>)
# Enable additional warnings
target_compile_options (${Target_NAME} PRIVATE $<$<OR:$<C_COMPILER_ID:GNU>,$<C_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                               $<$<C_COMPILER_ID:MSVC>:/W4 /permissive- /EHsc>)
# Add extra SYCL build rules
add_sycl_to_target(TARGET ${Target_NAME}
                   SOURCES ${SYCL_Sources})