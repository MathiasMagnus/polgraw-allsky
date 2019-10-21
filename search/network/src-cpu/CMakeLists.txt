#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.1.0)

# The name of our project
set (Target_NAME gwsearch-cpu)

# Adding source code files according to configuration
set (HDRS)
set (SRCS)

set (HDRS ${CMAKE_CURRENT_SOURCE_DIR}/auxi.h
          ${CMAKE_CURRENT_SOURCE_DIR}/init.h
          ${CMAKE_CURRENT_SOURCE_DIR}/jobcore.h
          ${CMAKE_CURRENT_SOURCE_DIR}/settings.h
          ${CMAKE_CURRENT_SOURCE_DIR}/struct.h
          ${CMAKE_CURRENT_SOURCE_DIR}/timer.h)

set (SRCS ${CMAKE_CURRENT_SOURCE_DIR}/auxi.c
          ${CMAKE_CURRENT_SOURCE_DIR}/main.c
          ${CMAKE_CURRENT_SOURCE_DIR}/settings.c
          ${CMAKE_CURRENT_SOURCE_DIR}/timer.c
          ${CMAKE_CURRENT_SOURCE_DIR}/init.c
          ${CMAKE_CURRENT_SOURCE_DIR}/jobcore.c)

# Variable to hold ALL files to build and be visible in IDE
set (BUILD ${HDRS} ${SRCS})

# Create filters for IDEs
source_group ("Headers" FILES ${HDRS})
source_group ("Sources" FILES ${SRCS})

# Specify executable sources
add_executable (${Target_NAME} ${BUILD})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${Target_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ${FFTW_INCLUDE_DIRS})

# Link dependant libraries
target_link_libraries (${Target_NAME} GSL::gsl
                                      GSL::gslcblas
                                      ${FFTW_LIBRARIES})

# Create project groups for IDEs and set strict language standard conformance
set_target_properties (${Target_NAME} PROPERTIES FOLDER "CPU"
                                                 C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

# Add verbose output if opted in for
target_compile_definitions (${Target_NAME} PRIVATE $<$<BOOL:${PG_VERBOSE}>:VERBOSE>
                                                   $<$<BOOL:${BUILD_TESTING}>:TESTING>)

#
# Set compiler-specific variables
#
  # If compiling with Microsoft Visual C
  if (CMAKE_C_COMPILER_ID MATCHES "MSVC")

    # Add implementations of complex operations
    target_sources(${Target_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/floats.h
                                          ${CMAKE_CURRENT_SOURCE_DIR}/floats.c)
    
    # Link it to our executable
    target_link_libraries (${Target_NAME} getopt Ws2_32)
    target_include_directories (${Target_NAME} PUBLIC ${POSIX_INCLUDE_DIR})
    
    # Silence all Secure CRT warnings (remove later)
    target_compile_definitions (${Target_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)

    # Highest sensible warning level and ISO conformance
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

  endif (CMAKE_C_COMPILER_ID MATCHES "MSVC")

  # If compiling with GCC
  if (CMAKE_C_COMPILER_ID MATCHES "GNU")
  
      # Add BSD language extension to C11 for DT_DIR
      target_compile_definitions (${Target_NAME} PRIVATE _BSD_SOURCE)

      # Add libm to have math functions of CRT
      target_link_libraries(${Target_NAME} m)

      # Highest sensible warning level and ISO conformance
      target_compile_options(${Target_NAME} PRIVATE "-Wall"
                                                    "-Wextra"
                                                    "-pedantic")
  
    endif (CMAKE_C_COMPILER_ID MATCHES "GNU")

#########################
#                       #
#         Sigen         #
#                       #
#########################

# The name of our project
set (Target_NAME sigen)

# Adding source code files according to configuration
set (HDRS)
set (SRCS)

set (HDRS ${CMAKE_CURRENT_SOURCE_DIR}/auxi.h
          ${CMAKE_CURRENT_SOURCE_DIR}/settings.h
          ${CMAKE_CURRENT_SOURCE_DIR}/struct.h)

set (SRCS ${CMAKE_CURRENT_SOURCE_DIR}/auxi.c
          ${CMAKE_CURRENT_SOURCE_DIR}/settings.c
          ${CMAKE_CURRENT_SOURCE_DIR}/sigen.c)

# Variable to hold ALL files to build and be visible in IDE
set (BUILD ${HDRS} ${SRCS})

# Create filters for IDEs
source_group ("Headers" FILES ${HDRS})
source_group ("Sources" FILES ${SRCS})

# Specify executable sources
add_executable (${Target_NAME} ${BUILD})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${Target_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ${FFTW_INCLUDE_DIRS})

# Link dependant libraries
#target_link_libraries (${Target_NAME} GSL::gsl
#                                      GSL::gslcblas
#                                      ${FFTW_LIBRARIES})

# Create project groups for IDEs and set strict language standard conformance
set_target_properties (${Target_NAME} PROPERTIES FOLDER "CPU"
                                                 C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

# Add verbose output if opted in for
target_compile_definitions (${Target_NAME} PRIVATE $<$<BOOL:${PG_VERBOSE}>:VERBOSE>
                                                   $<$<BOOL:${BUILD_TESTING}>:TESTING>)

#
# Set compiler-specific variables
#
  # If compiling with Microsoft Visual C
  if (CMAKE_C_COMPILER_ID MATCHES "MSVC")

    # Add implementations of complex operations
    target_sources(${Target_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/floats.h
                                          ${CMAKE_CURRENT_SOURCE_DIR}/floats.c)
    
    # Link it to our executable
    target_link_libraries (${Target_NAME} getopt Ws2_32)
    target_include_directories (${Target_NAME} PUBLIC ${POSIX_INCLUDE_DIR})
    
    # Silence all Secure CRT warnings (remove later)
    target_compile_definitions (${Target_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)

    # Highest sensible warning level and ISO conformance
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

  endif (CMAKE_C_COMPILER_ID MATCHES "MSVC")

  # If compiling with GCC
  if (CMAKE_C_COMPILER_ID MATCHES "GNU")
  
      # Add BSD language extension to C11 for DT_DIR
      target_compile_definitions (${Target_NAME} PRIVATE _BSD_SOURCE)

      # Add libm to have math functions of CRT
      target_link_libraries(${Target_NAME} m)

      # Highest sensible warning level and ISO conformance
      target_compile_options(${Target_NAME} PRIVATE "-Wall"
                                                    "-Wextra"
                                                    "-pedantic")
  
    endif (CMAKE_C_COMPILER_ID MATCHES "GNU")

#########################
#                       #
#      gauss-xdat       #
#                       #
#########################

# The name of our project
set (Target_NAME gauss-xdat)

# Adding source code files according to configuration
set (HDRS)
set (SRCS)

set (HDRS ${CMAKE_CURRENT_SOURCE_DIR}/timer.h)

set (SRCS ${CMAKE_CURRENT_SOURCE_DIR}/timer.c
          ${CMAKE_CURRENT_SOURCE_DIR}/gauss-xdat.c)

# Variable to hold ALL files to build and be visible in IDE
set (BUILD ${HDRS} ${SRCS})

# Create filters for IDEs
source_group ("Headers" FILES ${HDRS})
source_group ("Sources" FILES ${SRCS})

# Specify executable sources
add_executable (${Target_NAME} ${BUILD})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${Target_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link dependant libraries
target_link_libraries (${Target_NAME} GSL::gsl
                                      GSL::gslcblas)

# Create project groups for IDEs and set strict language standard conformance
set_target_properties (${Target_NAME} PROPERTIES FOLDER "CPU"
                                                 C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

# Add verbose output if opted in for
target_compile_definitions (${Target_NAME} PRIVATE $<$<BOOL:${PG_VERBOSE}>:VERBOSE>
                                                   $<$<BOOL:${BUILD_TESTING}>:TESTING>)

#
# Set compiler-specific variables
#
  # If compiling with Microsoft Visual C
  if (CMAKE_C_COMPILER_ID MATCHES "MSVC")
    
    # Link it to our executable
    target_link_libraries (${Target_NAME} getopt Ws2_32)
    target_include_directories (${Target_NAME} PUBLIC ${POSIX_INCLUDE_DIR})
    
    # Silence all Secure CRT warnings (remove later)
    target_compile_definitions (${Target_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)

    # Highest sensible warning level and ISO conformance
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

  endif (CMAKE_C_COMPILER_ID MATCHES "MSVC")

  # If compiling with GCC
  if (CMAKE_C_COMPILER_ID MATCHES "GNU")
  
      # Add BSD language extension to C11 for DT_DIR
      target_compile_definitions (${Target_NAME} PRIVATE _BSD_SOURCE)

      # Add libm to have math functions of CRT
      target_link_libraries(${Target_NAME} m)

      # Highest sensible warning level and ISO conformance
      target_compile_options(${Target_NAME} PRIVATE "-Wall"
                                                    "-Wextra"
                                                    "-pedantic")
  
    endif (CMAKE_C_COMPILER_ID MATCHES "GNU")

#########################
#                       #
#         CTest         #
#                       #
#########################

# Add unit test
if (BUILD_TESTS)

  add_test(NAME ${Target_NAME}
           COMMAND "${Target_NAME} -data ${POLGRAW_SEARCH_NETWORK_ROOT}/../../testdata/test-data-network-injection -ident 205 -label J0322+0441 -band 000 -fpo 1391.3 -r ${POLGRAW_SEARCH_NETWORK_ROOT}/../../testdata/range-J0322+0441.txt")
		
endif (BUILD_TESTS)
