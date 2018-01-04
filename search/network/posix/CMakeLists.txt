#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.1.0)

# The name of our project
project (getopt LANGUAGES C)

# Adding source code files according to configuration
set (HDRS)
set (SRCS)

list (APPEND HDRS ${PROJECT_SOURCE_DIR}/inc/dirent.h)
list (APPEND HDRS ${PROJECT_SOURCE_DIR}/inc/getopt.h)

list (APPEND SRCS ${PROJECT_SOURCE_DIR}/src/getopt.c)


# Variable to hold ALL files to build and be visible in IDE
set (BUILD ${HDRS} ${SRCS})

# Create filters for IDEs
source_group ("Headers" FILES ${HDRS})
source_group ("Sources" FILES ${SRCS})

# Specify executable sources
add_library (${PROJECT_NAME} SHARED ${BUILD})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/inc)

target_compile_definitions (${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)

# Set definition to export symbols into library
target_compile_definitions (${PROJECT_NAME} PRIVATE EXPORTS_GETOPT)

set (POSIX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inc PARENT_SCOPE)
