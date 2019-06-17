#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.9)

# The name of our project
set (Target_NAME gridgen)

# Adding source code files
set (Sources src/main.cpp)

# Variable to hold ALL files to build and be visible in IDE
set (Build_FILES ${Sources})

# Create filters for IDEs
source_group ("Sources" FILES ${Sources})

# Specify executable sources
add_executable (${Target_NAME} ${Build_FILES})

# Link dependant libraries
target_link_libraries (${Target_NAME} PRIVATE grid)

# Set language standard
set_target_properties (${Target_NAME} PROPERTIES CXX_STANDARD 11
                                                 CXX_STANDARD_REQUIRED ON)

# Add verbose output if opted in for
target_compile_definitions (${Target_NAME} PRIVATE $<$<BOOL:${PG_VERBOSE}>:VERBOSE>)

# Enable additional warnings
target_compile_options (${Target_NAME} PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                               $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->)