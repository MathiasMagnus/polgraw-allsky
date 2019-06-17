#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.9)

# The name of our project
set (Target_NAME grid)

# Adding source code files
set (Headers inc/CellS2.h
             inc/DataS2.h
             inc/DensityS1.h
             inc/DensityS2.h
             inc/FisherRM.h
             inc/GridS1.h
             inc/GridS2.h
             inc/Node.h
             inc/num.h
             inc/OptionValue.h
             inc/ReadConfig.h
             inc/ReadEphemeris.h
             inc/stringmanip.h)

set (Sources src/CellS2.cpp
             src/DataS2.cpp
             src/DensityS1.cpp
             src/DensityS2.cpp
             src/FisherRM.cpp
             src/GridS1.cpp
             src/GridS2.cpp
             src/Node.cpp
             src/num.cpp
             src/OptionValue.cpp
             src/ReadConfig.cpp
             src/ReadEphemeris.cpp
             src/stringmanip.cpp)

# Variable to hold ALL files to build and be visible in IDE
set (Build_FILES ${Headers} ${Sources})

# Create filters for IDEs
source_group ("Headers" FILES ${Headers})
source_group ("Sources" FILES ${Sources})

# Specify library sources
add_library (${Target_NAME} ${Build_FILES})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${Target_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Set language standard
set_target_properties (${Target_NAME} PROPERTIES CXX_STANDARD 11
                                                 CXX_STANDARD_REQUIRED ON)

# Add verbose output if opted in for
target_compile_definitions (${Target_NAME} PRIVATE $<$<BOOL:${PG_VERBOSE}>:VERBOSE>)

# Enable additional warnings
target_compile_options (${Target_NAME} PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                               $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /EHsc>)