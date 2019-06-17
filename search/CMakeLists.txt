#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of 'cmake' necessary to build this project
cmake_minimum_required (VERSION 3.0.0)
	
# Behavioural options for the project
option (PG_NETWORK "Build the network search library and utility" ON)
option (PG_ONEDETECTOR "Build the one-detector search librariey and utility" OFF)
option (PG_SPOTLIGHT "Build the one-detector search librariey and utility" OFF)

#
# Process targets based on user-input
#
if (${PG_NETWORK})
  add_subdirectory (network)
endif (${PG_NETWORK})

if (${PG_ONEDETECTOR})
  message (FATAL_ERROR "One-detector search suite CMake build automation not yet implemented.")
endif (${PG_ONEDETECTOR})

if (${PG_SPOTLIGHT})
  message (FATAL_ERROR "Spotlight search suite CMake build automation not yet implemented.")
endif (${PG_SPOTLIGHT})