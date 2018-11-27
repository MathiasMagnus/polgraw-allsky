# Find Gnuplot
find_package (Gnuplot REQUIRED)

# Commands and targets that generate data files
add_custom_command (COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gwsearch-ocl
                            -data ${PROJECT_SOURCE_DIR}/../../testdata/test-gpu-cpu-injected/
							-output .
							-ident 001
                            -band 0666
                            -dt 2
                            --nocheckpoint
                            -range ${PROJECT_SOURCE_DIR}/../../testdata/test-gpu-cpu-injected/range_zoom.dat
                            -usedet H1L1
                            -device gpu
					WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
					OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ocl_triggers_001_666_1.bin
					DEPENDS gwsearch-ocl
					COMMENT "Generating data for zoomed freq:SNR plot of the injected data")

add_custom_target (data-injected-zoomed DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ocl_triggers_001_666_1.bin)

add_custom_target (data DEPENDS data-injected-zoomed)

# Commands and targets that generate plots
add_custom_command (COMMAND ${GNUPLOT_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src-opencl/doc/script/freq_vs_SNR.plt
                    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                    OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/freq_vs_SNR.png
                    DEPENDS ${PROJECT_SOURCE_DIR}/src-opencl/doc/script/freq_vs_SNR.plt data-injected-zoomed
                    COMMENT "Generating zoomed freq:SNR plot of the injected data")

add_custom_target (plot-injected-zoomed DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/freq_vs_SNR.png)

add_custom_target (plot DEPENDS plot-injected-zoomed)
