# set term postscript eps size 1280,720
set term png size 1280,720
set out "freq_vs_SNR.png"

set title "SNR(Hz)"

set autoscale

set xlabel "Frequency [Hz/band]"
set ylabel "SNR [1]"

plot "ocl_triggers_001_666_2.bin" using 1:5