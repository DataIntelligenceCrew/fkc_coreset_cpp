#!/bin/sh

cd fkc_coreset/build
make
for DR in 50 100 200 300 400 500 600 700 800 900 
do 
    ./fkc_coreset cifar10 30 $DR > /localdisk2/fkc_coreset_cpp_results/metric_files/two_phase_cifar10_30_$DR.txt
done  