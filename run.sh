#!/bin/sh

cd fkc_coreset/build
make
# for DR in 50 100 200 300 400 500 600 700 800 900
for DR in 0.01 0.05 0.10 0.15 0.20
do 
    ./fkc_coreset nyc_taxicab 2 30 $DR > /localdisk2/fkc_coreset_cpp_results/metric_files/gfkc_nyc_taxicab_2_$DR.txt
done  