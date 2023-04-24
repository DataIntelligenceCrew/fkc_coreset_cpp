#!/bin/sh

cd fkc_coreset/build
make
for CF in 30 40 50
do 
    for DR in 30 40 50
    do 
        ./fkc_coreset cifar10 $CF $DR
    done 
done 