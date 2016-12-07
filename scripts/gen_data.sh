#!/bin/bash

num=1024

for p in {1..15}
do
	../bin/./cudaMC_pi -n $num -v >f
	python extract.py GPU SERIAL $num >> data1.dat
	../bin/./cudaMC_pi -n $num -t >f
	python extract.py THRUST GPU $num >> data2.dat
	num=`expr $num \* 2`
done
rm f
mv data1.dat ../plots/data1.dat
mv data2.dat ../plots/data2.dat
