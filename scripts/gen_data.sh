#!/bin/bash

num=32768

for p in {10..25}
do
	../bin/./cudaMC_pi -n $num -v >f
	python extract.py >> data.dat
	num=`expr $num \* 2`
	echo $num
done
rm f
mv data.dat ../plots/data.dat
