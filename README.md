# GPU_MCM
GPU_MCM
Implementation of Monte Carlo Simulation on GPUs
This program is to compare the serial and parallel implementation of MonteCarlo pi. 
The following are included in the module:
1. All source files 
2. Final executables
3. Script to collect data for plots
4. Script to plot using gnuplot

BEFORE YOU BEGIN

Clone the repository:
$ git clone https://github.com/shriya312/GPU_MCM.git

The following have to be run each new shell opened:
On stampede, load necessary modules by:
$ module load cuda
Set MC_HOME:
$ MC_HOME=directory/where/rep/was/cloned

GENERAL INSTRUCTIONS

Part 1: Instructions:
$ cd GPU_MCM/src
$ make
$ cd ../bin/
$ ./cudaMC_pi <options> (-h for help and usage)

Part 2: Generate data for scripts
1. Generate the executable as above
2. 
   $ cd GPU_MCM
   $ scripts/./gen_data.sh
 This generates data.dat for plotting

Part 3: Generate plots
1. Run Part 1 and Part 2
2. 
   $ cd GPU_MCM
   $ scripts/./plot.sh 
Reads data.dat and generates plot1.png and plot2.png

DEPENDENCIES:

Part 1: 
1. CUDA6
2. g++
Part 2: 
<none>
Part 3:
gnuplot

DIRECTORY STRUCTURE:

src/ : Contains all source files and Makefile
	main.cpp : Main file
	gpu_impl.cu : GPU implementation file
	serial_impl.c : CPU implementation file
bin/ : Contains executable
	cudaMC_pi : Executable generated
scripts/ : Contains necessary scripts
	gen_data.sh : Run this script
	extract.py : Python script to help
	plot.sh : To plot the given data
plots/ : Contains plots in ".png" file
	plot1.png : Compares serial vs Parallel impl
	plot2.png: Compares serial, my impl and thrust

