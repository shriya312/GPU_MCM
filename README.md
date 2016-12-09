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

GENERAL INSTRUCTIONS

Part 1: Instructions:
$ cd GPU_MCM/src
$ make
$ cd ../bin/
$ ./cudaMC_pi <options> (-h for help and usage)

Part 2: Generate data for scripts
1. Generate the executable as above
2. 
   $ cd GPU_MCM/scripts
   $ ./gen_data.sh
 This generates data1.dat  data2.dat for plotting in plots/

Part 3: Generate plots
1. Run Part 1 and Part 2
2. 
   $ cd GPU_MCM/plots
   $ gnuplot plot_script.sh 
Reads data1.dat and data2.dat and generates cpu_gpu.png and thrust_gpu.png

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
	gpu_impl_noVal.cu: GPU Implementation to compare with thrust
	monte_carlo.cu : Thrust Library implementation
	CycleTimer.h : header file with functions to measure execution time
	MC_pi.h: header file with includes and function declarations
bin/ : Contains executable
	cudaMC_pi : Executable generated
scripts/ : Contains necessary scripts
	gen_data.sh : Run this script
	extract.py : Python script to help
plots/ : Contains plots in ".png" file
	cpu_gpu.png : Compares serial vs Parallel impl
	thrust_gpu.png: Compares serial, my impl and thrust
	plot_script.sh : Script that generates plots
