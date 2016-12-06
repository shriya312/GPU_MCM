#include <iostream>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <stdio.h>
double cudaMC_pi (float * samples, int length, double & gpuTime, int SPT);
double cudaMC_noVal (int length, double & gpuTime, int SPT);
void printCudaInfo();
double thrustmain(int M);
