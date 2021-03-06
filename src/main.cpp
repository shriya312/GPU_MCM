#include "CycleTimer.h"
#include "MCpi.h"
using std::cout;
using std::endl;
long inside =0;
long outside = 0;

double mcm_pi (float * rand_samples,long int samples) {
	float x, y;
	for (long i =0; i < samples;i++) {
		x = rand_samples[i*2];
		y = rand_samples[i*2 + 1];
		//x = (double) rand()/ (double) RAND_MAX;
		//y = (double) rand()/ (double) RAND_MAX;
		if ((x*x + y*y) > 1) 
			outside ++;
		else
			inside ++; 	

	}
	//printf ("%d %d", outside, inside);
	return 4*(double)inside/(double)samples;
}

void usage () {
	printf ("Usage: MonteCarlo_pi\n");
	printf ("Estimation of the value of pi using Monte Carlo method \n");
	printf ("Program Options: \n");
	printf (" -n <INT>	 Number of samples \n");
	printf (" -t 		 Use thrust library\n");
	printf (" -v 	 	 Run serial \n");
	printf (" -c 		 Print Cuda Information\n");
	printf (" -h 		 Help\n");
}
int main(int argc,char **argv)
{
	int length = 512;
	double pi_val = 0;
	bool useThrust = false;
	bool runSerial = false;
	int opt; 
	int spt = 2;
	static struct option long_options[] = {{"arraysize",1,0,'n'},{"spt",1,0,'p'},{"thrust",0,0,'t'},{"serial",0,0,'v'},{"config",0,0,'c'},{"help",0,0,'h'}};
	while ((opt = getopt_long(argc,argv,"n:p:tvch", long_options, NULL)) != EOF) {
	switch (opt) {
		case 'n' :
			length = atoi(optarg);
			break;
		case 'p' :
			spt = atoi(optarg);
			break;
		case 't' :
			useThrust = true;
			break;
		case 'v' :
			runSerial = true;
			break;
		case 'c' :
			printCudaInfo();
			return 0;
		case 'h' :
			usage();
			return 0;
		default:
			usage();
			return 1;
	}
	}
	printf ("Checking if inputs passed are valid...\n");
	printf ("Samples per thread : %d \n", spt);
	int num_blocks = length/(spt*512);
	if (num_blocks >= 65536 ) {
		printf ("[ERROR] Max number of blocks exceeded \n");
		return 1;
	}
	printf ("OK..\n");
	double serialTime =0 , gpuTime= 0, thrustTime = 0;
	float *x;
	x = (float *) malloc (2*length *sizeof(float));
	for (int i =0 ; i < length ;i ++) {
		x[i*2] = (float) rand()/ (float) RAND_MAX;
		x[i*2 + 1] = (float) rand()/ (float) RAND_MAX;
	}
	printf ("\n");	
	printf ("=======PI ESTIMATION USING MONTE CARLO SIMULATION=======\n");
	printf ("\n");	
	if (useThrust) {
		thrustTime = thrustmain (length);	
		cout << "THRUST TIME : " << thrustTime << endl;
		printf ("\n");	
		pi_val = cudaMC_noVal(length, gpuTime, spt);
		cout << "GPU TIME : " << gpuTime << endl;
		printf ("\n");	
		
	} else {
		pi_val = cudaMC_pi(x, length, gpuTime, spt);
		cout << "GPU TIME : " << gpuTime << endl;
		printf ("\n");	
	}
	if (runSerial) {
		printf ("=====================VALIDATION=========================\n");
		printf ("\n");	
		double start = CycleTimer::currentSeconds();
		double serial_pi = mcm_pi(x,length);
		printf ("Value of pi from CPU Implementation : %f\n", serial_pi);
		double end = CycleTimer::currentSeconds();
		serialTime = end - start;
		cout << "SERIAL TIME : " << serialTime << endl;
		printf ("\n");	
	}

	free(x);		
}	
