#include <stdio.h>
#include <math.h>
#include <stdlib.h>

long inside =0;
long outside = 0;

double mcm_pi (long int samples) {
	double x, y;
	for (long i =0; i < samples;i++) {
		x = (double) rand()/ (double) RAND_MAX;
		y = (double) rand()/ (double) RAND_MAX;
		if ((x*x + y*y) > 1) 
			outside ++;
		else
			inside ++; 	

	}
	//printf ("%d %d", outside, inside);
	printf ("%f ", 4*(double)inside/(double)samples);
	return 4*(double)inside/(double)samples;
}


int main() {

	mcm_pi(1000000);


}
