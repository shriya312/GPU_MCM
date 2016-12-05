#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include "MCpi.h"
#include "CycleTimer.h"
#define THREADS 512

#define SPT 2
__device__ inline void rand_g (int &result) {
	int thid = threadIdx.x;
	curandState_t state;
	curand_init (clock64(),thid,0,&state);
	result = curand(&state)%RAND_MAX;
	if (result < 0) result = -result;
	//printf (" %d ", result);	
}

__global__ void monteCarlopi (float *samples, int *din, int length) {
	
	 __shared__ int sm[THREADS];

	int thid = threadIdx.x;
	int t = threadIdx.x + blockIdx.x*blockDim.x;
	int a = thid;
	float x,y;
	//int x1,y1;
	int ind1, ind2;
	int num = 0;	
	for (int i =0 ; i< SPT; i++ ) {
		//rand_g(x1);	
		//rand_g(y1);
		ind1 = t*SPT*2 + 2*i;
		ind2 = t*SPT*2 + 2*i + 1;	
		x = samples[ind1] ; //(double)x1/RAND_MAX;
		y = samples[ind2] ; //(double) y1/RAND_MAX;
		//printf ("%f %f ", x, y); 
		if (thid*SPT + i < length) {
			if ((x*x + y*y) < 1)
				num += 1; 
		}
	}
	// wait for all threads to finish
	//__syncthreads();
	//if (thid == 0) {
	////	printf (" rand %d %f ",  x1, x);
	//	for (int i =0;i<2*THREADS;i++)
	//		printf("%d,",sm[i]);

	//}
	//__syncthreads();

	// find sum of points inside the rectangle using sum function

//	sm[a] += sm[b]; // 128 elements
	sm[a] = num;
	__syncthreads();
	
	int idx;

	if (thid < 256) {  // 64 elements
		idx = thid * 2;	
		sm[idx] += sm[idx+1];
	}
 	__syncthreads();	
	
	if (thid < 128) { // 32 elements
		idx = thid*4;
		sm[idx] += sm[idx + 2]; 
		
	}
	__syncthreads();

	if (thid < 64) { // 16 elements
		idx = thid*8;
		sm[idx] += sm[idx+4];
	}  
 	__syncthreads();	

	if (thid < 32) { // 8 elements
		idx = thid*16;
		sm[idx] += sm[idx+8];
	}  
 	//__syncthreads();	
	
	if (thid < 16) { // 4 elements
		idx = thid*32;
		sm[idx] += sm[idx+16];
	}  
 	//__syncthreads();	
	if (thid < 8) { // 4 elements
		idx = thid*64;
		sm[idx] += sm[idx+32];
	}  
 	//__syncthreads();	
	if (thid < 4) { // 4 elements
		idx = thid*128;
		sm[idx] += sm[idx+64];
	}  
 	//__syncthreads();	
		
	if (thid < 2) { // 2 elements
		idx = thid*256;
		sm[idx] += sm[idx+128];
	}  
 	//__syncthreads();	

	if (thid == 0) { //  1 elements 
		sm[idx] += sm[idx+256];
		din[blockIdx.x] = sm[a];
		//printf ("sum : %d \n ", sm[idx]);
		//printf (" sum %d ", sm[a]);	
	}

}


__global__ void prefixSum ( int *d_in , int *temp, int length) {


	__shared__ int sm[2*THREADS];
	int thid = threadIdx.x;
	int t = blockIdx.x*blockDim.x + threadIdx.x;
	int a = 2*thid;
	int b = 2*thid + 1;
	int idx;
	// find sum of points inside the rectangle using sum function

	if (a < length) 
		sm[a] = d_in[2*t];	
	else
		sm[a] = 0;
	if (b < length) 
		sm[b] = d_in[2*t+1];
	else
		sm[b] = 0;

	sm[a] += sm[b]; // 128 elements
	__syncthreads();
	

	if (thid < 256) {  // 64 elements
		idx = thid * 4;	
		sm[idx] += sm[idx+2];
	}
 	__syncthreads();	
	
	if (thid < 128) { // 32 elements
		idx = thid*8;
		sm[idx] += sm[idx + 4]; 
		
	}
 	__syncthreads();	

	if (thid < 64) { // 16 elements
		idx = thid*16;
		sm[idx] += sm[idx+8];
	}  
 	__syncthreads();	

	if (thid < 32) { // 8 elements
		idx = thid*32;
		sm[idx] += sm[idx+16];
	}  
 	__syncthreads();	
	
	if (thid < 16) { // 4 elements
		idx = thid*64;
		sm[idx] += sm[idx+32];
	}  
 	__syncthreads();	
		
	if (thid < 8) { // 2 elements
		idx = thid*128;
		sm[idx] += sm[idx+64];
	}  
 	__syncthreads();	
	if (thid < 4) { // 2 elements
		idx = thid*256;
		sm[idx] += sm[idx+128];
	}  
 	__syncthreads();	
	if (thid < 2) { // 2 elements
		idx = thid*512;
		sm[idx] += sm[idx+256];
	}  
 	__syncthreads();	


	if (thid == 0) { //  1 elements 
		sm[idx] += sm[idx+512];
		temp[blockIdx.x] = sm[idx];
		//printf ("sum p : %d \n ", sm[idx]);
	}

}


__global__ void copy_kernel (int *out , int *in, int length) {
	int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (2*t < length) 
		out[2*t] = in[2*t];
	if ((2*t + 1) < length)
		out[2*t + 1] = in[2*t + 1];

}

double monteCarlopi(float *samples, int num_blocks, int length, double & pi_val) {

	// store the count in global array
	int *d_in, *temp;
	int  i = 0;
	cudaMalloc((void**)&d_in, num_blocks*sizeof(int));
	cudaMalloc((void**)&temp, num_blocks*sizeof(int));
	//printf ("num blocks %d ", num_blocks);
	double start = CycleTimer::currentSeconds();
	monteCarlopi<<<num_blocks,THREADS>>>(samples, d_in, length);
	if (num_blocks == 1) {
	}
	else {
		int num_blocks1 = num_blocks/(2*THREADS);
		if (num_blocks1 < 1) num_blocks1 = 1;
		while (num_blocks > 1) {
			if (i%2 == 0)  
				prefixSum<<<num_blocks1,THREADS>>>(d_in, temp, num_blocks);
			else 
				prefixSum<<<num_blocks1,THREADS>>>(temp, d_in, num_blocks);
			i++;
			num_blocks = num_blocks1;
			num_blocks1 = num_blocks/(2*THREADS);
			if (num_blocks1 < 1) num_blocks1 = 1;
			//copy_kernel<<<num_blocks1, THREADS>>> (d_in, temp, num_blocks);
		}
	}
	int total_sum;
	if (i%2 == 1)
		cudaMemcpy(&total_sum, &temp[0], sizeof(int), cudaMemcpyDeviceToHost );
	else
		cudaMemcpy(&total_sum, &d_in[0], sizeof(int), cudaMemcpyDeviceToHost );
	//printf ("total sum : %d ", total_sum);	
	double end = CycleTimer::currentSeconds();
	pi_val = (double) 4*total_sum/ double (length);	
	cudaFree(d_in);
	cudaFree(temp);
	return (end - start);
}


double cudaMC_pi(float *rand_samples, int length, double & gpuTime)
{
	double pi_val; 
	float *d_randSamples;
	cudaMalloc ((void**)&d_randSamples, 2*length* sizeof(float));
	cudaMemcpy (d_randSamples, rand_samples, sizeof(float) *2 *length , cudaMemcpyHostToDevice);
	int num_blocks = length/(SPT*THREADS);
	
	gpuTime = monteCarlopi(d_randSamples,num_blocks,length,pi_val); 
	printf ("Value of pi from GPU Implementation : %f\n", pi_val);	
	cudaFree(d_randSamples);	
	return pi_val; 
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
