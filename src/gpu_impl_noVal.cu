#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include "MCpi.h"
#include "CycleTimer.h"
#define THREADS 512

//#define SPT 4
__device__ inline void rand_cu (int &result) {
	int thid = threadIdx.x;
	curandState_t state;
	curand_init (clock64(),thid,0,&state);
	result = curand(&state)%RAND_MAX;
	if (result < 0) result = -result;
	//printf (" %d ", result);	
}

__global__ void monteCarlopi_noVal (int *din, int length, int SPT) {
	
	 __shared__ int sm[THREADS];

	int thid = threadIdx.x;
	int t = threadIdx.x + blockIdx.x*blockDim.x;
	int a = thid;
	float x,y;
	int x1,y1;
	int ind1, ind2;
	int num = 0;	
	for (int i =0 ; i< SPT; i++ ) {
		ind1 = t*SPT*2 + 2*i;
		if (ind1 < 2*length) { 
			rand_cu(x1);	
			rand_cu(y1); 
			x = (double)x1/RAND_MAX;
			y = (double) y1/RAND_MAX;
			if ((x*x + y*y) > 1)
			{}
			else
				num+=1;
		}
		//printf ("%f %f ", x, y); 
		//if (t*SPT + i + 1 < length) {
		//	if ((x*x + y*y) < 1)
		//		num += 1; 
		//}
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


__global__ void prefixSum_noVal ( int *d_in , int *temp, int length) {


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


double monteCarlopi_noVal(int num_blocks, int length, double & pi_val, int SPT) {

	// store the count in global array
	int *d_in, *temp;
	int  i = 0;
	double start = CycleTimer::currentSeconds();
	cudaMalloc((void**)&d_in, num_blocks*sizeof(int));
	cudaMalloc((void**)&temp, num_blocks*sizeof(int));
	//printf ("num blocks %d ", num_blocks);
	monteCarlopi_noVal<<<num_blocks,THREADS>>>( d_in, length, SPT);
	if (num_blocks == 1) {
	}
	else {
		int num_blocks1 = num_blocks/(2*THREADS);
		if (num_blocks1 < 1) num_blocks1 = 1;
		while (num_blocks > 1) {
			if (i%2 == 0)  
				prefixSum_noVal<<<num_blocks1,THREADS>>>(d_in, temp, num_blocks);
			else 
				prefixSum_noVal<<<num_blocks1,THREADS>>>(temp, d_in, num_blocks);
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
	pi_val = (double) 4*total_sum/ double (length);	
	cudaFree(d_in);
	cudaFree(temp);
	double end = CycleTimer::currentSeconds();
	return (end - start);
}


double cudaMC_noVal(int length, double & gpuTime, int samplesThread)
{
	double pi_val; 
	int SPT = samplesThread;
	int num_blocks = length/(SPT*THREADS);
	if (length % (SPT*THREADS) != 0) num_blocks++;	
	gpuTime = monteCarlopi_noVal(num_blocks,length,pi_val, SPT); 
	printf ("Value of pi from GPU Implementation : %f\n", pi_val);	
	return pi_val; 
}

