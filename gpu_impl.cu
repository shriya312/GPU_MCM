#include <stdio.h>
#include <stdlib.h>
#define THREADS 128

__global__ void hello()
{
    printf("Hello world! I'm a thread in block %d\n", blockIdx.x);
}


__global__ void monteCarlopi (int *din) {
	
	extern __shared__ int sm[2*THREADS];

	int t = blockIdx.x*blockDim.x + threadIdx.x;
	int thid = threadIdx.x;

	int a = 2*thid;
	int b = 2*thid + 1;

	double x,y;
	
	x = rand()/RAND_MAX;
	y = rand()/RAND_MAX;
	if ((x*x + y*y) < 1)
		sm[a] = 1; 
	else 
		sm[a] = 0;
	x = rand()/RAND_MAX;
	y = rand()/RAND_MAX;
	if ((x*x + y*y) < 1)
		sm[b] = 1;
	else
		sm[b] = 0; 
	// wait for all threads to finish

	__syncthreads();

	// find sum of points inside the rectangle using sum function

	sm[a] += sm[b]; // 128 elements
	__syncthreads();
	
	int id;

	if (thid < 64) {  // 64 elements
		idx = thid * 4;	
		sm[idx] += sm[idx+2];
	}
 	__syncthreads();	
	
	if (thid < 32) { // 32 elements
		idx = thid*8;
		sm[idx] += sm[idx + 4]; 
		
	}
 	__syncthreads();	

	if (thid < 16) { // 16 elements
		idx = thid*16;
		sm[idx] += sm[idx+8];
	}  
 	__syncthreads();	

	if (thid < 8) { // 8 elements
		idx = thid*32;
		sm[idx] += sm[idx+16];
	}  
 	__syncthreads();	
	
	if (thid < 4) { // 4 elements
		idx = thid*64;
		sm[idx] += sm[idx+32];
	}  
 	__syncthreads();	
		
	if (thid < 2) { // 2 elements
		idx = thid*128;
		sm[idx] += sm[idx+64];
	}  
 	__syncthreads();	

	if (thid == 0) { //  1 elements 
		sm[idx] += sm[idx+128];
		d_in[blockIdx.x] = sm[a];
	}

}


__global__ void prefixSum ( int *d_in , int *sum, int length) {


	extern __shared__ int sm[2*THREADS];
	int thid = threadIdx.x;
	int t = blockIdx.x*blockDim.x + threadIdx.x;
	int a = 2*thid;
	int b = 2*thid + 1;
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
	
	int id;

	if (thid < 64) {  // 64 elements
		idx = thid * 4;	
		sm[idx] += sm[idx+2];
	}
 	__syncthreads();	
	
	if (thid < 32) { // 32 elements
		idx = thid*8;
		sm[idx] += sm[idx + 4]; 
		
	}
 	__syncthreads();	

	if (thid < 16) { // 16 elements
		idx = thid*16;
		sm[idx] += sm[idx+8];
	}  
 	__syncthreads();	

	if (thid < 8) { // 8 elements
		idx = thid*32;
		sm[idx] += sm[idx+16];
	}  
 	__syncthreads();	
	
	if (thid < 4) { // 4 elements
		idx = thid*64;
		sm[idx] += sm[idx+32];
	}  
 	__syncthreads();	
		
	if (thid < 2) { // 2 elements
		idx = thid*128;
		sm[idx] += sm[idx+64];
	}  
 	__syncthreads();	

	if (thid == 0) { //  1 elements 
		sm[idx] += sm[idx+128];
		*sum = sm[idx];		
	}




}


void monteCarlopi(int num_blocks, int length, double pi_val) {

	// store the count in global array
	int *d_in, *sum;
	cudaMalloc(d_in, num_blocks*sizeof(int));
	cudaMalloc(sum, sizeof(int));
	monteCarlopi<<<num_blocks,THREADS, 2*THREADS>>>(d_in);
	if (num_blocks <= 2*THREADS) {
		prefixSum<<<1, THREADS>>>(d_in, &sum, num_blocks);
	} else {
		
		int num_blocks1 = num_blocks/(2*THREADS);
		while (num_blocks1 > 1) {
			num_blocks1 = num_blocks/(2*THREADS);
			prefixSum<<<num_block1,THREADS(d_in, &sum, num_blocks);
			num_blocks = num_blocks1;
		}
		prefixSum<<<1,THREADS>>>(d_in, &sum, num_blocks);
	}
	cudaFree(d_in);
	int h_sum[1];
	cudaMemcpy(h_sum, sum, sizeof(int), cudaMemcpyDeviceToHost );	
	cudaFree(sum);	
	double pi = 4*(double)h_sum[0]/(double)length;
}

int main(int argc,char **argv)
{
	int length = 0;
	double pi_val = 0;

	int num_blocks = length/(2*THREADS);

	monteCarlopi(num_blocks,length,&pi_val); 	
}
