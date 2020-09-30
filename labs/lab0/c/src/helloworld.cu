#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void kernel(void){
	printf("hello world from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(void){
	kernel <<<10, 10>>> ();
	cudaDeviceSynchronize();
	return 0;
}
