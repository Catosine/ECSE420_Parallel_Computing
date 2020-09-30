#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void rectificate(unsigned char *original, unsigned char *modified, int width, int height){
	int batch_size = height / blockDim.x;
	int start = threadIdx.x * batch_size;
	for (int i=0; i<batch_size; i++){
		for (int j=0; j<height; j++){
			modified[4*width*(start+i)+4*j+0] = (original[4*width*(start+i)+4*j+0] < 127) ? 127 : original[4*width*(start+i)+4*j+0];
			modified[4*width*(start+i)+4*j+1] = (original[4*width*(start+i)+4*j+1] < 127) ? 127 : original[4*width*(start+i)+4*j+1];	
			modified[4*width*(start+i)+4*j+2] = (original[4*width*(start+i)+4*j+2] < 127) ? 127 : original[4*width*(start+i)+4*j+2];
			modified[4*width*(start+i)+4*j+3] = (original[4*width*(start+i)+4*j+3] < 127) ? 127 : original[4*width*(start+i)+4*j+3];
		}
	}
}

/*
__global__ void test(void){
	printf("threadId: %d, blockId: %d, blockDim: %d\n", threadIdx.x, blockIdx.x, blockDim.x);
}

int main(){
	test <<<1,2>>>();
	cudaDeviceSynchronize();
}
*/
