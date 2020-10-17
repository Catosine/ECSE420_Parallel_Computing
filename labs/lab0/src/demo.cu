#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void vecAddKernel(int *a, int *b, int *c){
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main(){
	int v1[5] = {1,2,3,4,5};
	int v2[5] = {6,7,8,9,10};

	int *cuda_a, *cuda_b, *cuda_c;

	cudaMalloc((void**) &cuda_a, 5*sizeof(int));
	cudaMalloc((void**) &cuda_b, 5*sizeof(int));
	cudaMalloc((void**) &cuda_c, 5*sizeof(int));

	cudaMemcpy(cuda_a, v1, 5*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, v2, 5*sizeof(int), cudaMemcpyHostToDevice);

	vecAddKernel <<<1,5>>> (cuda_a, cuda_b, cuda_c);
	cudaDeviceSynchronize();

	int v3[5];
	cudaMemcpy(v3, cuda_c, 5*sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
	
	for (int i=0; i<5; i++){
		printf("%d: %d + %d = %d\n", i, v1[i], v2[i], v3[i]);
	}

}
