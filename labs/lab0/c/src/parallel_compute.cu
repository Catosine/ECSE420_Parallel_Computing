#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" void rectificate(unsigned char *original, unsigned char*modifed, int width, int height, int n_thread);

__global__ void rectificate_kernel(unsigned char *original, unsigned char *modified, int width, int height){
	int batch_size = 1;
	if (blockDim.x <= height){
		batch_size = height / blockDim.x;
	}
	if (threadIdx.x < height){
		int offset = threadIdx.x * batch_size;
		for (int i=0; i<batch_size; i++){
			for (int j=0; j<height; j++){
				modified[4*width*(offset+i)+4*j+0] = (original[4*width*(offset+i)+4*j+0] < 127) ? 127 : original[4*width*(offset+i)+4*j+0];
				modified[4*width*(offset+i)+4*j+1] = (original[4*width*(offset+i)+4*j+1] < 127) ? 127 : original[4*width*(offset+i)+4*j+1];	
				modified[4*width*(offset+i)+4*j+2] = (original[4*width*(offset+i)+4*j+2] < 127) ? 127 : original[4*width*(offset+i)+4*j+2];
				modified[4*width*(offset+i)+4*j+3] = (original[4*width*(offset+i)+4*j+3] < 127) ? 127 : original[4*width*(offset+i)+4*j+3];
			}
		}
	}
}

extern "C" void rectificate(unsigned char *original, unsigned char *modified, int width, int height, int n_thread){
	size_t png_size = width*height*4*sizeof(unsigned char);
		
	unsigned char* cuda_image;
	cudaMalloc((void**) &cuda_image, png_size);
	cudaMemcpy(cuda_image, original, png_size, cudaMemcpyHostToDevice);

	unsigned char* cuda_new_image;
	cudaMalloc((void**) &cuda_new_image, png_size);

	rectificate_kernel <<<1, n_thread>>> (cuda_image, cuda_new_image, width, height);
	cudaDeviceSynchronize();

	unsigned char* new_image = (unsigned char*)calloc(1, png_size);
	cudaMemcpy(modified, cuda_new_image, png_size, cudaMemcpyDeviceToHost);
		
	cudaFree(cuda_image);
	cudaFree(cuda_new_image);
}
