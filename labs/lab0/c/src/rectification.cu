#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>
#include "lodepng.h"
#include <time.h>

__global__ void rectificate_kernel(unsigned char *original, unsigned char *modified, int width, int height){
	int batch_size = 1;
	if (blockDim.x <= height){
		batch_size = height / blockDim.x;
	}
	if (threadIdx.x < height){
		int offset = threadIdx.x * batch_size;
		for (int i=0; i<batch_size; i++){
			for (int j=0; j<width; j++){
				int pos = 4*width*(offset+i)+4*j;
				modified[pos+0] = (original[pos+0] < 127) ? 127 : original[pos+0];
				modified[pos+1] = (original[pos+1] < 127) ? 127 : original[pos+1];	
				modified[pos+2] = (original[pos+2] < 127) ? 127 : original[pos+2];
				modified[pos+3] = (original[pos+3] < 127) ? 127 : original[pos+3];
			}
		}
	}
}

float rectificate(unsigned char *original, unsigned char *modified, int width, int height, int n_thread){

	size_t png_size = width*height*4*sizeof(unsigned char);
		
	unsigned char* cuda_image;
	cudaMalloc((void**) &cuda_image, png_size);
	cudaMemcpy(cuda_image, original, png_size, cudaMemcpyHostToDevice);

	unsigned char* cuda_new_image;
	cudaMalloc((void**) &cuda_new_image, png_size);
	
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	
	time_t start, end;

	//cudaEventRecord(start,NULL);
	
	start = clock();
	rectificate_kernel <<<1, n_thread>>> (cuda_image, cuda_new_image, width, height);
	cudaDeviceSynchronize();
	end = clock();

	//cudaEventRecord(stop,NULL);
	//cudaEventSynchronize(stop);

	modified = (unsigned char*)calloc(1, png_size);
	cudaMemcpy(modified, cuda_new_image, png_size, cudaMemcpyDeviceToHost);

	cudaFree(cuda_image);
	cudaFree(cuda_new_image);

	//cudaEventElapsedTime(&elapsed, start, stop);
	
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

	return (double)(end-start)/(double)CLOCKS_PER_SEC;
}

int check_if_png(char *name){
	regex_t reg;
	const char *pattern = "^.*\\.(png)?$";
	regcomp(&reg, pattern, REG_EXTENDED);
	const size_t nmatch = 1;
	regmatch_t pmatch[1];
	int status = regexec(&reg, name, nmatch, pmatch, 0);
	regfree(&reg);
	return (status == REG_NOMATCH) ? 0:1;
}

int main(int argc, char *argv[]){
	printf("ECSE 420 Lab 0: Simple CUDA Processing - Rectification\n");

	if (argc != 4) {
		printf("RuntimeError: Please use the correct input format: ./rectify original.png rectficated.png [n_thread]\n");
		return 1;
	}
	
	if (check_if_png(argv[1]) && check_if_png(argv[2])) {
		unsigned char *original, *modified;
		unsigned error, width, height;
		float time = -1.0;

		error = lodepng_decode32_file(&original, &width, &height, argv[1]);

		if (error) {
			printf("ERROR %u: %s\n", error, lodepng_error_text(error));
			free(original);
			return 1;
		} else {
			modified = (unsigned char*)calloc(width*height*4, sizeof(unsigned char));
			
			time = rectificate(original, modified, width, height, atoi(argv[3])) * 1000;

			error = lodepng_encode32_file(argv[2], modified, width, height);

			int status = 0;
			if (error) {
				printf("ERROR %u: %s\n", error, lodepng_error_text(error));
				status = 1;
			}
			
			free(original);
			free(modified);
			
			printf("Total time: %f ms\nDone\n", time);
			return status;
		}
	} else {
		printf("RuntimeError: Invalid PNG inputs: %s %s\n", argv[1], argv[2]);
		return 1;
	}
}
