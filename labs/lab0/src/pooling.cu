#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lodepng.h"

__global__ void compression(unsigned char* image, unsigned char* new_image, unsigned width, unsigned int size, unsigned int blocks_per_row) {
	unsigned int index = threadIdx.x + (blockIdx.x % blocks_per_row) * blockDim.x;
	unsigned int new_index = (threadIdx.x + blockIdx.x * blockDim.x) + 4;

	if (index < size) {
		for (int i = 0; i < 4; i++) {						// iterate through R, G, B, A
			unsigned int max = image[index];
			if (image[index + 4 + i] > max) {				// pixel to the right
				max = image[index + 4 + i];
			}
			if (image[index + (4 * width) + i] > max) {		// pixel below
				max = image[index + (4 * width) + i];
			}
			if (image[index + (4 * width) + 4 + i] > max) {	// pixel below & to the right
				max = image[index + (4 * width) + 4 + i];
			}
			new_image[new_index + i] = max;
		}
	}
}

void pooling(unsigned char *original, unsigned char *modified, int width, int height, int n_thread){

	unsigned int size_image = width * height * 4 * sizeof(unsigned char);

	unsigned char* cuda_image_pool, * cuda_new_image_pool;
	cudaMalloc((void**)& cuda_image_pool, size_image);
	cudaMalloc((void**)& cuda_new_image_pool, size_image);

	
	cudaMemcpy(cuda_image_pool, image, size_image, cudaMemcpyHostToDevice);

	// maximum number of threads we can use is 1 per 16 pixel values
	// that's because we can use maximum 1 thread per 2x2 area, and each pixel in that 2x2 area has 4 values
	if (n_thread > ceil(size_image / 16)) {
		n_thread = ceil(size_image / 16);
	}

	
	num_blocks = ceil((size_image / n_thread) / 16) + 1;
	unsigned int blocks_per_row = ceil(width / n_thread);

	// call method on GPU
	compression <<< num_blocks, n_thread >>> (cuda_image_pool, cuda_new_image_pool, width, size_image, blocks_per_row);
	cudaDeviceSynchronize();

	// CPU copies input data from GPU back to CPU
	unsigned char* new_image_pool = (unsigned char*)malloc(size_image);
	cudaMemcpy(new_image_pool, cuda_new_image_pool, size_image, cudaMemcpyDeviceToHost);
	cudaFree(cuda_image_pool);
	cudaFree(cuda_new_image_pool);

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
	printf("ECSE 420 Lab 0: Simple CUDA Processing - Pooling\n");

	if (argc != 4) {
		printf("RuntimeError: Please use the correct input format: ./ [original png].png [new png].png [n_thread]\n");
		return 1;
	}
	
	if (check_if_png(argv[1]) && check_if_png(argv[2])) {
		unsigned char *original, *modified;
		unsigned error, width, height;

		error = lodepng_decode32_file(&original, &width, &height, argv[1]);

		if (error) {
			printf("ERROR %u: %s\n", error, lodepng_error_text(error));
			free(original);
			return 1;
		} else {
			modified = (unsigned char*)calloc(width*height*4, sizeof(unsigned char));

			pooling(original, modified, width, height, atoi(argv[3]));

			error = lodepng_encode32_file(argv[2], modified, width, height);

			int status = 0;
			if (error) {
				printf("ERROR %u: %s\n", error, lodepng_error_text(error));
				status = 1;
			}
			
			free(original);
			free(modified);
			
			printf("Done\n");
			return status;
		}
	} else {
		printf("RuntimeError: Invalid PNG inputs: %s %s\n", argv[1], argv[2]);
		return 1;
	}
}
