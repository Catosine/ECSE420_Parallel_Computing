#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <regex.h>
#include "lodepng.h"



__global__ void pool_kernel(unsigned char* image, unsigned char* new_image, unsigned width, unsigned height, int iter, int thread_define)
{
	//indexing
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//input number of threads
	int num_thread = thread_define;

	unsigned offset;
	unsigned char top_left, top_right, bottom_left, bottom_right, max_value;

	
	if (index < thread_define) {
		for (int k = 0; k < 4; k++) {
			offset = iter * num_thread * 2 + index * 2;
			offset += width * (offset / width);

			//topleft, topright, bottomleft, bottomright
			top_left = image[(offset) * 4 + k];
			top_right = image[(offset + 1) * 4 + k];
			bottom_left = image[(offset + width) * 4 + k];
			bottom_right = image[(offset + width + 1) * 4 + k];
			
			max_value = 0;
			
			//find the max value 
			if (top_left > max_value) max_value = top_left;
			if (top_right > max_value) max_value = top_right;
			if (bottom_left > max_value) max_value = bottom_left;
			if (bottom_right > max_value) max_value = bottom_right;

			new_image[(iter * num_thread + index) * 4 + k] = max_value;
		}
	}
}

void pool(unsigned char *original, unsigned char *modified, int width, int height, int n_thread){
	size_t png_size = width*height*4*sizeof(unsigned char);
		
	unsigned char* cuda_image;
	cudaMalloc((void**) &cuda_image, png_size);
	cudaMemcpy(cuda_image, original, png_size, cudaMemcpyHostToDevice);

	unsigned char* cuda_new_image;
	cudaMalloc((void**) &cuda_new_image, png_size);

	//top_rightack how many iter we need 
	int iter = 0;

	//number of block 
  	int num_blocks = 1;
	int num_threads = n_thread;
  
  	//int thread;
 	if (num_threads > 1024) {
    	while (num_threads > 1024) {
        	num_blocks *= 2; 
        	num_threads /= 2;
        	if (num_threads <= 1024) {
              	break; 
            }
    	} 
  	}

	while (iter < width * height / n_thread / 4) {
		//maxmimum number of threads per block is 1024
		pool_kernel << <(n_thread + 1023) / 1024, 1 >> > (cuda_image, cuda_new_image, width, height, iter, n_thread);
		iter++;
	}
	pool_kernel << <(n_thread + 1023) / 1024, (height * width) % 1024 >> > (cuda_image, cuda_new_image, width, height, iter, n_thread);

	cudaDeviceSynchronize();

	modified = (unsigned char*)calloc(1, png_size);
	cudaMemcpy(modified, cuda_new_image, png_size, cudaMemcpyDeviceToHost);
		
	cudaFree(cuda_image);
	cudaFree(cuda_new_image);
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
		printf("RuntimeError: Please use the correct input format: ./pooling original.png rectficated.png [n_thread]\n");
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
			
			time_t start = clock();
			pool(original, modified, width, height, atoi(argv[3]));
			time_t end = clock();

			error = lodepng_encode32_file(argv[2], modified, width/2, height/2);

			int status = 0;
			if (error) {
				printf("ERROR %u: %s\n", error, lodepng_error_text(error));
				status = 1;
			}
			
			free(original);
			free(modified);
			
			printf("Total time: %f\nDone\n", (double)(end-start)/(double)CLOCKS_PER_SEC);
			return status;
		}
	} else {
		printf("RuntimeError: Invalid PNG inputs: %s %s\n", argv[1], argv[2]);
		return 1;
	}
}
