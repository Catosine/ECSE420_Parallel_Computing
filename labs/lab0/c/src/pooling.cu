#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void pooling(char* input_filename, char* output_filename, int thread_define)
{
	unsigned error;
	unsigned char* image, * new_image;
	unsigned width, height;


	//lode input file to image
	error = lodepng_decode32_file(&image, &width, &height, input_filename); 

	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	//malloc a new image
	new_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));  

	//top_rightact start and end time 
	clock_t start_time, end_time; 

	start_time = clock();
	
	// parallel way of pooling
	cudaSetDevice(0);

	//move to GPU
	unsigned char* image_cp;
	cudaMallocManaged((void**)&image_cp, 4 * width * height * sizeof(unsigned char));
	cudaMallocManaged((void**)&new_image, width * height * sizeof(unsigned char));


	// copy image into image_cp
	for (int i = 0; i < width * height * 4; i++) {
		image_cp[i] = image[i];
	}

	//initialize new_image to 0 
	for (int i = 0; i < width * height; i++) {
		new_image[i] = 0;
	}


	//top_rightack how many iter we need 
	int iter = 0;

  //number of block 
  int num_blocks = 1;
  int num_threads = thread_define;
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

	while (iter < width * height / thread_define / 4) {
		//maxmimum number of threads per block is 1024
		pool_kernel << <(thread_define + 1023) / 1024, 1 >> > (image_cp, new_image, width, height, iter, thread_define);
		iter++;
	}
		pool_kernel << <(thread_define + 1023) / 1024, (height * width) % 1024 >> > (image_cp, new_image, width, height, iter, thread_define);

	cudaDeviceSynchronize();

	end_time = clock();

	printf("time=%f\n", (double)(end_time - start_time) / (double)CLOCKS_PER_SEC);

	////output the image
	lodepng_encode32_file(output_filename, new_image, width / 2, height / 2); 
	cudaFree(image); cudaFree(new_image); cudaFree(image_cp);
	free(image);

	//free(image_cp);
	//free(new_image);
	
}

int main(int argc, char* argv[])
{
	char* input_filename = argv[1];
	char* output_filename_pooling = argv[2];
  int thread_define = atoi(argv[3]);
	//char* input_filename = "Test_1.png";
	//char* output_filename_pooling = "Test_1_output.png";
	
	pooling(input_filename, output_filename_pooling, thread_define);

	return 0;
}