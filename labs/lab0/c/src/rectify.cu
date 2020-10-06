#include "gputimer.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lodepng.h"



__global__ void rectify(unsigned char * image_input, unsigned char* image_output, unsigned int thread_num, unsigned int size) {
  int thread_start =  threadIdx.x;
  for (int i = thread_start; i < size; i += thread_num) {
		if (image_input[i] < 127)
			image_output[i] = 127;
		else
			image_output[i] = image_input[i];
	}
}



int main(int argc, char *argv[])
{
  if (argc != 4) {
		printf("Command should follow the format ./rectify <name of input png> <name of output png> < # threads>\n");
		return -1;
	}

 //initiate timer and parse the parameters
  GpuTimer timer;
  char* input_file = argv[1];
  char* output_file = argv[2];
  unsigned int thread_num = atoi(argv[3]);
 
 if (thread_num > 1024) {
		printf("Warning: number of thread should not be greater than 1024 \n");
	}

  // initiate host data
	unsigned error;
 	unsigned width, height;
	unsigned char* image, * new_image;
 	unsigned char* image_input;
  unsigned char* image_output;

 

	error = lodepng_decode32_file(&image, &width, &height, input_file);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
  
  unsigned int imagesize = width * height * 4 * sizeof(unsigned char);
 	new_image = (unsigned char*) malloc(imagesize);
  cudaMalloc((void**)&image_input, imagesize);
  cudaMalloc((void**)&image_output, imagesize);

 
  //transfer  data to device memory
  cudaMemcpy(image_input, image, imagesize, cudaMemcpyHostToDevice);
 
 //start the gpu timer
  timer.Start();
  rectify <<<1, thread_num >>> (image_input, image_output, thread_num, imagesize);
 

  timer.Stop();
	cudaDeviceSynchronize();

  //obtain the runtime from timer
  float timeTaken = timer.Elapsed();
  printf("Rectification with %d threads, runtime :  %f milliseconds \n", thread_num, timeTaken); 
  cudaMemcpy(new_image, image_output, imagesize, cudaMemcpyDeviceToHost);

  //output the new_image
	lodepng_encode32_file(output_file, new_image, width, height);
  cudaFree(new_image);
  cudaFree(image_input);
  cudaFree(image_output);
 
  printf("Done！");
	return 0;  
}
