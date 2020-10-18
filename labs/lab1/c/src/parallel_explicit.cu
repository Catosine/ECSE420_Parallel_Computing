#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "reader.h"
#include <stdlib.h>
#include "gputimer.h"

__global__ void kernel(int* data, int* result, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		if (*(data+idx*3+2)==0)
		{
			//and
			*(result+idx) = *(data+idx*3) && *(data+idx*3+1);
		} 
		else if (*(data+idx*3+2)==1)
		{
			//or
			*(result+idx) = *(data+idx*3) || *(data+idx*3+1);
		}
		else if (*(data+idx*3+2)==2)
		{
			//nand
			*(result+idx) = !(*(data+idx*3) && *(data+idx*3+1));
		}
		else if (*(data+idx*3+2)==3)
		{
			//nor
			*(result+idx) = !(*(data+idx*3) || *(data+idx*3+1));
		}
		else if (*(data+idx*3+2)==4)
		{
			//xor
			*(result+idx) = *(data+idx*3) ^ *(data+idx*3+1);
		}
		else if (*(data+idx*3+2)==5)
		{
			//xnor
			*(result+idx) = !(*(data+idx*3) ^ *(data+idx*3+1));
		}
		else
		{
			//invalid
			*(result+idx) = -1;
		}
		//*(result+idx)=idx;
	}
}

int main(int argc, char *argv[])
{
	GpuTimer timer;
	timer.Start();
	printf("ECSE 420 Lab 1: Logic Gates Simulation - parallel_explicit\n");
	if (argc != 4)
	{
		printf("RuntimeError: Wrong inputs.\n");
		printf("You should input follows this format: ./parallel_explicit <input_file_path> <input_file_length> <output_file_path>\n");
		return 1;
	}
	
	int size = atoi(argv[2]);

	GpuTimer loadTimer;
	loadTimer.Start();
	int* file = (int *)calloc(size*3, sizeof(int));
	
	if(read_csv_array(argv[1], file, &size)==0){
		
		int *cuda_file;	
		cudaMalloc((void **) &cuda_file, size*3*sizeof(int));
		cudaMemcpy(cuda_file, file, size*3*sizeof(int), cudaMemcpyHostToDevice);

		int *cuda_output;
		cudaMalloc((void **) &cuda_output, size*sizeof(int));
		loadTimer.Stop();

		float laodTime = loadTimer.Elapsed();

		int block = size/1024;
		if (size%1024) {
			block++;
		}

		kernel <<<block, 1024>>> (cuda_file, cuda_output, size);
		cudaDeviceSynchronize();
		
		GpuTimer retrivetimer;
		retrivetimer.Start();
		
		int *output = (int *)calloc(size, sizeof(int));
		cudaMemcpy(output, cuda_output, size*sizeof(int), cudaMemcpyDeviceToHost);
		retrivetimer.Stop();
		float retriveTime = retrivetimer.Elapsed();

		save(argv[3], output, size);

		cudaFree(cuda_file);
		cudaFree(cuda_output);

		file = NULL;
		free(file);

		output = NULL;
		free(output);
		
		timer.Stop();

		float totalTime = timer.Elapsed();

		printf("Done\n");
		printf("Load Time: %f ms\nRetrive Time: %f ms\nTotal Time: %f ms\n", laodTime, retriveTime, totalTime);

		return 0;
	}
	else
	{
		return 1;
	}
	
}
