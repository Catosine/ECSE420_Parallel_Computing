#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define INPUT1_LEN 200001
#define INPUT2_LEN 50000
#define INPUT3_LEN 200000
#define INPUT4_LEN 10000
#define N_BLOCK 32
#define N_THREAD 10

__global__ void kernel(int* nodePtrs, int* nodeNeightbors, int* nodeStatus, int* currLevelNodes, int* idxCurrLevelNodes, int* outputQueue, int* idxOutputQueue)
{
    int idx = atomicAdd(idxCurrLevelNodes, 1);
    while(idx < INPUT4_LEN)
    {
        // get node
        int node = *(currLevelNodes+idx);
        int nbr_idx = *(nodePtrs+node);
        int nbr_end = *(nodePtrs+node+1);
        for(; nbr_idx<=nbr_end; nbr_idx++)
	{   
                int nbr = *(nodeNeightbors+nbr_idx);
                // check if visited
                if(*(nodeStatus+nbr*4)==0)
                {
		    	atomicAdd(nodeStatus+nbr*4, 1);
                    	int gate = *(nodeStatus+nbr*4+1);
                    	int input1 = *(nodeStatus+nbr*4+2);
                    	int input2 = *(nodeStatus+node*4+3);
                    	int result = -3;
                    	//AND
                    	if (gate == 0) {result = (input1 & input2);}
                    	//OR
                    	else if (gate == 1) {result = (input1 | input2);}
                    	//NAND
                    	else if (gate == 2) {result = !(input1 & input2);}
                    	//NOR
                    	else if (gate == 3) {result = !(input1 | input2);}
                    	//XOR
                    	else if (gate == 4) {result = (input1 ^ input2);}
                    	//XNOR
                    	else if (gate == 5) {result = !(input1 ^ input2);}
                    	else {result = -1;}
                    	*(nodeStatus+nbr*4+3) = result;
                    	int oidx = atomicAdd(idxOutputQueue, 1);
                    	*(outputQueue+oidx) = nbr;
        	}
        }
        // increment idx by 1
        idx = atomicAdd(idxCurrLevelNodes, 1);
    }
}

int readFile124(char* name, int* data)
{
    FILE* f = fopen(name, "r");
    int offset = 0;
    char* line = (char* )calloc(10, sizeof(char));
    if (f)
    {   
        // omit the first
        fgets(line, 10, f);
        while(fgets(line, 10, f))
        {
            *(data+offset) = atoi(line);
            offset++;
        }
    }
    free(line);
    line = NULL;
    fclose(f);
    return 0;
}

int readFile3(char* name, int* data)
{
    FILE* f = fopen(name, "r");
    int offset = 0;
    char* line = (char *)calloc(10, sizeof(char));
    if (f) 
    {
        // omit the first
        fgets(line, 10, f);
        while(fgets(line, 10, f))
        {
            *(data+offset+0) = *(line+0) - '0';
            *(data+offset+1) = *(line+2) - '0';
            *(data+offset+2) = *(line+4) - '0';
            if (*(line+6) == '-') 
            {
                *(data+offset+3) = -1;
            }
            else if (*(line+6) == '0') 
            {
                *(data+offset+3) = 0;
            }
            else if (*(line+6) == '1') 
            {
                *(data+offset+3) = 1;
            }
            offset+=4;
        }
    }
    free(line);
    line = NULL;
    fclose(f);
    return 0;
}

int main(int argc, char* argv[])
{
    if (argc == 7)
    {
        // read input 1
        int* data1 = (int* )calloc(INPUT1_LEN, sizeof(int));
        readFile124(argv[1], data1);
        // read input 2
        int* data2 = (int* )calloc(INPUT2_LEN, sizeof(int));
        readFile124(argv[2], data2);
        // read input 3
        int* data3 = (int* )calloc(INPUT3_LEN*4, sizeof(int));
        readFile3(argv[3], data3);
        // read input 1
        int* data4 = (int* )calloc(INPUT4_LEN, sizeof(int));
        readFile124(argv[4], data4);
        // setup idxBfsQueue
        int idxOutputQueue = 0;
        // setup idxNextLevelNodes
        int idxCurrLevelNodes = 0;

        // cuda setup
        int *c_data1, *c_data2, *c_data3, *c_data4, *c_outputQueue, *c_idxCurrLevelNodes, *c_idxOutputQueue;
        cudaMalloc((void **) &c_data1, INPUT1_LEN*sizeof(int));
        cudaMalloc((void **) &c_data2, INPUT2_LEN*sizeof(int));
        cudaMalloc((void **) &c_data3, INPUT3_LEN*4*sizeof(int));
        cudaMalloc((void **) &c_data4, INPUT4_LEN*sizeof(int));
        cudaMalloc((void **) &c_outputQueue, INPUT3_LEN*sizeof(int));
        cudaMalloc((void **) &c_idxCurrLevelNodes, sizeof(int));
        cudaMalloc((void **) &c_idxOutputQueue, sizeof(int));
        cudaMemcpy(c_data1, data1, INPUT1_LEN*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(c_data2, data2, INPUT2_LEN*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(c_data3, data3, INPUT3_LEN*4*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(c_data4, data4, INPUT4_LEN*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(c_idxCurrLevelNodes, &idxCurrLevelNodes, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(c_idxOutputQueue, &idxOutputQueue, sizeof(int), cudaMemcpyHostToDevice);

        // run
        kernel <<<N_BLOCK, N_THREAD>>> (c_data1, c_data2, c_data3, c_data4, c_idxCurrLevelNodes, c_outputQueue, c_idxOutputQueue);
	cudaDeviceSynchronize();

        // data3 retrival
        cudaMemcpy(data3, c_data3, INPUT3_LEN*4*sizeof(int), cudaMemcpyDeviceToHost);

        // outputs retrival
        cudaMemcpy(&idxOutputQueue, c_idxOutputQueue, sizeof(int), cudaMemcpyDeviceToHost);
        int* output = (int *)calloc(INPUT3_LEN, sizeof(int));
        cudaMemcpy(output, c_outputQueue, INPUT3_LEN*sizeof(int), cudaMemcpyDeviceToHost);

        // save gate status
        FILE *gateStatus = fopen(argv[5], "w");
        fprintf(gateStatus, "%d\n", INPUT3_LEN);
        for(int i=0; i<INPUT3_LEN; i++)
        {
            fprintf(gateStatus, "%d\n", *(data3+i*4+3));
        }
        fclose(gateStatus);

        // save output
        FILE *queueStatus = fopen(argv[6], "w");
        fprintf(queueStatus, "%d\n", idxOutputQueue);
        for(int i=0; i<idxOutputQueue; i++)
        {
            fprintf(queueStatus, "%d\n", *(output+i));
        }
        fclose(queueStatus);

        // clean up
        cudaFree(c_data1);
        cudaFree(c_data2);
        cudaFree(c_data3);
        cudaFree(c_data4);
        cudaFree(c_outputQueue);
        cudaFree(c_idxCurrLevelNodes);
        cudaFree(c_idxOutputQueue);

        free(data1);
        free(data2);
        free(data3);
        free(data4);
        free(output);

        data1 = NULL;
        data2 = NULL;
        data3 = NULL;
        data4 = NULL;
        output = NULL;
    
    }
    else
    {
        printf("RuntimeError: Please follow the correct input format as: /global_queuing <path_to_input_1.raw> <path_to_input_2.raw> <path_to_input_3.raw> <path_to_input_4.raw> <output_nodeOutput_filepath> <output_nextLevelNodes_filepath> ");
        return 1;
    }
}
