#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gputimer.h"
#define INPUT1_LEN 200001
#define INPUT2_LEN 50000
#define INPUT3_LEN 200000
#define INPUT4_LEN 10000

__global__ void kernel(int* nodePtrs, int* nodeNeightbors, int* nodeStatus, int* currLevelNodes, int* idxCurrLevelNodes, int* outputQueue, int sharedQueueSize)
{
    extern __shared__ int blockQueue[];
    __shared__ int blockCounter[1];
    __shared__ int outputBlockOffset[1];

    // initialize the counter
    if (threadIdx.x==0) 
    {
        *blockCounter = 0;
        *outputBlockOffset = 0;
    }

    __syncthreads();

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

            int bidx = atomicAdd(blockCounter, 1);
            if (bidx >= sharedQueueSize)
            {
                //copy to queue;
                __syncthreads();
                if (threadIdx.x==0)
                {
                    int boffset = atomicAdd(outputBlockOffset, 1);
                    memcpy(outputQueue+boffset*sharedQueueSize, blockQueue, sharedQueueSize*sizeof(int));
                    *blockCounter = 0;
                }
                __syncthreads();
            }
            //save to block mem
            bidx = atomicAdd(blockCounter, 1);
            *(blockQueue+bidx) = nbr;
            
        }
        // increment idx by 1
        idx = atomicAdd(idxCurrLevelNodes, 1);
    }

    __syncthreads();
    if (threadIdx.x==0)
    {
        int boffset = atomicAdd(outputBlockOffset, 0) - 1;
        memcpy(outputQueue+boffset*sharedQueueSize, blockQueue, atomicAdd(blockCounter, 1)*sizeof(int));            
    }
    __syncthreads();
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

void sort(int *pointer, int size){
    //get from https://stackoverflow.com/questions/13012594/sorting-with-pointers-instead-of-indexes
    int *i, *j, temp;
    for(i = pointer; i < pointer + size; i++){
        for(j = i + 1; j < pointer + size; j++){
            if(*j < *i){
                temp = *j;
                *j = *i;
                *i = temp;
            }
        }
    }
}

void compareFiles(char *file_name1, char *file_name2)
{
//get from https://www.tutorialspoint.com/c-program-to-compare-two-files-and-report-mismatches
    FILE* fp1 = fopen(file_name1, "r");
    FILE* fp2 = fopen(file_name2, "r");
    // fetching character of two file
    // in two variable ch1 and ch2
    char ch1 = getc(fp1);
    char ch2 = getc(fp2);

    // error keeps track of number of errors
    // pos keeps track of position of errors
    // line keeps track of error line
    int error = 0, pos = 0, line = 1;

    // iterate loop till end of file
    while (ch1 != EOF && ch2 != EOF)
    {
        pos++;

        // if both variable encounters new
        // line then line variable is incremented
        // and pos variable is set to 0
        if (ch1 == '\n' && ch2 == '\n')
        {
            line++;
            pos = 0;
        }

        // if fetched data is not equal then
        // error is incremented
        if (ch1 != ch2)
        {
            error++;
            printf("Line Number : %d \tError"
                   " Position : %d \n", line, pos);
        }

        // fetching character until end of file
        ch1 = getc(fp1);
        ch2 = getc(fp2);
    }

    printf("Total Errors : %d\t", error);
}

void compareNextLevelNodeFiles(char *file_name1, char *file_name2)
{

    FILE* fp_1 = fopen(file_name1, "r");
    if (fp_1 == NULL){
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    FILE* fp_2 = fopen(file_name2, "r");
    if (fp_2 == NULL){
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    int counter = 0;
    int len_1;
    int len_2;
    int length_file_1 = fscanf(fp_1, "%d", &len_1);
    int length_file_2 = fscanf(fp_2, "%d", &len_2);

    if(length_file_1 != length_file_2){
        fprintf(stderr, "Wrong file length\n");
        exit(1);
    }
    int *input1 = (int *)malloc(len_1 * sizeof(int));
    int *input2 = (int *)malloc(len_2 * sizeof(int));

    int temp1;
    int temp2;

    while ((fscanf(fp_1, "%d", &temp1) == 1) && (fscanf(fp_2, "%d", &temp2) == 1)) {
        (input1)[counter] = temp1;
        (input2)[counter] = temp2;
        counter++;
    }

    sort(input1, len_1);
    sort(input2, len_2);

    for(int i=0; i< len_1; i++){
        if(input1[i] != input2[i]){
            fprintf(stderr, "Something goes wrong\n");
            exit(1);
        }
    }

    fprintf(stderr, "No errors!\n");
    exit(1);

}

int main(int argc, char* argv[])
{
    if (argc == 10)
    {   
        // read parameters
        int numBlock = atoi(argv[1]);
        int blockSize = atoi(argv[2]);
        int sharedQueueSize = atoi(argv[3]);

        // read input 1
        int* data1 = (int* )calloc(INPUT1_LEN, sizeof(int));
        readFile124(argv[4], data1);
        // read input 2
        int* data2 = (int* )calloc(INPUT2_LEN, sizeof(int));
        readFile124(argv[5], data2);
        // read input 3
        int* data3 = (int* )calloc(INPUT3_LEN*4, sizeof(int));
        readFile3(argv[6], data3);
        // read input 1
        int* data4 = (int* )calloc(INPUT4_LEN, sizeof(int));
        readFile124(argv[7], data4);
        // setup idxNextLevelNodes
        int idxCurrLevelNodes = 0;

        // cuda setup
        int *c_data1, *c_data2, *c_data3, *c_data4, *c_outputQueue, *c_idxCurrLevelNodes;
        cudaMalloc((void **) &c_data1, INPUT1_LEN*sizeof(int));
        cudaMalloc((void **) &c_data2, INPUT2_LEN*sizeof(int));
        cudaMalloc((void **) &c_data3, INPUT3_LEN*4*sizeof(int));
        cudaMalloc((void **) &c_data4, INPUT4_LEN*sizeof(int));
        cudaMalloc((void **) &c_outputQueue, INPUT3_LEN*sizeof(int));
        cudaMalloc((void **) &c_idxCurrLevelNodes, sizeof(int));
        cudaMemcpy(c_data1, data1, INPUT1_LEN*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(c_data2, data2, INPUT2_LEN*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(c_data3, data3, INPUT3_LEN*4*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(c_data4, data4, INPUT4_LEN*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(c_idxCurrLevelNodes, &idxCurrLevelNodes, sizeof(int), cudaMemcpyHostToDevice);

        // run
        GpuTimer timer;
        timer.Start();
        kernel <<<numBlock, blockSize, sharedQueueSize*sizeof(int)>>> (c_data1, c_data2, c_data3, c_data4, c_idxCurrLevelNodes, c_outputQueue, sharedQueueSize);
	    cudaDeviceSynchronize();
        timer.Stop();
        printf("Block Queuing Kernel Runtime (blockSize=%d, numBlock=%d, blockQueueCapacity=%d): %f ms\n", numBlock, blockSize, sharedQueueSize, timer.Elapsed());

        // data3 retrival
        cudaMemcpy(data3, c_data3, INPUT3_LEN*4*sizeof(int), cudaMemcpyDeviceToHost);

        // outputs retrival
        int* output = (int *)calloc(INPUT3_LEN, sizeof(int));
        cudaMemcpy(output, c_outputQueue, INPUT3_LEN*sizeof(int), cudaMemcpyDeviceToHost);

        // save gate status
        FILE *gateStatus = fopen(argv[8], "w");
        fprintf(gateStatus, "%d\n", INPUT3_LEN);
        for(int i=0; i<INPUT3_LEN; i++)
        {
            fprintf(gateStatus, "%d\n", *(data3+i*4+3));
        }
        fclose(gateStatus);

        // save output
        FILE *queueStatus = fopen(argv[9], "w");
        fprintf(queueStatus, "%d\n", INPUT3_LEN);
        for(int i=0; i<INPUT3_LEN; i++)
        {
            fprintf(queueStatus, "%d\n", *(output+i));
        }
        fclose(queueStatus);

        compareFiles(argv[8], "sol_nodeOutput.raw");
        compareNextLevelNodeFiles(argv[9], "sol_nextLevelNodes.raw");

        // clean up
        cudaFree(c_data1);
        cudaFree(c_data2);
        cudaFree(c_data3);
        cudaFree(c_data4);
        cudaFree(c_outputQueue);
        cudaFree(c_idxCurrLevelNodes);

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
        printf("RuntimeError: Please follow the correct input format as: /block_queuing <numBlock> <blockSize> <sharedQueueSize> <path_to_input_1.raw> <path_to_input_2.raw> <path_to_input_3.raw> <path_to_input_4.raw> <output_nodeOutput_filepath> <output_nextLevelNodes_filepath> ");
        return 1;
    }
}
