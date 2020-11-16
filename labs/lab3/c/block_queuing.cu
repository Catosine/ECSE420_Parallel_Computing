#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int read_input_one_two_four(int **input1, const char *filepath)
{
    FILE *fp = fopen(filepath, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    int counter = 0;
    int len;
    if (fscanf(fp, "%d", &len) == 1)
    {
        cudaMallocManaged((int **)input1, len * sizeof(int));
        int temp1;
        while (fscanf(fp, "%d", &temp1) == 1)
        {
            (*input1)[counter] = temp1;
            counter++;
        }
    }

    fclose(fp);
    return len;
}

int read_input_three(int **input1, int **input2, int **input3, int **input4, const char *filepath)
{
    FILE *fp = fopen(filepath, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    int counter = 0;
    int len;
    if (fscanf(fp, "%d", &len) == 1)
    {
        cudaMallocManaged((int **)input1, len * sizeof(int));
        cudaMallocManaged((int **)input2, len * sizeof(int));
        cudaMallocManaged((int **)input3, len * sizeof(int));
        cudaMallocManaged((int **)input4, len * sizeof(int));
        int temp1;
        int temp2;
        int temp3;
        int temp4;
        while (fscanf(fp, "%d,%d,%d,%d", &temp1, &temp2, &temp3, &temp4) == 4)
        {
            (*input1)[counter] = temp1;
            (*input2)[counter] = temp2;
            (*input3)[counter] = temp3;
            (*input4)[counter] = temp4;
            counter++;
        }
    }

    fclose(fp);
    return len;
}

void write_output(int nNodes, int nNextLevelNodes,
                  const int *nodesOutput, const int *nextLevelNodes,
                  const char *output_file_1, const char *output_file_2)
{
    FILE *fp = fopen(output_file_1, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "Couldn't open file for writing\n");
        exit(1);
    }
    fprintf(fp, "%d\n", nNodes);
    for (int i = 0; i < nNodes; ++i)
    {
        fprintf(fp, "%d\n", nodesOutput[i]);
    }
    fclose(fp);

    fp = fopen(output_file_2, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "Couldn't open file for writing\n");
        exit(1);
    }
    fprintf(fp, "%d\n", nNextLevelNodes);
    for (int i = 0; i < nNextLevelNodes; ++i)
    {
        fprintf(fp, "%d\n", nextLevelNodes[i]);
    }
    fclose(fp);
}

__device__ int calcGate(int gate, int inp1, int inp2)
{
    switch (gate)
    {
    case 0:
        return inp1 & inp2;
    case 1:
        return inp1 | inp2;
    case 2:
        return !(inp1 & inp2);
    case 3:
        return !(inp1 | inp2);
    case 4:
        return inp1 ^ inp2;
    case 5:
        return !(inp1 ^ inp2);
    default:
        return 0;
    }
}

__global__ void BFS(int nNodePtrs, int nNodeNeighours, int nNodes, int nCurrLevelNodes, int blockQueCap,
                    const int *nodePtrs, const int *nodeNeighbours, int *nodesVisited, const int *nodesGate,
                    const int *nodesInput, int *nodesOutput, const int *currLevelNodes,
                    int *nextLevelNodes, int *nNextLevelNodes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    if (index >= nCurrLevelNodes)
    {
        return;
    }

    __shared__ int counter[1];
    __shared__ int localNextLevelNodes[64];

    if (tid == 0)
    {
        *counter = 0;
    }

    __syncthreads();

    for (int i = index; i < nCurrLevelNodes; i += stride)
    {
        int node = currLevelNodes[i];
        for (int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; nbrIdx++)
        {
            int neighbour = nodeNeighbours[nbrIdx];
            if (atomicCAS(&nodesVisited[neighbour], 0, 1) == 0)
            {
                nodesOutput[neighbour] = calcGate(nodesGate[neighbour], nodesOutput[node], nodesInput[neighbour]);

                int _counter = atomicAdd(counter, 1);
                if (_counter > blockQueCap - 1)
                {
                    nextLevelNodes[atomicAdd(nNextLevelNodes, 1)] = neighbour;
                }
                else
                {
                    localNextLevelNodes[_counter] = neighbour;
                }
            }
        }
    }

    __syncthreads();

    if (tid == 0)
    {
        memcpy(nextLevelNodes + atomicAdd(nNextLevelNodes, blockQueCap),
               localNextLevelNodes, blockQueCap * sizeof(int));
    }
}

int main(int argc, char *argv[])
{
    if (argc != 10)
    {
        fprintf(stderr, "%s <nBlocks> <blockSz> <blockQueCap> <input1> <input2> <input3> <input4> <output1> <output2>\n", argv[0]);
        return 1;
    }

    const int nBlocks = atoi(argv[1]);
    const int blockSz = atoi(argv[2]);
    const int blockQueCap = atoi(argv[3]);

    const char *input_file_1 = argv[4];
    const char *input_file_2 = argv[5];
    const char *input_file_3 = argv[6];
    const char *input_file_4 = argv[7];

    const char *output_file_1 = argv[8];
    const char *output_file_2 = argv[9];

    int *nodePtrs;
    int nNodePtrs = read_input_one_two_four(&nodePtrs, input_file_1);
    printf("nNodePtrs : %d\n", nNodePtrs);

    int *nodeNeighbours;
    int nNodeNeighbours = read_input_one_two_four(&nodeNeighbours, input_file_2);
    printf("nNodeNeighbours : %d\n", nNodeNeighbours);

    int *nodesVisited;
    int *nodesGate;
    int *nodesInput;
    int *nodesOutput;
    int nNodes = read_input_three(&nodesVisited, &nodesGate, &nodesInput, &nodesOutput, input_file_3);
    printf("nNodes : %d\n", nNodes);

    int *currLevelNodes;
    int nCurrLevelNodes = read_input_one_two_four(&currLevelNodes, input_file_4);
    printf("nCurrLevelNodes : %d\n", nCurrLevelNodes);

    int *nextLevelNodes;
    int *nNextLevelNodes;
    cudaMallocManaged((int **)&nextLevelNodes, nNodePtrs * sizeof(int));
    cudaMallocManaged((int **)&nNextLevelNodes, sizeof(int));
    *nNextLevelNodes = 0;
    BFS<<<nBlocks, blockSz>>>(nNodePtrs, nNodeNeighbours, nNodes, nCurrLevelNodes, blockQueCap,
                              nodePtrs, nodeNeighbours, nodesVisited, nodesGate,
                              nodesInput, nodesOutput, currLevelNodes,
                              nextLevelNodes, nNextLevelNodes);
    cudaDeviceSynchronize();
    printf("nNextLevelNodes : %d\n", *nNextLevelNodes);

    write_output(nNodes, *nNextLevelNodes, nodesOutput, nextLevelNodes, output_file_1, output_file_2);

    cudaFree(nodePtrs);
    cudaFree(nodeNeighbours);
    cudaFree(nodesVisited);
    cudaFree(nodesGate);
    cudaFree(nodesInput);
    cudaFree(nodesOutput);
    cudaFree(currLevelNodes);
    cudaFree(nextLevelNodes);
    cudaFree(nNextLevelNodes);
}
