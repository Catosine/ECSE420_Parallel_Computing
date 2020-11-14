#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define INPUT1_LEN 200001
#define INPUT2_LEN 50000
#define INPUT3_LEN 200000
#define INPUT4_LEN 10000

int numNextLevelNodes = 0;

int gate_kernel(int gate, int input1, int input2)
{
    //AND
    if (gate == 0) {
        return input1 & input2;
    }
        //OR
    else if (gate == 1) {
        return (input1 | input2);
    }
        //NAND
    else if (gate == 2) {
        return !(input1 & input2);
    }
        //NOR
    else if (gate == 3) {
        return !(input1 | input2);
    }
        //XOR
    else if (gate == 4) {
        return input1 ^ input2;
    }
        //XNOR
    else if (gate == 5) {
        return !(input1 ^ input2);
    }
    else {
        printf("The input logic gate is invalid");
        exit(-1);
    }

}

void sequential(int* nodePtrs, int* nodeNeighbors, int** data3, int* numCurrLevelNodes, int* nextLevelNodes) {
    // Loop over all nodes in the current level
    for (int idx = 0; idx<INPUT4_LEN; idx++) {
        int node = numCurrLevelNodes[idx];
        // Loop over all neighbors of the node
        for (int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; nbrIdx++) {
            int neighbor = nodeNeighbors[nbrIdx];

            // If the neighbor hasn't been visited yet
            if (data3[neighbor][0] == 0) {
                // Mark it and add it to the queue
                data3[neighbor][0] = 1;
                data3[neighbor][3] = gate_kernel(data3[neighbor][1], data3[node][3], data3[neighbor][2]);

                nextLevelNodes[numNextLevelNodes++] = neighbor;
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
    if (argc == 7) {
        FILE* input1 = fopen(argv[1], "r");
        int* data1 = (int*)malloc(INPUT1_LEN*sizeof(int));
        if (input1 != NULL) {
            char line[10];
            fgets(line, 10, input1);
            for (int i = 0; i < INPUT1_LEN; i++) {
                fgets(line, 10, input1);
                data1[i] = atoi(line);
            }
        }

        FILE* input2 = fopen(argv[2], "r");
        int* data2 = (int*)malloc(INPUT2_LEN*sizeof(int));
        if (input2 != NULL) {
            char line[10];
            fgets(line, 10, input2);
            for (int i = 0; i < INPUT2_LEN; i++) {
                fgets(line, 10, input2);
                data2[i] = atoi(line);
            }
        }

        int** data3 = (int**) malloc(INPUT3_LEN * sizeof(int*));
        FILE* input3 = fopen(argv[3], "r");
        if (input3 != NULL) {
            char line[10];
            fgets(line, 10, input3);
            for (int i = 0; i < INPUT3_LEN; i++) {
                fgets(line, sizeof(line), input3);
                data3[i] = (int*)malloc(4*sizeof(int));
                data3[i][0] = line[0] - '0';
                data3[i][1] = line[2] - '0';
                data3[i][2] = line[4] - '0';

                if (line[6] == '-') {
                    data3[i][3] = -1;
                } else if (line[6] == '0') {
                    data3[i][3] = 0;
                } else if (line[6] == '1') {
                    data3[i][3] = 1;
                }
            }
        }

        FILE* input4 = fopen(argv[4], "r");
        int* data4 = (int*)malloc(INPUT4_LEN*sizeof(int));
        if (input4!= NULL) {
            char line[10];
            fgets(line, 10, input4);
            for (int i = 0; i < INPUT4_LEN; i++) {
                fgets(line, 10, input4);
                data4[i] = atoi(line);
            }
        }
        fclose(input1);
        fclose(input2);
        fclose(input3);
        fclose(input4);

        int* output_nextNode = (int*) malloc(INPUT3_LEN * sizeof(int));
        sequential(data1, data2, data3, data4, output_nextNode);

        FILE* output1 = fopen(argv[5], "w");
        fprintf (output1, "%d\n", INPUT3_LEN);
        for (int i=0; i<INPUT3_LEN; i++) {
            fprintf (output1, "%d\n", data3[i][3]);
        }
        fclose(output1);

        FILE* output2 = fopen(argv[6], "w");
        fprintf (output2, "%d\n", numNextLevelNodes);
        for (int i=0; i<numNextLevelNodes; i++) {
            fprintf (output2, "%d\n", output_nextNode[i]);
        }
        fclose(output2);


        compareFiles(argv[5], "sol_nodeOutput.raw");
        compareNextLevelNodeFiles(argv[6], "sol_nextLevelNodes.raw");

        free(data1);
        free(data2);
        free(data4);

        for (int i=0; i<INPUT3_LEN; i++) {
            free(data3[i]);
        }
        free(data3);

        return 0;
    }
    else {
        printf("Incorrect format!\n");
        printf("./sequential <path_to_input_1.raw> <path_to_input_2.raw> <path_to_input_3.raw> <path_to_input_4.raw> <output_nodeOutput_filepath> <output_nextLevelNodes_filepath>\n");
        return -1;
    }
}



