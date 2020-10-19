#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../inc/reader.h"
#include <unistd.h>
#include <string.h>
#include <time.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5


int csv_check(char* path)
{
    if (access(path, F_OK))
    {   
        printf("Cannot find the file at %s\n", path);
        return -1;
    } 
    if (access(path, R_OK)) 
    {
        printf("Cannot read the file at %s\n", path);
        return -1;
    }
    return 0;
}

int read_csv(char* file, int** output, int *line)
{
    if (csv_check(file))
    {
        //error detected
        printf("RuntimeError: Cannot access the file at %s\n", file);
        return -1;
    } 
    else 
    {
        // check passed
        // allocate memory
        *line = 0;

        // read the csv file
        FILE* f = fopen(file, "r");
        char *tmp = (char *)calloc(10, sizeof(char));
        char *token;
        if (f)
        {
            while(fgets(tmp, 10, f))
            {
		        *(output+*line) = (int *)calloc(3, sizeof(int));
                token = strtok(tmp, ",");
                int i = 0;
                while(token!=NULL)
                {
                    *(*(output+*line)+i) = atoi(token);
                    token = strtok(NULL, ",");
                    i++;
                }
		        *line = *line + 1;	
            }
        }
        fclose(f);

        tmp = NULL;
        free(tmp);

        return 0;
    }
}

int save(char* file, int* results, int size)
{
    FILE *f = fopen(file, "w");
    if (f)
    {
        for(int i=0; i<size; i++)
        {
            fprintf(f, "%d\n", *(results+i));
        }
    }
    fclose(f);
    return 0;
}

int logicGate(int input1, int input2, int operand){
    int output;
    switch(operand){
        case 0:
            if(input1==0 && input2 ==0) output = 0;
            if(input1==0 && input2 ==1) output = 0;
            if(input1==1 && input2 ==0) output = 0;
            if(input1==1 && input2 ==1) output = 1;
            break;
        case 1:
            if(input1==0 && input2 ==0) output = 0;
            if(input1==0 && input2 ==1) output = 1;
            if(input1==1 && input2 ==0) output = 1;
            if(input1==1 && input2 ==1) output = 1;
            break;
        case 2:
            if(input1==0 && input2 ==0) output = 1;
            if(input1==0 && input2 ==1) output = 1;
            if(input1==1 && input2 ==0) output = 1;
            if(input1==1 && input2 ==1) output = 0;
            break;
        case 3:
            if(input1==0 && input2 ==0) output = 1;
            if(input1==0 && input2 ==1) output = 0;
            if(input1==1 && input2 ==0) output = 0;
            if(input1==1 && input2 ==1) output = 0;
            break;
        case 4:
            if(input1==0 && input2 ==0) output = 0;
            if(input1==0 && input2 ==1) output = 1;
            if(input1==1 && input2 ==0) output = 1;
            if(input1==1 && input2 ==1) output = 0;
            break;
        case 5:
            if(input1==0 && input2 ==0) output = 1;
            if(input1==0 && input2 ==1) output = 0;
            if(input1==1 && input2 ==0) output = 0;
            if(input1==1 && input2 ==1) output = 1;
            break;
    }
    return output;
}

int main(int argc, char*argv[]){
    if(argc != 4){
        printf("Input should follow this specific pattern \n <input_file_path> <input_file_length> <output_file_path>.\n");
        exit(1);
    }

    clock_t begin = clock();

    int num_lines = atoi(argv[2]);
    int **sourceInput = (int **)calloc(num_lines, sizeof(int *));
    int result[num_lines];

    read_csv(argv[1], sourceInput, &num_lines);

    for(int i=0; i<num_lines; i++){
        result[i] = logicGate(**(sourceInput+i), *(*(sourceInput+i)+1), *(*(sourceInput+i)+2));
    }

    free(sourceInput);

    switch(num_lines){
        case 10000:
            save("../../res/result_10000.txt", result, num_lines);
            break;
        case 100000:
            save("../../res/result_100000.txt", result, num_lines);
            break;
        case 1000000:
            save("../../res/result_1000000.txt", result, num_lines);
            break;
        printf("result file is saved\n");
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("The total runtime is %f.\n", time_spent);

    EXIT_SUCCESS;
}