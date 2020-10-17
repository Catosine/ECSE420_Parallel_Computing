#include<stdio.h>
#include<math.h>
#include<reader.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

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

        return output;
    }
}

int main(int argc, char*argv[]){
    if(argc != 3){
        printf("Input should follow this specific pattern \n <input_file_path> <input_file_length> <output_file_path>.\n");
        exit(1);
    }

    int num_lines = argv[2];
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
        printf("result file si saved\n");
    }

    return 0;
}