#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

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

int read_csv(char* file, char** output, int *line)
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
        output = (char **)calloc(1000000, sizeof(char *));
        *line = 0;

        // read the csv file
        FILE* f = fopen(file, "r");
        char *tmp = (char *)calloc(10, sizeof(char));
        if (f)
        {
            while(fgets(tmp, 10, f))
            {
		*(output+*line) = strtok(tmp, ",");
		*line = *line + 1;	
            }
        }
        fclose(f);
        tmp = NULL;
        free(tmp);
        return 0;
    }
}

int main(){
	char *test="../../res/input_10000.txt";
	int line = 0;
	char **output;
	read_csv(test, output, &line);
	
	for(int i=0; i<line; i++){
		printf("input1=%d input2=%d, gate=%d\n", atoi(*(output+i)), atoi(*(output+i)+1), atoi(*(output+i)+2));
	}
	printf("Total: %d\n", line);
	output = NULL;
	free(output);
	return 0;
}
