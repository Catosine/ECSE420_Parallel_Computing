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
/* this flie takes three parameters:
 *      char* file: path to the txt file formatted in csv
 *      int** output: pointer stores data in the csv file
 *      int* line: pointer to line count
 */
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
        //printf("GOOD TILL HERE\n");
	FILE* f = fopen(file, "r");
        //printf("Still good\n");
	char *tmp = (char *)calloc(10, sizeof(char));
        char *token;
        if (f)
        {
		//printf("test\n");
		while(fgets(tmp, 10, f))
            	{
			//printf("line=%d\n",*line);
			//printf("boom\n");
		        *(output+*line) = (int *)calloc(3, sizeof(int));
			token = strtok(tmp, ",");
                	//printf("line=%d\n",*line);
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
	//printf("error\n");
        tmp = NULL;
        free(tmp);

        return 0;
    }
}

int read_csv_array(char* file, int* output, int *line)
{
	if (csv_check(file))
	{
		printf("RuntimeError: Cannot access the file at %s\n", file);
		return -1;
	}
	else
	{
		*line = 0;
		FILE* f = fopen(file, "r");
		char *tmp = (char *)calloc(10, sizeof(char));
		char *token;
		if (f) 
		{
			while(fgets(tmp,10,f))
			{
				token = strtok(tmp, ",");
				for(int i = 0; i<3; i++)
				{
					*(output + *line * 3 + i) = atoi(token);
					token = strtok(NULL, ",");
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

// an example of how to use this read_csv and save
/*
int main(){
 	char *test="../res/input_10000.txt";
 	int line = 0;
 	int **output = (int **)calloc(1000000, sizeof(int *));
 	read_csv(test, output, &line);
 	for(int i=0; i<line; i++){
 		printf("line %d: input1: %d, linput2: %d, type: %d\n", i, **(output+i), *(*(output+i)+1), *(*(output+i)+2));
     }
 	printf("Total: %d\n", line);
 	output = NULL;
 	free(output);

	int result[3] = {1,2,3};
      	save("./test.txt", result, 3);

 	return 0;
}
*/
