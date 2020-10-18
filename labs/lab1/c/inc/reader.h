/* this flie takes three parameters:
 *      char* file: path to the txt file formatted in csv
 *      int** output: pointer stores data in the csv file
 *      int* line: pointer to line count
 */
int read_csv(char* file, int** output, int *line);

int read_csv_array(char* file, int* output, int* line);

/* this flie takes three parameters:
 *      char* file: path to the txt file formatted in csv
 *      int* results: array of int results
 *      int size: size of results array
 */
int save(char* file, int* results, int size);
