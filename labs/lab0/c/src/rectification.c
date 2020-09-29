#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lodepng.h"

int main(int argc, char *argv[]){
	
	printf("ECSE 420 Lab 0: Simple CUDA Processing - Rectification\n");

	if (argc == 1) {
		printf("Please enter at least one PNG for the rectification process\n");
		return 0;
	}

	printf("There are %d PNG files to be processed:\n", argc-1);
	
	char **png_path = (char **)calloc(argc-1, sizeof(char *));
	
	// Read the arguments suggesting the path to PNGs
	for (int i = 0; i<argc-1; i++) {
		//printf("%d: %s\n", count, argv[count]);
		int len = strlen(argv[i+1])+1;
		*png_path = (char *)calloc(len, sizeof(char));
		memcpy(*png_path, argv[i+1], len);
		printf("%d: %s\n", i, *png_path);
		*png_path+=1;
	}

	//wipe out
	for (int i = 0; i<argc-1; i++) {
		*png_path = NULL;
		free(*png_path);
		*png_path-=1;
	}

	free(png_path);

	return 0;
}
