#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>
#include "lodepng.h"

int check_if_png(char *name);

int main(int argc, char *argv[]){
	
	printf("ECSE 420 Lab 0: Simple CUDA Processing - Rectification\n");

	if (argc == 1) {
		printf("Please enter at least one PNG for the rectification process\n");
		return 0;
	}

	printf("There are %d PNG files to be processed:\n", argc-1);
	
	// Read the arguments suggesting the path to PNGs
	for (int i = 0; i<argc-1; i++) {
		//printf("%d: %s\n", count, argv[count]);
		printf("%d: %s\n", i, argv[i+1]);
		
		if (check_if_png(argv[i+1])) {
			//TODO: add read file

			//TODO: add rectification

			//TODO: add save file
		} else {
			printf("Invalid PNG file: %s\n", argv[i+1]);
		}
	}

	return 0;
}

int check_if_png(char *name){
	regex_t reg;
	const char *pattern = "^.*\\.(png)?$";
	regcomp(&reg, pattern, REG_EXTENDED);
	const size_t nmatch = 1;
	regmatch_t pmatch[1];
	int status = regexec(&reg, name, nmatch, pmatch, 0);
	regfree(&reg);

	if (status == REG_NOMATCH){
		return 0;
	} else if (status == 0){
		return 1;
	}
}
