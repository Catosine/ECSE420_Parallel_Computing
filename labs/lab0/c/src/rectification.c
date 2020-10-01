#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>
#include "lodepng.h"

int check_if_png(char *name);

extern void rectificate(char *original, char *modified, int width, int height, int n_thread);

int main(int argc, char *argv[]){
	
	printf("ECSE 420 Lab 0: Simple CUDA Processing - Rectification\n");

	if (argc == 1) {
		printf("Please enter at least one PNG for the rectification process\n");
		return 1;
	}
	if (argc == 2) {
		printf("Please enter the output PNG file name\n");
		return 1;
	}
	int thread = 1;
	if (argc == 3) {
		printf("Warning: No valid thread number input. Then 1 will be used as thread number\n");
	}
	thread = atoi(argv[3]);
	if (argc > 4){
		printf("RuntimeError: Too many parameters\n");
		return 1;
	}

	printf("Processing: %s with %d thread(s)\n", argv[1], thread);
		
	if (check_if_png(argv[1]) && check_if_png(argv[2])) {
		
		unsigned char *original, *modified;
		unsigned error;
		unsigned width, height;

		error = lodepng_decode32_file(&original, &width, &height, argv[1]);

		if (error) {
			printf("ERROR %u: %s\n", error, lodepng_error_text(error));
			free(original);
			return 1;
		} else {
			modified = (unsigned char*)calloc(width*height*4, sizeof(unsigned char));
			
			rectificate(original, modified, width, height, thread);
			
			error = lodepng_encode32_file(argv[2], modified, width, height);

			int status = 0;
			if (error) {
				printf("ERROR %u: %s\n", error, lodepng_error_text(error));
				status = 1;
			}

			free(original);
			free(modified);

			return status;
		}

	} else {
		printf("RuntimeError: Invalid PNG inputs: %s %s\n", argv[1], argv[2]);
		return 1;
	}
	
	printf("Done\n");

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
