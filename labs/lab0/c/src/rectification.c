#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>
#include "lodepng.h"

int check_if_png(char *name);
int process(char *input, char* output, int n_thread);

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
		process(argv[1], argv[2], thread);
	} else {
		printf("RuntimeError: Invalid PNG inputs: %s %s\n", argv[1], argv[2]);
		return 1;
	}
	
	printf("Done\n");

	return 0;
}

int process(char *input, char *output, int n_thread){
	unsigned error;
	unsigned char *image, *new_image;
	unsigned width, height;
	error = lodepng_decode32_file(&image, &width, &height, input);
	if (error) {
		printf("ERROR %u: %s\n", error, lodepng_error_text(error));
		return 1;
	} else {
		new_image = (unsigned char*)calloc(width*height*4, sizeof(unsigned char));
		
		// process image
		/*
		unsigned char value;
		for (int i = 0; i < height; i++){
			for (int j = 0; j<height; j++){
				//value = image[4*width*i+4*j];
				new_image[4*width*i+4*j+0] = image[4*width*i+4*j+0];//R
				new_image[4*width*i+4*j+1] = image[4*width*i+4*j+1];//G
				new_image[4*width*i+4*j+2] = image[4*width*i+4*j+2];//B
				new_image[4*width*i+4*j+3] = image[4*width*i+4*j+3];//Alpha
			}
		}
		*/
		
		lodepng_encode32_file(output, new_image, width, height);

		free(image);
		free(new_image);

		return 0;
	}
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
