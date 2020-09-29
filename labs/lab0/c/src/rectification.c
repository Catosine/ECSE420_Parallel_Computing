#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>

#include "lodepng.h"

int check_if_png(char *name);
int rectificate(char *input, char* output);

int main(int argc, char *argv[]){
	
	printf("ECSE 420 Lab 0: Simple CUDA Processing - Rectification\n");

	if (argc == 1) {
		printf("Please enter at least one PNG for the rectification process\n");
		return 0;
	}
	if (argc >2){
		printf("Warning: ONLY the first PNG file will be processed\n");
	}

	printf("Processing: %s\n", argv[1]);
		
	if (check_if_png(argv[1])) {
		//TODO: modify the output name based on input name
		rectificate(argv[1], "rectification_output.png");
		//TODO: add read file

		//TODO: add rectification

		//TODO: add save file
	} else {
		printf("Invalid PNG file: %s\n", argv[1]);
		return 1;
	}
	
	printf("Done\n");

	return 0;
}

int rectificate(char *input, char *output){
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
