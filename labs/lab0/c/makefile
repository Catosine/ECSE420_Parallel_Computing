dir_inc='./inc'
dir_src='./src'
dir_bin='./bin'
dir_cuda='/usr/local/cuda-10.2/lib64'

test_equality: bin lodepng_c
	gcc -I ${dir_inc} ./bin/lodepng.o ./src/test_equality.c -o test_equality -lm

lodepng_c: bin
	gcc -I ${dir_inc} ${dir_src}/lodepng.c -c
	mv ./lodepng.o ${dir_bin}

lodepng_cu: bin
	nvcc -I ${dir_inc} ${dir_src}/lodepng.cu -c
	mv ./lodepng.o ${dir_bin}

rectify: lodepng_cu
	nvcc -I ${dir_inc} ${dir_bin}/lodepng.o ${dir_src}/rectification.cu -o rectify

loadpng_test: bin lodepng_c
	gcc -I ${dir_inc} ./bin/lodepng.o ./src/lodepng_test.c -o loadpng_test

bin:
	mkdir ./bin

clean:
	rm ./bin/*
