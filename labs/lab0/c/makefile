dir_inc='./inc'
dir_src='./src'
dir_bin='./bin'
dir_cuda='/usr/local/cuda-10.2/lib64'

test_equality: bin lodepng
	gcc -I ${dir_inc} ./bin/lodepng.o ./src/test_equality.c -o test_equality -lm

lodepng:
	gcc -I ${dir_inc} -c ./src/lodepng.c
	mv ./lodepng.o ${dir_bin}

parallel_compute: bin lodepng
	nvcc -O3 -I ${dir_inc} ./bin/lodepng.o ./src/parallel_compute.cu
	#nvcc -I ${dir_inc} -L ${dir_cuda} ./bin/lodepng.o -c ./src/parallel_compute.cu
	#mv ./parallel_compute.o ${dir_bin}

rectify: parallel_compute
	nvcc -I ${dir_inc} -L ${dir_cuda} -c ./src/rectification.c
	mv ./rectification.o ${dir_bin}
	ncvv -I ${dir_inc} -L ${dir_cuda} -lcudart ./bin/parallel_compute.o ./bin/rectification.o -o rectify

loadpng_test: bin lodepng
	gcc -I ${dir_inc} ./bin/lodepng.o ./src/lodepng_test.c -o loadpng_test

bin:
	mkdir ./bin

clean:
	rm ./bin/*
