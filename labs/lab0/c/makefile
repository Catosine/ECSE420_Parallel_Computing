dir_inc='./inc'
dir_src='./src'
dir_bin='./bin'

test_equality: bin lodepng
	gcc -I ${dir_inc} ./bin/lodepng.o ./src/test_equality.c -o test_equality -lm

lodepng:
	gcc -I ${dir_inc} -c ./src/lodepng.c
	mv ./lodepng.o ${dir_bin}

rectification: bin lodepng
	gcc -I ${dir_inc} ./bin/lodepng.o ./src/rectification.c -o rectification

loadpng_test: bin lodepng
	gcc -I ${dir_inc} ./bin/lodepng.o ./src/lodepng_test.c -o loadpng_test

bin:
	mkdir ./bin

clean:
	rm ./bin/*
