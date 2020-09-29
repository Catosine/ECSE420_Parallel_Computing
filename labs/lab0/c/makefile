inc='./inc'
src='./src'
bin='./bin'

loadpng_test: bin
	gcc -I ${inc} -c ./src/lodepng.c
	gcc -I ${inc} ./lodepng.o ./src/lodepng_test.c -o loadpng_test
	mv ./lodepng.o ${bin}


bin:
	mkdir ./bin

clean:
	rm ./bin/*
