parallel_unified: bin reader
	nvcc -I ./inc ./bin/reader.o ./src/parallel_unified.cu -o parallel_unified

parallel_explicit: bin reader
	nvcc -I ./inc ./bin/reader.o ./src/parallel_explicit.cu -o parallel_explicit

compareResults:
	gcc ./src/compareResults.c -o compareResults

reader: bin
	nvcc ./src/reader.cu -c
	mv ./reader.o ./bin

bin:
	mkdir ./bin
