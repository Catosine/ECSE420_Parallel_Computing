sequential:
	gcc ./src/sequential.c -o sequential
	
parallel_unified: bin reader
	nvcc -I ./inc ./bin/reader.o ./src/parallel_unified.cu -o parallel_unified

parallel_explicit: bin reader
	nvcc -I ./inc ./bin/reader.o ./src/parallel_explicit.cu -o parallel_explicit

parallel_unified_prefetch: bin reader
	nvcc -I ./inc ./bin/reader.o ./src/parallel_unitifed_prefetch.cu -o parallel_unified_prefetch

compareResults:
	gcc ./src/compareResults.c -o compareResults

reader: bin
	nvcc ./src/reader.cu -c
	mv ./reader.o ./bin

bin:
	mkdir ./bin
