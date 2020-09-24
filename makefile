bin:
	mkdir ./bin

build_sample_cuda: bin
	nvcc ./src/playground/playground_cuda.cu -o ./bin/sample_cuda

clean:
	rm -r ./bin/*
