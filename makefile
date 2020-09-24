PROJ='playground'

bin:
	mkdir ./bin

build: bin
ifeq ($(PROJ), 'playground')
	nvcc ./src/playground/playground_cuda.cu -o ./bin/sample_cuda
else	
	echo "Unimplemented project: $(PROJ)"
endif

clean:
	rm -r ./bin/*
