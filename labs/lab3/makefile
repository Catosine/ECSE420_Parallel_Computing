global_queuing:
	nvcc -I ./c/inc ./c/global_queuing.cu -o ./global_queuing

output:
	mkdir ./output

clean_global_queuing:
	rm ./global_queuing
	rm ./output/gq_nodeOutput.raw ./output/gq_nextLevelNode.raw

run_global_queuing: global_queuing output
	./global_queuing ./input1.raw ./input2.raw ./input3.raw ./input4.raw ./output/gq_nodeOutput.raw ./output/gq_nextLevelNode.raw
