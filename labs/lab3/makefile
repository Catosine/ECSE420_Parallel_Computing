sequential:
	gcc ./c/sequential.c -o sequential

run_sequential: sequential output
	./sequential ./input1.raw ./input2.raw ./input3.raw ./input4.raw ./output/sequential_nodeOutput.raw ./output/sequential_nextLevelNode.raw

global_queuing:
	nvcc -I ./c/inc ./c/global_queuing.cu -o ./global_queuing

block_queuing:
	nvcc -I ./c/inc ./c/block_queuing.cu -o ./block_queuing

output:
	mkdir ./output

clean_global_queuing:
	rm ./global_queuing
	rm ./output/gq_nodeOutput.raw ./output/gq_nextLevelNode.raw
	
clean_block_queuing:
	rm ./block_queuing
	rm ./output/bq_nodeOutput.raw ./output/bq_nextLevelNode.raw

run_block_queuing: block_queuing output
	./block_queuing 35 64 64 ./input1.raw ./input2.raw ./input3.raw ./input4.raw ./output/bq_nodeOutput.raw ./output/bq_nextLevelNode.raw

run_global_queuing: global_queuing output
	./global_queuing ./input1.raw ./input2.raw ./input3.raw ./input4.raw ./output/gq_nodeOutput.raw ./output/gq_nextLevelNode.raw
