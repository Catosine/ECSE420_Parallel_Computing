#include <stdio.h>
#include <time.h>

int main(){
	printf("tic: %ld\nclock: %ld\n", CLOCKS_PER_SEC, clock());
}
