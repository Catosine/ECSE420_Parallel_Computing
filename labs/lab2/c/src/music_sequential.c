#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GRID_SIZE 512
#define MIU 0.0002
#define RHO 0.5
#define G 0.75

int simulation(float *grid, float *grid_1, float *grid_2)
{
    // compute center
    for(int y=1; y<GRID_SIZE-1; y++){
        for(int x=1; x<GRID_SIZE-1; x++){
            float u1_x_y = *(grid_1+GRID_SIZE*y+x);
            float u2_x_y = *(grid_2+GRID_SIZE*y+x);
            float u1_1x_y = *(grid_1+GRID_SIZE*y+(x+1));
            float u1_x1_y = *(grid_1+GRID_SIZE*y+(x-1));
            float u1_x_1y = *(grid_1+GRID_SIZE*(y+1)+x);
            float u1_x_y1 = *(grid_1+GRID_SIZE*(y-1)+x);
            *(grid+GRID_SIZE*y+x) = (RHO*(u1_x1_y + u1_1x_y + u1_x_y1 + u1_x_1y - 4*u1_x_y) + 2*u1_x_y - (1-MIU)*u2_x_y )/(1+MIU);
        }
    }

    // compute edges
    for(int i=1;i<GRID_SIZE-1;i++)
    {
        float temp = *(grid_1+GRID_SIZE*i+1);
        *(grid+GRID_SIZE*i) = G * temp;

        temp = *(grid_1+GRID_SIZE*i+GRID_SIZE-2);
        *(grid+GRID_SIZE*i+GRID_SIZE-1) = G * temp;

        temp = *(grid_1+GRID_SIZE+i);
        *(grid+i) = G * temp;

        temp = *(grid_1+GRID_SIZE*(GRID_SIZE-2)+i);
        *(grid+GRID_SIZE*(GRID_SIZE-1)+i) = G * temp;
    }

    // compute corner
    float temp = *(grid_1+GRID_SIZE);
    *(grid) = G * temp;

    temp = *(grid_1+GRID_SIZE-2);
    *(grid+GRID_SIZE-1) = G * temp;

    temp = *(grid_1+GRID_SIZE*(GRID_SIZE-2));
    *(grid+GRID_SIZE*(GRID_SIZE-1)) = G * temp;

    temp = *(grid_1+GRID_SIZE*(GRID_SIZE-2)+GRID_SIZE-1);
    *(grid+GRID_SIZE*(GRID_SIZE-1)+GRID_SIZE-1) = G * temp;

    memcpy(grid_2, grid_1, GRID_SIZE*GRID_SIZE*sizeof(float));
    memcpy(grid_1, grid, GRID_SIZE*GRID_SIZE*sizeof(float));

    return 0;

}

int print_grid(float *grid)
{
    for(int y=0; y<GRID_SIZE; y++){
        for(int x=0; x<GRID_SIZE; x++){
            printf("(%d,%d): %f ", y, x, *(grid+GRID_SIZE*y+x));
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}

int main(int argc, char* argv[])
{
    if(argc!=2)
    {
        printf("RuntimeError: Wrong inputs\n");
        printf("Correct Format: ./grid_%d_%d_sequential <number of iterations>\n", GRID_SIZE, GRID_SIZE);
        return 1;
    }

    int iter = atoi(argv[1]);

    float *grid = (float *)calloc(GRID_SIZE*GRID_SIZE, sizeof(float));
    float *grid_1 = (float *)calloc(GRID_SIZE*GRID_SIZE, sizeof(float));
    float *grid_2 = (float *)calloc(GRID_SIZE*GRID_SIZE, sizeof(float));

    *(grid_1+GRID_SIZE*(GRID_SIZE/2)+(GRID_SIZE/2)) = 1.0f;
    
    printf("Size of the grid: %d nodes\n", GRID_SIZE*GRID_SIZE);
    
    clock_t start, end;
    clock_t s = clock();
    int rt;
    for(int i = 0; i<iter; i++){
    	start = clock();
	simulation(grid, grid_1, grid_2);
        end = clock();
	rt += end - start;
	//print_grid(grid);
        printf("#%d (%d,%d): %f\n", i, GRID_SIZE/2, GRID_SIZE/2, *(grid+GRID_SIZE*(GRID_SIZE/2)+GRID_SIZE/2));
    	printf("Runtime for #%d: %f ms\n", i, (double)(end-start)*1000/CLOCKS_PER_SEC);
    }
    clock_t t = clock();

    printf("------------------------------\n");
    printf("Runtime for simluation: %f ms\n", (double)(t-s)*1000/CLOCKS_PER_SEC);
    printf("Avg. runtime per iteration %f ms\n", (double)(rt)*1000/(CLOCKS_PER_SEC*iter));
    grid=NULL;
    grid_1=NULL;
    grid_2=NULL;
    free(grid);
    free(grid_1);
    free(grid_2);

}
