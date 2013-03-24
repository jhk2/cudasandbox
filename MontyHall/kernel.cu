
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <time.h>

__global__ void mhNoSwitch(unsigned int *randNums, bool *results)
{
	unsigned int idx = blockIdx.x * 1024 + threadIdx.x;
	unsigned int rand0 = randNums[2*idx];
	unsigned int rand1 = randNums[2*idx+1];
	// door with prize
	unsigned int prize = rand0 % 3;
	// door choice
	unsigned int choice = rand1 % 3;
	// since choice does not change after extra door opening, just check if choice and prize door are the same
	results[idx] = (prize == choice);
}

__global__ void mhSwitch(unsigned int *randNums, bool *results)
{
	unsigned int idx = blockIdx.x * 1024 + threadIdx.x;
	unsigned int rand0 = randNums[2*idx];
	unsigned int rand1 = randNums[2*idx+1];

	// door with prize
	unsigned int prize = rand0 % 3;
	// initial choice
	unsigned int choice = rand1 % 3;

	unsigned int reveal = 0;
	unsigned int final = 0;
	if (prize == 0) {
		if (choice == 0) {
			reveal = 1 + rand0 % 2;
			final = 3 - reveal;
		} else if (choice == 1) {
			reveal = 2;
			final = 0;
		} else { // choice == 2
			reveal = 1;
			final = 0;
		}
	} else if (prize == 1) {
		if (choice == 0) {
			reveal = 2;
			final = 1;
		} else if (choice == 1) {
			reveal = 2 * (rand0 % 2);
			final = 2 - reveal;
		} else { // choice == 2
			reveal = 0;
			final = 1;
		}
	} else if (prize == 2) {
		if (choice == 0) {
			reveal = 1;
			final = 2;
		} else if (choice == 1) {
			reveal = 0;
			final = 2;
		} else { // choice == 2
			reveal = rand0 % 2;
			final = 1 - reveal;
		}
	}

	results[idx] = (prize == final);
}

int main()
{
	cudaSetDevice(0);

	const unsigned int numBlocks = 10;
	const unsigned int numTests = 1024 * numBlocks;
	bool results_noswitch[numTests];
	bool results_switch[numTests];
	bool *dev_results_noswitch = 0;
	bool *dev_results_switch = 0;
	unsigned int *dev_randNums = 0;

	cudaMalloc((void**) &dev_results_noswitch, numTests * sizeof(bool));
	cudaMalloc((void**) &dev_results_switch, numTests * sizeof(bool));
	cudaMalloc((void**) &dev_randNums, 2 * numTests * sizeof(unsigned int));

	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
	curandGenerate(generator, dev_randNums, 2 * numTests);

	mhNoSwitch<<<numBlocks, 1024>>>(dev_randNums, dev_results_noswitch);
	mhSwitch<<<numBlocks, 1024>>>(dev_randNums, dev_results_switch);

	cudaMemcpy(results_noswitch, dev_results_noswitch, numTests * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(results_switch, dev_results_switch, numTests * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(dev_results_noswitch);
	cudaFree(dev_results_switch);
	curandDestroyGenerator(generator);

	unsigned int noswitchpass = 0;
	unsigned int switchpass = 0;
	for (int i = 0; i < numTests; i++) {
		noswitchpass += results_noswitch[i] ? 1 : 0;
		switchpass += results_switch[i] ? 1 : 0;
	}

	printf("no switch picked correctly %i out of %i, which is %g percent\n", noswitchpass, numTests, 100 * ((float) noswitchpass) / ((float) numTests));
	printf("switch picked correctly %i out of %i, which is %g percent\n", switchpass, numTests, 100 * ((float) switchpass) / ((float) numTests));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
