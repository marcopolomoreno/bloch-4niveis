#ifndef	PROPDEVICE_H
#define PROPDEVICE_H
	
void propriedades()
{
	cudaDeviceProp prop;
	int count;

	cudaGetDeviceCount(&count);
	for (int s = 0; s < count; s++)
	{
		cudaGetDeviceProperties(&prop, s);
		printf("General information\n");
		printf("%s", prop.name);
		printf(", Compute Capability %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d MHz\n", int(prop.clockRate / 1000));
		printf("Kernel execition timeout:  ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else  printf("Disabled\n");
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d)\n", prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
		printf("Shared memory per MP: %ld kb\n", prop.sharedMemPerBlock / 1024);
		printf("Registers per MP: %d\n", prop.regsPerBlock);
		printf("Warp size: %ld\n\n", prop.warpSize);
	}
}
#endif