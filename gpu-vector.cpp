
#include <stdlib.h>
#include "hip/hip_runtime.h"

#define toDevice hipMemcpyHostToDevice
#define fromDevice hipMemcpyDeviceToHost

__global__ void vector_euclidian_distance(int width, int n, double *a, double *b, double *c)
{

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < n)
    {

        double accumulator = 0;

        int i;

        for (i = 0; i < w; i++)
        {
            accumulator = 0;
        }
    }
}

int *generate_random_int_vector(int width, int num_vectors)
{

    int *vectors = (int *)malloc(width * num_vectors * sizeof(int));

    //fill with random values

    for (int i = 0; i < num_vectors; i++)
    {
        for (int j = 0; j < width; j++)
        {
            *(vectors + i * width + j) = rand();
        }
    }

    return vectors;
}

void print_int_vector(int *vector, int width, int num_vectors)
{

    for (int i = 0; i < num_vectors; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("{%d,%d} = %d\n", i, j, *(vector + i * width + j));
        }
    }
}

int main(int argc, char *argv[])
{

    const int BLOCK_SIZE = 1024;
    const int VECTOR_WIDTH = 10;
    const int NUMBER_VECTORS = 2048;
    const int NUM_GRIDS = (NUMBER_VECTORS / BLOCK_SIZE) + 1;

    //generate the input & target vector
    int *host_vectors = generate_random_int_vector(VECTOR_WIDTH, NUMBER_VECTORS);
    print_int_vector(host_vectors, VECTOR_WIDTH, NUMBER_VECTORS);

    int *host_target = generate_random_int_vector(VECTOR_WIDTH, 1);
    print_int_vector(host_target, VECTOR_WIDTH, 1);

    //container for the output
    double *host_output = (double *)malloc(NUMBER_VECTORS * sizeof(double));

    //device IO, allocate & copy
    int *device_vectors;
    int *device_target;
    double *device_output;

    int vectors_bytes = VECTOR_WIDTH * NUMBER_VECTORS * sizeof(int);
    int target_bytes = VECTOR_WIDTH * sizeof(int);
    int output_bytes = NUMBER_VECTORS * sizeof(double);

    hipMalloc(&device_vectors, vectors_bytes);
    hipMalloc(&device_target, target_bytes);
    hipMalloc(&device_output, output_bytes);

    hipMemcpy(device_vectors, host_vectors, vectors_bytes, toDevice);
    hipMemcpy(device_vectors, host_vectors, vectors_bytes, toDevice);

    // Launch the kernel
    hipLaunchKenerlGGL(vector_euclidian_distance, dim3(NUM_GRIDS), dim3(BLOCK_SIZE), 0, 0);
    hipDeviceSynchronize();

    //Clear up
    free(host_vectors);
    free(host_target);
    free(host_output);

    free(device_vectors);
    free(device_target);
    free(device_output);
}