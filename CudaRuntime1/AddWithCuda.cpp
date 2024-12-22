#include "stdafx.h"

using namespace System;

#include "cuda_runtime.h"

cudaError_t addWithCuda(int* c, int const* a, int const* b, unsigned int size);

int main(array<System::String^>^ args)
{
    Console::WriteLine(L"Hello World");

    int const arraySize = 5;
    int const a[arraySize] = { 1, 2, 3, 4, 5 };
    int const b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"addWithCuda failed!");
        return 1;
    }

    Console::WriteLine(L"[1,2,3,4,5] + [10,20,30,40,50] = {0},{1},{2},{3},{4}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, int const* a, int const* b, unsigned int size)
{
    extern void addKernel(int* c, int const* a, const int* b);
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //  replace addKernel<<<1, size>>>(dev_c, dev_a, dev_b); with:
    void* args[] = { &dev_c, &dev_a, &dev_b };
    cudaStatus = cudaLaunchKernel(
        (void const*)&addKernel, // pointer to kernel func.
        dim3(1), // grid
        dim3(size), // block
        args  // arguments
    );

    // Check for any errors launching the kernel
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"addKernel launch failed: {0}\n", gcnew String(cudaGetErrorString(cudaStatus)));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"cudaDeviceSynchronize returned error code {0} after launching addKernel!\n", (int)cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        Console::Error->WriteLine(L"cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}