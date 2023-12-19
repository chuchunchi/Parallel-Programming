#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int* img, int pitch, int resX, int resY, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < maxIterations; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    if (thisX < resX && thisY < resY) {
        int* row = (int*)((char*)img + thisY * pitch);
        row[thisX] = i;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int* d_img;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, resX * sizeof(int), resY);


    dim3 dimBlock(16, 16);
    dim3 dimGrid((resX + dimBlock.x - 1) / dimBlock.x, (resY + dimBlock.y - 1) / dimBlock.y);

    mandelKernel<<<dimGrid, dimBlock>>>(lowerX, lowerY, stepX, stepY, d_img, pitch, resX, resY, maxIterations);
    
    int* h_img;
    cudaHostAlloc((void**)&h_img, resY * pitch, cudaHostAllocDefault);
    cudaMemcpy2D(h_img, pitch, d_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    for (int y = 0; y < resY; y++) {
        memcpy(img + y * resX, (char*)h_img + y * pitch, resX * sizeof(int));
    }
    
    cudaFree(d_img);
    cudaFreeHost(h_img);
}
