#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define PIXEL_GROUP 4

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int* img, int pitch, int resX, int resY, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int blockX = blockIdx.x * blockDim.x * PIXEL_GROUP;
    int blockY = blockIdx.y * blockDim.y * PIXEL_GROUP;


    for (int groupX = 0; groupX < PIXEL_GROUP; ++groupX) {
        for (int groupY = 0; groupY < PIXEL_GROUP; ++groupY) {
            int x = blockX + threadIdx.x * PIXEL_GROUP + groupX;
            int y = blockY + threadIdx.y * PIXEL_GROUP + groupY;

            if (x >= resX || y >= resY) {
                continue; // Skip the pixels outside the image boundaries
            }

            float c_re = lowerX + x * stepX;
            float c_im = lowerY + y * stepY;
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

            int* row = (int*)((char*)img + y * pitch);
            row[x] = i;
        }
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int* d_img;
    size_t pitch;
    // cudaHostAlloc((void**)&img, img_size, cudaHostAllocDefault);
    cudaMallocPitch((void**)&d_img, &pitch, resX * sizeof(int), resY);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((resX + PIXEL_GROUP * dimBlock.x - 1) / (PIXEL_GROUP * dimBlock.x), 
                 (resY + PIXEL_GROUP * dimBlock.y - 1) / (PIXEL_GROUP * dimBlock.y));

    mandelKernel<<<dimGrid, dimBlock>>>(lowerX, lowerY, stepX, stepY, d_img, pitch, resX, resY, maxIterations);

    int* h_img;
    cudaHostAlloc((void**)&h_img, pitch * resY, cudaHostAllocDefault);
    cudaMemcpy2D(h_img, pitch, d_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);

    // Copy row by row from h_img to img considering the pitch
    for (int y = 0; y < resY; y++) {
        memcpy(img + y * resX, (char*)h_img + y * pitch, resX * sizeof(int));
    }

    cudaFree(d_img);
    cudaFreeHost(h_img);
}
