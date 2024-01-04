#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;

    // Size of the filter and image data in bytes
    int filterSize = filterWidth * filterWidth;
    size_t filterDataSize = sizeof(float) * filterSize;
    size_t imageDataSize = sizeof(float) * imageHeight * imageWidth;

    // Create memory buffers
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterDataSize, NULL, &status);
    cl_mem inputImageBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageDataSize, NULL, &status);
    cl_mem outputImageBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageDataSize, NULL, &status);


    cl_command_queue command_queue;
    command_queue = clCreateCommandQueue(*context, *device, 0, &status);
    if (status != CL_SUCCESS) {
        // Handle error
        printf("Error creating command queue\n");
    }   
    // Write data to buffer
    status = clEnqueueWriteBuffer(command_queue, filterBuffer, CL_TRUE, 0, filterDataSize, filter, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, inputImageBuffer, CL_TRUE, 0, imageDataSize, inputImage, 0, NULL, NULL);

    // Create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    // Set the kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&filterBuffer);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inputImageBuffer);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputImageBuffer);
    status = clSetKernelArg(kernel, 3, sizeof(int), (void *)&filterWidth);
    status = clSetKernelArg(kernel, 4, sizeof(int), (void *)&imageHeight);
    status = clSetKernelArg(kernel, 5, sizeof(int), (void *)&imageWidth);

    // Execute the kernel
    size_t global_item_size[] = {imageWidth, imageHeight};
    size_t local_item_size[] = {1, 1}; // Can be optimized based on the hardware
    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);

    // Read the output back to host
    status = clEnqueueReadBuffer(command_queue, outputImageBuffer, CL_TRUE, 0, imageDataSize, outputImage, 0, NULL, NULL);

    // Clean up
    clReleaseMemObject(filterBuffer);
    clReleaseMemObject(inputImageBuffer);
    clReleaseMemObject(outputImageBuffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
}
