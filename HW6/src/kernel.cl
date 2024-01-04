__kernel void convolution(__constant float *filter, __global float *inputImage, __global float *outputImage,
                          int filterWidth, int imageHeight, int imageWidth)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int halfFilterSize = filterWidth / 2;
    float sum = 0.0;
    int currentX, currentY;

    // Apply the filter to the neighborhood
    for (int i = -halfFilterSize; i <= halfFilterSize; i++)
    {
        for (int j = -halfFilterSize; j <= halfFilterSize; j++)
        {
            currentX = x + j;
            currentY = y + i;

            // Check for boundary conditions
            if (currentX >= 0 && currentX < imageWidth && currentY >= 0 && currentY < imageHeight)
            {
                sum += inputImage[currentY * imageWidth + currentX] * 
                       filter[(i + halfFilterSize) * filterWidth + (j + halfFilterSize)];
            }
        }
    }

    // Write the sum to the output image
    outputImage[y * imageWidth + x] = sum;
}
