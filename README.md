# Exp 3 Sobel edge detection filter using CUDA to enhance the performance of image processing tasks

### ENTER YOUR NAME: Vignesh M
### REGISTER NO: 212223240176

## Background: 
  - The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. 
  - This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

## Aim
To utilize CUDA to parallelize the Sobel filter implementation for efficient processing of images.

## Tools Required:
- A system with CUDA-capable GPU.
- CUDA Toolkit and OpenCV installed.

## Procedure

1. **Environment Setup**:
   - Ensure that CUDA and OpenCV are installed and set up correctly on your system.
   - Have a sample image (`images.jpg`) available in the correct directory to use as input.

2. **Load Image and Convert to Grayscale**:
   - Use OpenCV to read the input image in color mode.
   - Convert the image to grayscale as the Sobel operator works on single-channel images.

3. **Initialize and Allocate Memory**:
   - Determine the width and height of the grayscale image.
   - Allocate memory on both the host (CPU) and device (GPU) for the image data. Allocate device memory using `cudaMalloc` and check for successful allocation with `checkCudaErrors`.

4. **Performance Analysis Function**:
   - Define `analyzePerformance`, a function to test the CUDA kernel with different image sizes and block configurations.
   - For each specified image size (e.g., 256x256, 512x512, 1024x1024), set up the grid and block dimensions.
   - Launch the Sobel kernel using different block sizes (8x8, 16x16, 32x32) to evaluate the performance impact of each configuration. Record the execution time using CUDA events.

5. **Run Sobel Filter on Original Image**:
   - Set up the grid and block dimensions for the input image based on a 16x16 block size.
   - Use CUDA events to measure execution time for the Sobel filter applied to the original image.
   - Copy the resulting data from device memory to host memory.

6. **Save CUDA Output Image**:
   - Convert the processed image data on the host back to an OpenCV `Mat` object.
   - Save the CUDA-processed output image as `output_sobel_cuda.jpeg`.

7. **Compare with OpenCV Sobel Filter**:
   - For comparison, apply the OpenCV Sobel filter to the grayscale image on the CPU.
   - Measure the execution time using `std::chrono` for the CPU-based approach.
   - Save the OpenCV output as `output_sobel_opencv.jpeg`.

8. **Display Results**:
   - Print the input and output image dimensions.
   - Print the execution time for the CUDA Sobel filter and the CPU (OpenCV) Sobel filter to compare performance.
   - Display the breakdown of times for each block size and image size tested.

9. **Cleanup**:
   - Free all dynamically allocated memory on the host and device to avoid memory leaks.
   - Destroy CUDA events created for timing.

## Program

```cpp
%%writefile sobel_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,
                            unsigned int width, unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

        int sumX = 0, sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                unsigned char pixel = srcImage[(y + i) * width + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        int magnitude = sqrtf(float(sumX * sumX + sumY * sumY));
        magnitude = min(max(magnitude, 0), 255);
        dstImage[y * width + x] = (unsigned char)magnitude;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {

    Mat image = imread("creative2.jpg", IMREAD_COLOR);
    if (image.empty()) {
        printf("Error: Image not found at /content/image.jpg\n");
        return -1;
    }

    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    int width = grayImage.cols;
    int height = grayImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);

    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, grayImage.data, imageSize, cudaMemcpyHostToDevice));

    // Kernel configuration
    dim3 blockDim(16, 16);
    dim3 gridSize((width + blockDim.x - 1) / blockDim.x,
                  (height + blockDim.y - 1) / blockDim.y);

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockDim>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cudaTime = 0;
    cudaEventElapsedTime(&cudaTime, start, stop);

    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("/content/output_sobel_cuda.jpg", outputImage);

    // OpenCV Sobel timing
    Mat opencvOutput;
    auto startCpu = std::chrono::high_resolution_clock::now();
    Sobel(grayImage, opencvOutput, CV_8U, 1, 1, 3);
    auto endCpu = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(endCpu - startCpu).count();

    imwrite("/content/output_sobel_opencv.jpg", opencvOutput);

    printf("Image Size: %d x %d\n", width, height);
    printf("CUDA Sobel Time: %f ms\n", cudaTime);
    printf("OpenCV Sobel Time: %f ms\n", cpuTime);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    free(h_outputImage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

## Output Explanation

| Original 	|  Output using Cuda |
|:-:	|:-:	|
| ![creative2](https://github.com/user-attachments/assets/4ebd5792-49d3-4302-8363-0948d56ea11d) | <img width="513" height="401" alt="image" src="https://github.com/user-attachments/assets/d2c51980-937b-4288-9ed7-e97578c7414c" /> |

| Original 	|  Output using OpenCV |
|:-:	|:-:	|
| ![creative2](https://github.com/user-attachments/assets/161d1698-49cd-4808-8423-bb55aad3f4fb) | <img width="513" height="338" alt="image" src="https://github.com/user-attachments/assets/969b837c-a34f-45b5-a15b-e9f673de5663" /> |

- **Sample Execution Results**:
  - **CUDA Execution Times (Sobel filter) AND OpenCV Execution Time**
  </br>
<img width="250" height="56" alt="image" src="https://github.com/user-attachments/assets/bc42154f-6124-4fdf-9129-d7e53a82377c" />

- **Graph Analysis**:
  - Displayed a graph showing the relationship between image size, block size, and execution time.
 </br>

<img width="585" height="468" alt="image" src="https://github.com/user-attachments/assets/1cf9bb69-5174-4af7-99f7-abb590647d89" />


## Answers to Questions

1. **Challenges Implementing Sobel for Color Images**:
   - Converting images to grayscale in the kernel increased complexity. Memory management and ensuring correct indexing for color to grayscale conversion required attention.

2. **Influence of Block Size**:
   - Smaller block sizes (e.g., 8x8) were efficient for smaller images but less so for larger ones, where larger blocks (e.g., 32x32) reduced overhead.

3. **CUDA vs. CPU Output Differences**:
   - The CUDA implementation was faster, with minor variations in edge sharpness due to rounding differences. CPU output took significantly more time than the GPU.

4. **Optimization Suggestions**:
   - Use shared memory in the CUDA kernel to reduce global memory access times.
   - Experiment with adaptive block sizes for larger images.

## Result
Successfully implemented a CUDA-accelerated Sobel filter, demonstrating significant performance improvement over the CPU-based implementation, with an efficient parallelized approach for edge detection in image processing.
