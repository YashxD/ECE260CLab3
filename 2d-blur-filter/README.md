#Assignment 5 - Due date: 09/1/2017 07:59:00
In this assignment, you will learn more about GPU programming. You will need to develop and run your code on the Jetson TX2 board.

## Part 1: Blur Filter
This is a continuation from the lab and is straight forward. It's a simpler version of the sobel filter in part 2. 

1. Complete the kernel_filter code in filter.cu to produce a 5x5 blurring of the video
2. Complete the grid size and block size when launching the kernel

## Part 2: Sobel filter
For this part, you can either use a camera or use the input.raw video that is provided. Do not use the integrated camera on the board.

Complete the filter.cu file to encode a sobel filter on the GPU.

1. Complete the grid size and block size variables in sobel_filter_gpu. Use X and Y dimensions. You can use the provided divup function to calculate the grid size.
2. Complete the kernel. You can reuse most of the CPU sobel filter function. Note: the output has already been initialized to 0.
3. Report the approximate FPS for OpenCV Sobel, CPU Sobel, and GPU Sobel, for different sizes. You can use square sizes from 512 to 4096 (note: your code should still work for non-square sizes). Note that for smaller sizes, the FPS will be limited by the camera FPS, and beyong 1024, the images will not display. If you wish, you can completely disable the display (comment out "imshow" in main.cpp) for all sizes to get a more stable result for GPU.

Examples:

Run with OpenCV:
```
./lab5 1024 1024 0
```
Run CPU code:

```
./lab5 1024 1024 1
```
Run GPU code:

```
./lab5 1024 1024 2
```

## Part 3: Blocked matrix multiplication
In this part, you will multiply two matrices using shared memory. Make sure you carefully read the description of the problem.

### Description
* You will perform C = AB where A, B, and C are matrices of float values.
* The size of A is NxM, the size of B is MxN, and the size of C is NxN. To clarify: A has N rows and M columns, B has M rows and N columns, C is square with N rows and N columns.
* M can be smaller, equal, or larger than N
* You must perform the multiplication using shared memory. We provide hints below to help you.
* The block size is the same on X and Y, and is defined as a constant. However, your code should still work if we change the provided block size value.
* You can assume that **M and N are multiple of the block size**. No need for edge cases.
* The RMSE should be small, below 0.001.

### Tasks
1. Complete block_mm_kernel.
2. Run your code for various sizes of M and N, from 16 to 1024. Report the speedup for each case (we expect at least 5-6 data points, showing the increase of the speedup, and for both M&lt>N and N&lt<M).

### Hints
* You will launch NxN threads (one per output), divided into blocks of size SxS.
* We will illustrate what happens inside each thread with the example below:

![](matmul.jpg)

* Let's take C11 as an example.
* C11 is a block of SxS threads. Inside the block C11, the threads will first load A11 and B11 into shared memory. Very simply, each thread can load one value from A11 and one value from B11.
* The threads need to synchronize, then perform A11 x B11. This means that each thread multiplies one row of A11 by one column of B11.
* Then threads synchronize and load A12 and B21 into shared memory. They synchronize and perform A12 x B21. This will repeat for all the blocks until (in this example) A16 and B61 (in reality, you do not have 6 blocks, but N/S blocks). Then the cumulative result is saved into C11 (each thread saves one value).
* All the other C blocks (C12, C13, etc.) perform the same task simultaneously.

# Work Submission
IMPORTANT: Follow the instructions exactly, and double check your repository, folder, and file names. You may lose points if you do not follow the exact instructions. If you have a doubt or a question, please ask the TAs, preferably on Piazza.

* All of your codes related to assignment 5 must be placed in a folder called Assignment_5 in your repository's source directory
* You have to submit these:
    * A short report named **Report.pdf** with your measurements and comments from part 2 and part 3. This should be at the root of the assignment folder.
    * All of your source code in the following format.
Do not submit other files than those listed, unless you have a good reason, and explain so in your report.

```
code/
    filter/
        Makefile
        input.raw
        include/
            cuda_timer.h
            filter.h
            timer.h
        src/
            filter.cu
            main.cpp
    sobel/
        Makefile
        input.raw
        include/
            cuda_timer.h
            filter.h
            timer.h
        src/
            filter.cu
            main.cpp
    matrix/
        Makefile
        include/
            cuda_error.h
            matrixmul.h
        src/
            matrixmul.cu
            main.cpp
```
