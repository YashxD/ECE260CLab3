#include <iostream>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "filter.h"
#include "timer.h"

using namespace std;
using namespace cv;


int main(int argc, const char * argv[])
{
	VideoCapture cap("input.raw"); 
	//VideoCapture cap(1); //output of ls /dev/video* will display available video devices and their indices

	int WIDTH  = 768;
	int HEIGHT = 768;

	// 1 argument on command line: WIDTH = HEIGHT = arg
	if(argc >= 2)
	{
		WIDTH = atoi(argv[1]);
		HEIGHT = WIDTH;
	}
	// 2 arguments on command line: WIDTH = arg1, HEIGHT = arg2
	if(argc >= 3)
	{
		HEIGHT = atoi(argv[2]);
	}


	// Profiling
	LinuxTimer timer;
	LinuxTimer fps_counter;
	double time_elapsed = 0;

	// Allocate memory
	unsigned char* gray_ptr;
	unsigned char* out_ptr;
	
	cudaMallocManaged(&gray_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
	cudaMallocManaged(&out_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
	
	Mat gray = Mat(HEIGHT, WIDTH, CV_8U, gray_ptr);
	Mat out  = Mat(HEIGHT, WIDTH, CV_8U, out_ptr);

	// More declarations
	Mat frame;

	char key = 0;
	int count = 0;


	// Main loop
	while(key != 'q')
	{
		// Get frame
		cap >> frame;

		// If no more frames, wait and exit
		if(frame.empty())
		{
			waitKey();
			break;
		}

		// Resize and grayscale
		resize(frame, frame, Size(WIDTH, HEIGHT));
		// cvtColor(frame, gray, CV_BGR2GRAY);
    cvtColor(frame, gray, COLOR_BGR2GRAY);

		// Run filter
		timer.start();
		out = (Scalar)0;
		filter_gpu(gray.ptr<uchar>(), out.ptr<uchar>(), gray.rows, gray.cols);
		timer.stop();

		size_t time_filter = timer.getElapsed();




		count++;

		// FPS count
		fps_counter.stop();
		time_elapsed += (fps_counter.getElapsed())/1000000000.0;
		fps_counter.start();

		if(count % 10 == 0)
		{
			double fps = 10/time_elapsed;
			time_elapsed = 0;
			cout << "FPS = " << fps << endl;
		}


		// Display results
		if(gray.cols <= 1024 || gray.rows <= 1024)
		{
			imshow("Input", gray);
			imshow("Filter", out);
			if(count <= 1) { moveWindow("Filter", WIDTH, 0); }
		}



		key = waitKey(1);
	}


	cudaFree(gray_ptr);
	cudaFree(out_ptr);

	return 0;
}
