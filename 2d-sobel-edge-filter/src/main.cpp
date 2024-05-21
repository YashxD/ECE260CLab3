#include <iostream>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "filter.h"
#include "timer.h"

using namespace std;
using namespace cv;

enum SobelType
{
	SOBEL_OPENCV,
	SOBEL_CPU,
	SOBEL_GPU
};


int main(int argc, const char * argv[])
{
  //Uncomment the following line to use the external camera.
	//VideoCapture cap(1);

  //Comment this line if you're using the external camera.
  VideoCapture cap("input.raw");

	int WIDTH  = 768;
	int HEIGHT = 768;

	SobelType sobel_type = SOBEL_GPU;

	// 1 argument on command line: WIDTH = HEIGHT = arg
	if(argc >= 2) {
		WIDTH = atoi(argv[1]);
		HEIGHT = WIDTH;
	}
	// 2 arguments on command line: WIDTH = arg1, HEIGHT = arg2
	if(argc >= 3) {
		HEIGHT = atoi(argv[2]);
	}

	// 3 arguments on command line: WIDTH = arg1, HEIGHT = arg2, type = arg3
	if(argc >= 4) {
		sobel_type = static_cast<SobelType>(atoi(argv[3]));
	}

	switch(sobel_type) {
		case SOBEL_OPENCV:
			cout << "Using OpenCV" << endl;
			break;
		case SOBEL_CPU:
			cout << "Using CPU" << endl;
			break;
		case SOBEL_GPU:
			cout << "Using GPU" << endl;
			break;
	}

	// Profiling
	LinuxTimer timer;
	LinuxTimer fps_counter;
	double time_elapsed = 0;

	// Allocate memory
	unsigned char* gray_ptr;
	unsigned char* sobel_out_ptr;

	cudaMallocManaged(&gray_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
	cudaMallocManaged(&sobel_out_ptr, WIDTH*HEIGHT*sizeof(unsigned char));

	Mat gray      = Mat(HEIGHT, WIDTH, CV_8U, gray_ptr);
	Mat sobel_out = Mat(HEIGHT, WIDTH, CV_8U, sobel_out_ptr);

	// More declarations
	Mat frame, s_x, s_y;

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

		// OpenCV Sobel
		timer.start();
		switch(sobel_type)
		{
			case SOBEL_OPENCV:
				Sobel(gray, s_x, CV_8U, 1, 0, 3, 1, 0, BORDER_ISOLATED);
				Sobel(gray, s_y, CV_8U, 0, 1, 3, 1, 0, BORDER_ISOLATED);
				addWeighted(s_x, 0.5, s_y, 0.5, 0, sobel_out);
				break;

			case SOBEL_CPU:
				sobel_out = (Scalar)0;
				sobel_filter_cpu(
						gray.ptr<uchar>(),
						sobel_out.ptr<uchar>(),
						gray.rows, gray.cols);
				break;

			case SOBEL_GPU:
				sobel_out = (Scalar)0;
				sobel_filter_gpu(
						gray.ptr<uchar>(),
						sobel_out.ptr<uchar>(),
						gray.rows, gray.cols);
				break;
		}
		timer.stop();

		size_t time_sobel = timer.getElapsed();

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
			imshow("Sobel", sobel_out);
			if(count <= 1) { moveWindow("Sobel", WIDTH, 0); }
		}

		key = waitKey(1);
	}

	cudaFree(gray_ptr);
	cudaFree(sobel_out_ptr);

	return 0;
}
