#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include "directory_parser.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]){
	if(argc != 4){
		fprintf(stdout, "Usage: %s images_directory output_video.avi fps\n", argv[0]);
		return 1;
	}

	const char* images_directory = argv[1];
	const string filename{argv[2]};
	double fps = atof(argv[3]);

	vector<string> rgb_images;
	ParseImages parser(images_directory, rgb_images);
	parser.start_processing();

	Mat first_image = imread(rgb_images[0], IMREAD_COLOR);

	int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
	cv::Size video_dimensions(first_image.cols, first_image.rows);

	VideoWriter writer{filename, codec, fps, video_dimensions};

	const string window_name = "Video_frames";

	for(int i=0; i<rgb_images.size(); ++i){
		Mat frame = imread(rgb_images[i], IMREAD_COLOR);
		writer.write(frame);

		imshow(window_name, frame);
		waitKey(1000.0/fps);
	}	

	return 0;
}