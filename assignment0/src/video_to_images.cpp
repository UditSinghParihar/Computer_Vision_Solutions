#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

void extract_into_frames(const string& video_name){
	VideoCapture capture(video_name);
	if(!capture.isOpened()){
		cout << "Failed to open video file\n";
		return ;
	}
	
	Mat frame;
	const string window_name = "Video_frames";
	namedWindow(window_name, WINDOW_KEEPRATIO);
	const string image_prefix = "image";
	const string image_suffix = ".jpg";

	for(int i=0; ; ++i){
		capture >> frame;
		if(frame.empty()){
			break;
		}

		imshow(window_name, frame);
		waitKey(5);
		
		string image_name = image_prefix + to_string(i) + image_suffix;
		imwrite(image_name, frame);
		cout << "Saved " << image_name << endl;
	}		
}

int main(int argc, char const *argv[]){
	if(argc != 2){
		fprintf(stdout, "Usage: %s video.mp4\n", argv[0]);
		return 1;
	}

	extract_into_frames(argv[1]);

	return 0;
}