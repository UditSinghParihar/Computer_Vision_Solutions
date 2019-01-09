#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;


void get_mask(const Mat3b& frame, Mat1b& mask, const float threshold){
	for(int y=0; y<frame.rows; ++y){
		for(int x=0; x<frame.cols; ++x){
			float green_value = frame.at<Vec3b>(y, x)[1];
			if(green_value > threshold){
				mask(y, x) = 0;
			}
			else{
				mask(y, x) = 255;
			}	
		}
	}
}

void get_new_image(const Mat3b& old_image, const Mat3b& background_image, 
					Mat3b& new_image, const Mat1b& mask){
	const cv::Vec3b blue_color{255, 0, 0};

	for(int y=0; y<mask.rows; ++y){
		for(int x=0; x<mask.cols; ++x){
			if(mask(y, x) == 255){
				new_image(y, x) = old_image(y, x);
			}
			else{
				new_image(y, x) = background_image(y, x);
			}
		}
	}
}

void play_video(const string& original_video, const string& background_video){
	VideoCapture capture1(original_video);
	VideoCapture capture2(background_video);

	if(!capture1.isOpened() || !capture2.isOpened()){
		cout << "Failed to open video files\n";
		return ;
	}
	
	Mat original_frame;
	Mat background_frame;

	const string window_name = "Video_frames";
	namedWindow(window_name, WINDOW_KEEPRATIO);
	const float fps = capture1.get(CAP_PROP_FPS);
	
	for(int i=0; ; ++i){
		capture1 >> original_frame;
		capture2 >> background_frame; 

		if(original_frame.empty() || background_frame.empty()){
			break;
		}

		Mat1b mask = cv::Mat1b::zeros(original_frame.size());
		get_mask(original_frame, mask, 200);

		cv::Mat3b new_image(original_frame.size());
		get_new_image(original_frame, background_frame, new_image, mask);

		imshow(window_name, new_image);
		waitKey(1000/fps);
	}		
}

int main(int argc, char const *argv[]){
	if(argc != 3){
		fprintf(stdout, "Usage: %s original_video.mp4 background_video.mp4\n", 
				argv[0]);
		return 1;
	}

	play_video(argv[1], argv[2]);
	return 0;
}